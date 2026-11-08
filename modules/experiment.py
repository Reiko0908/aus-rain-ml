"""Experiment runner: build pipeline from a config, fit, evaluate, log results,
save extracted feature matrices."""
from __future__ import annotations

import os
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluation import compute_metrics, get_scores
from .models import make_model
from .preprocessing import build_preprocessor, make_config


def build_pipeline(cfg: dict) -> Pipeline:
    """Full pipeline = preprocessing (from cfg) + model.
    CatBoost-native skips preprocessing entirely (handles raw categories / NaNs)."""
    model = make_model(cfg["model"], cfg.get("model_params"))
    if cfg["model"] == "catboost_native":
        return Pipeline([("model", model)])
    return Pipeline([("prep", build_preprocessor(cfg)), ("model", model)])


def _subsample(X, y, n, seed=0):
    if n is None or n >= len(y):
        return X, y
    rng = np.random.RandomState(seed)
    # stratified subsample
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], int(round(n * np.mean(y == c))), replace=False)
        for c in np.unique(y)])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def describe(cfg: dict) -> dict:
    return {
        "name": cfg.get("name", ""),
        "model": cfg["model"],
        "scaler": cfg["scaler"] if cfg["model"] != "catboost_native" else "none",
        "feature_range": str(tuple(cfg["feature_range"])) if cfg["scaler"] == "minmax" else "",
        "pca": cfg["pca"] if cfg["pca"] else "none",
        "encoder": cfg["encoder"] if cfg["model"] != "catboost_native" else "native",
        "cat_imputer": cfg["cat_imputer"],
        "num_imputer": cfg["num_imputer"],
        "outliers": cfg["outliers"],
        "log_skewed": cfg["log_skewed"],
        "model_params": str(cfg.get("model_params") or {}),
        "train_subsample": cfg.get("train_subsample") or "",
    }


def run_experiment(cfg: dict, X_train, y_train, X_eval, y_eval, return_model=False):
    cfg = make_config(**cfg)
    Xt, yt = _subsample(X_train, y_train, cfg.get("train_subsample"))
    pipe = build_pipeline(cfg)
    t0 = time.time()
    pipe.fit(Xt, yt)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = pipe.predict(X_eval)
    y_score = get_scores(pipe, X_eval)
    pred_time = time.time() - t0
    rec = describe(cfg)
    if "prep" in pipe.named_steps:
        prep = pipe.named_steps["prep"]
        rec["n_encoded"] = len(prep.named_steps["columns"].get_feature_names_out())
        red = prep.named_steps["reduce"]
        rec["n_features"] = int(red.n_components_) if red != "passthrough" else rec["n_encoded"]
    else:
        rec["n_encoded"] = rec["n_features"] = X_train.shape[1]
    rec.update(compute_metrics(y_eval, y_pred, y_score))
    rec["fit_s"] = round(fit_time, 2)
    rec["predict_s"] = round(pred_time, 2)
    if return_model:
        return rec, pipe, y_pred, y_score
    return rec


def run_grid(configs, X_train, y_train, X_eval, y_eval, csv_path=None, verbose=True):
    rows = []
    for i, cfg in enumerate(configs, 1):
        rec = run_experiment(cfg, X_train, y_train, X_eval, y_eval)
        rows.append(rec)
        if verbose:
            print(f"[{i:>2}/{len(configs)}] {rec['name']:<40} F1-macro={rec['f1_macro']:.4f} "
                  f"AUC={rec['roc_auc']:.4f} dims={rec['n_features']:<4} fit={rec['fit_s']:.1f}s")
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)


def save_features(path, prep, splits: dict, extra_attrs: dict | None = None):
    """Transform each split with a FITTED preprocessor and store the feature
    matrices (float32, gzip) plus labels and feature names in one .h5 file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for split_name, (X, y) in splits.items():
            Z = np.asarray(prep.transform(X), dtype=np.float32)
            f.create_dataset(f"X_{split_name}", data=Z, compression="gzip", compression_opts=6)
            f.create_dataset(f"y_{split_name}", data=np.asarray(y, dtype=np.int8))
        try:
            red = prep.named_steps["reduce"]
            if red != "passthrough":
                names = [f"PC{i+1}" for i in range(red.n_components_)]
            else:
                names = list(prep.named_steps["columns"].get_feature_names_out())
            f.attrs["feature_names"] = np.array(names, dtype="S")
        except Exception:
            pass
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = str(v)
    return path


def save_npy(dir_path, prefix, prep, splits: dict):
    """Same as save_features but as plain .npy files (one per split)."""
    os.makedirs(dir_path, exist_ok=True)
    paths = []
    for split_name, (X, y) in splits.items():
        Z = np.asarray(prep.transform(X), dtype=np.float32)
        p = os.path.join(dir_path, f"{prefix}_X_{split_name}.npy")
        np.save(p, Z)
        np.save(os.path.join(dir_path, f"{prefix}_y_{split_name}.npy"), np.asarray(y, np.int8))
        paths.append(p)
    return paths


def tune(cfg: dict, param_grid: dict, X, y, cv=3, scoring="f1_macro", verbose=True):
    """Grid search (stratified k-fold CV on the training data) over model
    hyper-parameters, with the preprocessing fixed by `cfg`."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    cfg = make_config(**cfg)
    Xs, ys = _subsample(X, y, cfg.get("train_subsample"))
    gs = GridSearchCV(build_pipeline(cfg), param_grid, scoring=scoring,
                      cv=StratifiedKFold(cv, shuffle=True, random_state=0),
                      n_jobs=1, refit=False)
    t0 = time.time()
    gs.fit(Xs, ys)
    res = pd.DataFrame(gs.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")]
    res = res[cols + ["mean_test_score", "std_test_score", "mean_fit_time"]]
    res.columns = [c.replace("param_model__", "") for c in res.columns]
    res = res.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    if verbose:
        print(f"{cfg.get('name', cfg['model'])}: best CV {scoring}={gs.best_score_:.4f} "
              f"with {best}  ({time.time()-t0:.0f}s)")
    return res, best
