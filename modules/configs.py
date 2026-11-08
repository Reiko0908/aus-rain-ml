"""Experiment definitions (kept in one place so the notebook stays readable)."""
from __future__ import annotations

SVM_SUBSAMPLE = 20_000  # RBF-SVM training cost grows ~quadratically with n

# Recommended configuration per model (the methodology being tested)
RECOMMENDED = {
    "LR": {"model": "lr", "scaler": "standard", "pca": None},
    "RBF-SVM": {"model": "svm", "scaler": "minmax", "feature_range": (0, 1), "pca": 0.95,
                "outliers": "quantile", "train_subsample": SVM_SUBSAMPLE},
    "CatBoost": {"model": "catboost_native"},
}


def scaler_pca_runs(svm_subsample=SVM_SUBSAMPLE):
    """Experiment 1: a short, targeted comparison (10 runs) of the scaler / PCA decision
    for each model: the recommended setting plus the alternatives that matter."""
    sv = {"model": "svm", "train_subsample": svm_subsample}
    return [
        # Logistic Regression: does scaling matter, does PCA help?
        {"name": "LR | StandardScaler | no PCA (recommended)", **RECOMMENDED["LR"]},
        {"name": "LR | MinMax(0,1) | no PCA", "model": "lr", "scaler": "minmax", "pca": None},
        {"name": "LR | StandardScaler | PCA 95%", "model": "lr", "scaler": "standard", "pca": 0.95},
        # RBF-SVM: distance-based -> scaling is essential; PCA is optional
        {"name": "RBF-SVM | clip + MinMax(0,1) | PCA 95% (recommended)",
         **RECOMMENDED["RBF-SVM"], "train_subsample": svm_subsample},
        {"name": "RBF-SVM | MinMax(0,1) | PCA 95% (no clipping)", **sv, "scaler": "minmax", "pca": 0.95},
        {"name": "RBF-SVM | clip + StandardScaler | PCA 95%", **sv, "scaler": "standard", "pca": 0.95,
         "outliers": "quantile"},
        {"name": "RBF-SVM | clip + MinMax(0,1) | no PCA", **sv, "scaler": "minmax", "pca": None,
         "outliers": "quantile"},
        # CatBoost: trees do not need scaling or PCA
        {"name": "CatBoost | native categoricals, no scaler, no PCA (recommended)",
         **RECOMMENDED["CatBoost"]},
        {"name": "CatBoost | one-hot, no scaler, no PCA", "model": "catboost", "scaler": "none",
         "pca": None},
        {"name": "CatBoost | one-hot, StandardScaler, PCA 95%", "model": "catboost",
         "scaler": "standard", "pca": 0.95},
    ]


def ablations(base: dict, label: str):
    """Experiment 2: change ONE preprocessing step at a time around `base`.
    This dataset has gaps in numeric AND categorical columns, so both kinds of imputer are compared."""
    variants = {
        "baseline": {},
        "cat_imputer=most_frequent": {"cat_imputer": "most_frequent"},
        "num_imputer=mean": {"num_imputer": "mean"},
        "num_imputer=iterative": {"num_imputer": "iterative"},
        "encoder=ordinal": {"encoder": "ordinal"},
        "encoder=target": {"encoder": "target"},
        "outliers=iqr": {"outliers": "iqr"},
        "outliers=quantile": {"outliers": "quantile"},
        "log1p skewed": {"log_skewed": True},
        "class_weight=balanced": {"model_params": {"class_weight": "balanced"}},
    }
    if base.get("train_subsample"):     # KNN imputation is O(n^2): only affordable on the subsample
        variants["num_imputer=knn"] = {"num_imputer": "knn"}
    out = []
    for vname, v in variants.items():
        if v.get("outliers") == base.get("outliers", "none"):
            continue  # identical to baseline
        out.append({**base, **v, "name": f"{label} | {vname}"})
    return out


# Experiment 3: hyper-parameter search spaces (3-fold stratified CV on train)
TUNING = {
    "LR": {"model__C": [0.01, 0.1, 1.0, 10.0],
           "model__class_weight": [None, "balanced"]},
    "RBF-SVM": {"model__C": [1.0, 4.0, 16.0],
                "model__class_weight": [None, "balanced"]},
    "CatBoost": {"model__depth": [6, 8],
                 "model__learning_rate": [0.05, 0.1],
                 "model__iterations": [500]},
}
