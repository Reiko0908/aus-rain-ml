"""Configurable preprocessing: imputation -> transform -> outlier handling -> encoding
-> scaling -> dimensionality reduction. Every step is selected by a config dict and
assembled into a single sklearn Pipeline, so all statistics are learned on the
training split only (no data leakage)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, QuantileTransformer, RobustScaler,
                                   StandardScaler, TargetEncoder)

from .data_loader import CAT_COLS, NUM_COLS, SKEWED_COLS

import sklearn
_SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

DEFAULT_CONFIG = {
    # --- imputation ---
    "num_imputer": "median",          # mean | median | knn | iterative
    "cat_imputer": "constant",        # constant ('Missing' category) | most_frequent
    # --- numeric transforms ---
    "log_skewed": False,              # log1p on heavily right-skewed columns
    "outliers": "none",               # none | iqr | quantile
    "iqr_factor": 1.5,
    "quantile_range": (0.01, 0.99),
    # --- encoding ---
    "encoder": "onehot",              # onehot | ordinal | target
    "onehot_min_frequency": 30,       # rarer categories are merged into 'infrequent'
    # --- scaling (applied to the full encoded matrix) ---
    "scaler": "standard",             # standard | minmax | robust | quantile | none
    "feature_range": (0, 1),          # used when scaler == minmax
    # --- dimensionality reduction ---
    "pca": None,                      # None | float in (0,1) = variance kept | int = n comps
}


def make_config(**overrides) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    unknown = set(overrides) - set(DEFAULT_CONFIG) - {"model", "model_params", "name",
                                                      "train_subsample"}
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    cfg.update(overrides)
    return cfg


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip (winsorise) numeric columns using bounds learned on the training data.

    method='iqr'      : [Q1 - k*IQR, Q3 + k*IQR]; columns with IQR == 0 are left
                        untouched (otherwise every non-zero value of a sparse count
                        column such as days_in_waiting_list would be clipped to 0).
    method='quantile' : [q_low, q_high] percentiles.
    """

    def __init__(self, method="iqr", factor=1.5, quantile_range=(0.01, 0.99)):
        self.method = method
        self.factor = factor
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.method == "iqr":
            q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            lo, hi = q1 - self.factor * iqr, q3 + self.factor * iqr
            keep = iqr == 0
            lo[keep], hi[keep] = -np.inf, np.inf
        elif self.method == "quantile":
            lo, hi = np.nanpercentile(X, [100 * q for q in self.quantile_range], axis=0)
        else:
            raise ValueError(self.method)
        self.lower_, self.upper_ = lo, hi
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _log_skewed(num_cols):
    idx = [num_cols.index(c) for c in SKEWED_COLS if c in num_cols]

    def f(X):
        X = np.array(X, dtype=float, copy=True)
        X[:, idx] = np.log1p(np.clip(X[:, idx], 0, None))
        return X
    return FunctionTransformer(f, feature_names_out="one-to-one")


def _num_imputer(name):
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=5),
        "iterative": IterativeImputer(max_iter=10, random_state=0),
    }[name]


def _cat_imputer(name):
    if name == "constant":
        return SimpleImputer(strategy="constant", fill_value="Missing")
    if name == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    raise ValueError(name)


def _encoder(cfg):
    name = cfg["encoder"]
    if name == "onehot":
        return OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=cfg["onehot_min_frequency"], sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "target":
        # cross-fitted target (mean) encoding: internal 5-fold CV avoids target leakage
        if _SKLEARN_VERSION >= (1, 9):   # newer API: pass the splitter
            return TargetEncoder(target_type="binary",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
        return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=0)
    raise ValueError(name)


def make_scaler(cfg):
    name = cfg["scaler"]
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler(feature_range=tuple(cfg["feature_range"]))
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=0)
    if name in (None, "none"):
        return "passthrough"
    raise ValueError(name)


def make_reducer(cfg):
    p = cfg["pca"]
    if p in (None, "none", 0):
        return "passthrough"
    return PCA(n_components=p, svd_solver="full", random_state=0)


def build_preprocessor(cfg, num_cols=NUM_COLS, cat_cols=CAT_COLS) -> Pipeline:
    """Everything except the model: column-wise cleaning/encoding, then global
    scaling and optional PCA."""
    num_steps = [("impute", _num_imputer(cfg["num_imputer"]))]
    if cfg["log_skewed"]:
        num_steps.append(("log1p", _log_skewed(list(num_cols))))
    if cfg["outliers"] != "none":
        num_steps.append(("clip", OutlierClipper(cfg["outliers"], cfg["iqr_factor"],
                                                 cfg["quantile_range"])))
    cat_steps = [("impute", _cat_imputer(cfg["cat_imputer"])), ("encode", _encoder(cfg))]
    columns = ColumnTransformer([
        ("num", Pipeline(num_steps), list(num_cols)),
        ("cat", Pipeline(cat_steps), list(cat_cols)),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("columns", columns),
        ("scale", make_scaler(cfg)),
        ("reduce", make_reducer(cfg)),
    ])
