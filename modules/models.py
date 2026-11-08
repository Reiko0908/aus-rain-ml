"""Model factory: Logistic Regression, RBF-kernel SVM and CatBoost (gradient-boosted
trees, used either on the encoded matrix or with native categorical support)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .data_loader import CAT_COLS

DEFAULT_PARAMS = {
    "lr": {"C": 1.0, "max_iter": 5000},
    "svm": {"C": 1.0, "gamma": "scale"},
    "catboost": {"iterations": 500, "depth": 6, "learning_rate": 0.1},
    "catboost_native": {"iterations": 500, "depth": 6, "learning_rate": 0.1},
}


def _catboost(params):
    from catboost import CatBoostClassifier
    p = {"verbose": 0, "random_seed": 0, "thread_count": -1, "allow_writing_files": False}
    p.update(params)
    return CatBoostClassifier(**p)


def make_model(name: str, params: dict | None = None):
    p = dict(DEFAULT_PARAMS.get(name, {}))
    p.update(params or {})
    if name == "lr":
        return LogisticRegression(solver="lbfgs", random_state=0, **p)
    if name == "svm":
        return SVC(kernel="rbf", cache_size=1000, random_state=0, **p)
    if name == "catboost":
        return _catboost(p)
    if name == "catboost_native":
        return CatBoostNative(**p)
    raise ValueError(f"Unknown model {name}")


class CatBoostNative(BaseEstimator, ClassifierMixin):
    """CatBoost on the raw DataFrame: categorical columns are handled internally with
    ordered target statistics, numeric NaNs are handled natively. No encoding,
    scaling or PCA is needed (trees are invariant to monotonic rescaling)."""

    def __init__(self, iterations=500, depth=6, learning_rate=0.1, l2_leaf_reg=3.0,
                 cat_cols=tuple(CAT_COLS)):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.cat_cols = cat_cols

    def _prep(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_cols:
            X[c] = X[c].fillna("Missing").astype(str)
        return X

    def fit(self, X, y):
        X = self._prep(X)
        self.model_ = _catboost({"iterations": self.iterations, "depth": self.depth,
                                 "learning_rate": self.learning_rate,
                                 "l2_leaf_reg": self.l2_leaf_reg})
        self.model_.fit(X, y, cat_features=list(self.cat_cols))
        self.classes_ = np.array([0, 1])
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X)).astype(int).ravel()

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))

    @property
    def feature_importances_(self):
        return self.model_.get_feature_importance()
