"""Deep learning pipeline: MLP with categorical embeddings (PyTorch).

Numeric features   -> median imputation + quantile transform to N(0,1)
Categorical feats  -> integer ids ('Missing' is its own category, rare levels share id 0)
                      -> one learned embedding table per column
[numeric | all embeddings] -> Linear-BN-ReLU-Dropout x 2 -> Linear -> logit

The 128-d output of the last hidden layer is exported as a learned feature vector."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from .data_loader import CAT_COLS, NUM_COLS

DEFAULT_MLP_CONFIG = {
    "hidden": [256, 128],
    "dropout": 0.2,
    "max_emb_dim": 16,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 60,
    "patience": 8,          # early stopping on validation ROC-AUC
    "min_cat_freq": 10,     # rarer categories share the "rare/unknown" embedding
    "seed": 0,
}


# --------------------------------------------------------------------------- data
class TabularPreprocessor:
    def __init__(self, num_cols=NUM_COLS, cat_cols=CAT_COLS, min_cat_freq=10):
        self.num_cols, self.cat_cols, self.min_cat_freq = list(num_cols), list(cat_cols), min_cat_freq

    @staticmethod
    def _new_qt():
        return QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                   subsample=10**9, random_state=0)

    def _impute(self, X):
        return X[self.num_cols].astype(float).fillna(
            pd.Series(self.medians_, index=self.num_cols)).values

    def fit(self, X: pd.DataFrame):
        self.medians_ = X[self.num_cols].astype(float).median().values
        self.qt_ = self._new_qt().fit(self._impute(X))
        self.vocab_ = {}
        for c in self.cat_cols:
            vc = X[c].fillna("Missing").astype(str).value_counts()
            keep = vc[vc >= self.min_cat_freq].index
            self.vocab_[c] = {v: i + 1 for i, v in enumerate(keep)}
        self.cardinalities_ = [len(self.vocab_[c]) + 1 for c in self.cat_cols]
        return self

    def transform(self, X: pd.DataFrame):
        x_num = self.qt_.transform(self._impute(X)).astype(np.float32)
        x_cat = np.stack([X[c].fillna("Missing").astype(str).map(self.vocab_[c]).fillna(0)
                          .astype(np.int64).values for c in self.cat_cols], axis=1)
        return x_num, x_cat

    # plain-Python / tensor state -> checkpoint loads under any numpy / sklearn version
    def get_state(self):
        return {"num_cols": self.num_cols, "cat_cols": self.cat_cols,
                "min_cat_freq": self.min_cat_freq,
                "medians": torch.tensor(self.medians_, dtype=torch.float64),
                "quantiles": torch.tensor(self.qt_.quantiles_, dtype=torch.float64),
                "references": torch.tensor(self.qt_.references_, dtype=torch.float64),
                "vocab": self.vocab_}

    @classmethod
    def from_state(cls, s):
        p = cls(s["num_cols"], s["cat_cols"], s["min_cat_freq"])
        p.medians_ = s["medians"].numpy()
        qt = cls._new_qt()
        qt.quantiles_ = s["quantiles"].numpy()
        qt.references_ = s["references"].numpy()
        qt.n_quantiles_ = len(qt.references_)
        qt.n_features_in_ = qt.quantiles_.shape[1]
        p.qt_ = qt
        p.vocab_ = s["vocab"]
        p.cardinalities_ = [len(p.vocab_[c]) + 1 for c in p.cat_cols]
        return p


# -------------------------------------------------------------------------- model
def emb_dim(cardinality, max_dim=16):
    """Rule of thumb (fast.ai): grows sub-linearly with the number of categories."""
    return int(min(max_dim, round(1.6 * cardinality ** 0.56)))


class EmbeddingMLP(nn.Module):
    def __init__(self, n_num, cardinalities, hidden=(256, 128), dropout=0.2, max_emb_dim=16, **_):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c, max_emb_dim)) for c in cardinalities])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, 1)
        self.d_input = n_num + sum(e.embedding_dim for e in self.embs)

    def forward(self, x_num, x_cat, return_hidden=False):
        e = [emb(x_cat[:, j]) for j, emb in enumerate(self.embs)]
        h = self.body(torch.cat([x_num] + e, dim=1))
        logit = self.head(h).squeeze(-1)
        return (logit, h) if return_hidden else logit


# ----------------------------------------------------------------------- training
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def _forward_all(model, x_num, x_cat, bs=8192):
    model.eval()
    dev = next(model.parameters()).device
    logits, hidden = [], []
    for b in _batches(len(x_num), bs, False):
        l, h = model(x_num[b].to(dev), x_cat[b].to(dev), return_hidden=True)
        logits.append(l.float().cpu())
        hidden.append(h.float().cpu())
    return torch.cat(logits).numpy(), torch.cat(hidden).numpy()


class MLPClassifier:
    """sklearn-style wrapper: fit(X_df, y, X_val, y_val) / predict / predict_proba."""

    def __init__(self, **config):
        self.config = {**DEFAULT_MLP_CONFIG, **config}

    def fit(self, X, y, X_val, y_val, verbose=True):
        c = self.config
        t_start = time.time()
        torch.manual_seed(c["seed"])
        np.random.seed(c["seed"])
        self.device_ = _device()
        self.prep_ = TabularPreprocessor(min_cat_freq=c["min_cat_freq"]).fit(X)
        xn, xc = map(torch.from_numpy, self.prep_.transform(X))
        vn, vc = map(torch.from_numpy, self.prep_.transform(X_val))
        yt = torch.tensor(y, dtype=torch.float32)
        self.model_ = EmbeddingMLP(xn.shape[1], self.prep_.cardinalities_, **c).to(self.device_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        gen = torch.Generator().manual_seed(c["seed"])
        best, best_state, wait = -np.inf, None, 0
        self.history_ = []
        for epoch in range(1, c["max_epochs"] + 1):
            t0 = time.time()
            self.model_.train()
            tot = 0.0
            for b in _batches(len(xn), c["batch_size"], True, gen):
                if len(b) < 2:          # BatchNorm needs >1 sample
                    continue
                logit = self.model_(xn[b].to(self.device_), xc[b].to(self.device_))
                loss = F.binary_cross_entropy_with_logits(logit, yt[b].to(self.device_))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            val_logit, _ = _forward_all(self.model_, vn, vc)
            val_auc = roc_auc_score(y_val, val_logit)
            val_loss = F.binary_cross_entropy_with_logits(
                torch.tensor(val_logit), torch.tensor(y_val, dtype=torch.float32)).item()
            self.history_.append({"epoch": epoch, "train_loss": tot / len(xn), "val_loss": val_loss,
                                  "val_auc": val_auc, "seconds": time.time() - t0})
            if verbose:
                print(f"epoch {epoch:>2}  train_loss={tot/len(xn):.4f}  val_loss={val_loss:.4f}  "
                      f"val_auc={val_auc:.4f}  ({time.time()-t0:.1f}s)")
            if val_auc > best + 1e-4:
                best, best_state, wait = val_auc, copy.deepcopy(self.model_.state_dict()), 0
            else:
                wait += 1
                if wait >= c["patience"]:
                    if verbose:
                        print(f"early stopping (best val_auc={best:.4f})")
                    break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmax([h["val_auc"] for h in self.history_]) + 1)
        self.train_seconds_ = time.time() - t_start
        return self

    def _tensors(self, X):
        return map(torch.from_numpy, self.prep_.transform(X))

    def predict_proba(self, X):
        logit, _ = _forward_all(self.model_, *self._tensors(X))
        p = 1 / (1 + np.exp(-logit))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def hidden_features(self, X):
        """Learned representation: output of the last hidden layer (one row per booking)."""
        _, h = _forward_all(self.model_, *self._tensors(X))
        return h.astype(np.float32)

    def permutation_importance(self, X, y, n_repeats=3, seed=0):
        """Drop in ROC-AUC when one column is shuffled (model-agnostic importance)."""
        rng = np.random.RandomState(seed)
        base = roc_auc_score(y, self.predict_proba(X)[:, 1])
        out = {}
        for col in self.prep_.num_cols + self.prep_.cat_cols:
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                drops.append(base - roc_auc_score(y, self.predict_proba(Xp)[:, 1]))
            out[col] = max(float(np.mean(drops)), 0.0)
        return pd.Series(out).sort_values(ascending=False)

    def n_parameters(self):
        return sum(p.numel() for p in self.model_.parameters())

    def save(self, path):
        torch.save({"state_dict": {k: v.cpu() for k, v in self.model_.state_dict().items()},
                    "config": self.config, "prep_state": self.prep_.get_state(),
                    "history": self.history_, "best_epoch": self.best_epoch_,
                    "train_seconds": float(self.train_seconds_)}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        clf = cls(**ck["config"])
        clf.device_ = _device()
        clf.prep_ = TabularPreprocessor.from_state(ck["prep_state"])
        clf.model_ = EmbeddingMLP(len(clf.prep_.num_cols), clf.prep_.cardinalities_, **clf.config)
        clf.model_.load_state_dict(ck["state_dict"])
        clf.model_.to(clf.device_).eval()
        clf.history_, clf.best_epoch_ = ck["history"], ck["best_epoch"]
        clf.train_seconds_ = ck["train_seconds"]
        return clf
