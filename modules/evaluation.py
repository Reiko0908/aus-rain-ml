"""Metrics and evaluation plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import CLASS_NAMES
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def get_scores(model, X):
    """Continuous score for ROC-AUC (probability or SVM decision value)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # SVC without probability=True
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(float(v), 4) for k, v in m.items()}


def plot_confusion_matrices(results: dict, y_true, path=None, ncols=4):
    """results: {model_name: y_pred}"""
    n = len(results)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, y_pred) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=list(CLASS_NAMES.values()),
            normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
        ax.grid(False)
        ax.set_title(name, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(scores: dict, y_true, path=None):
    """scores: {model_name: y_score}"""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - test set")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_grid_heatmaps(df, metric="f1_macro", path=None):
    """One heatmap per model: rows = scaler, columns = PCA setting."""
    from . import plot_style as ps
    models = list(dict.fromkeys(df["model_label"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 2.9),
                             gridspec_kw={"wspace": 0.45})
    axes = np.atleast_1d(axes)
    vmin, vmax = df[metric].min(), df[metric].max()
    for ax, m in zip(axes, models):
        t = df[df.model_label == m].pivot(index="scaler_label", columns="pca_label", values=metric)
        t = t.loc[list(dict.fromkeys(df.scaler_label)), list(dict.fromkeys(df.pca_label))]
        d = df[df.model_label == m].pivot(index="scaler_label", columns="pca_label",
                                          values="n_features").loc[t.index, t.columns]
        im = ax.imshow(t.values, cmap=ps.SEQ_BLUE, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, f"{v:.4f}\n{int(d.values[i, j])} dims", ha="center", va="center",
                        fontsize=8, color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else ps.INK)
        ax.set_xticks(range(t.shape[1]), t.columns)
        ax.set_yticks(range(t.shape[0]), t.index)
        ax.grid(False)
        ax.set_title(m)
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"validation {metric}")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation(df, metric="f1_macro", path=None):
    """Horizontal bars of the change vs. each group's baseline."""
    from . import plot_style as ps
    groups = list(dict.fromkeys(df["group"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 0.42 * df.groupby("group").size().max() + 1.2))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        d = df[df.group == g].copy()
        base = d.loc[d.variant == "baseline", metric].iloc[0]
        d = d[d.variant != "baseline"]
        d["delta"] = d[metric] - base
        d = d.sort_values("delta")
        cols = [ps.SERIES[0] if v >= 0 else ps.SERIES[1] for v in d["delta"]]
        ax.barh(d["variant"], d["delta"], color=cols, height=0.6)
        for i, (dv, v) in enumerate(zip(d["delta"], d[metric])):
            ax.text(dv + (0.0007 if dv >= 0 else -0.0007), i, f"{dv:+.4f}", va="center",
                    ha="left" if dv >= 0 else "right", fontsize=7, color=ps.INK2)
        ax.axvline(0, color=ps.INK2, lw=1)
        lim = max(abs(d["delta"]).max() * 1.6, 0.005)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{g} (baseline {metric}={base:.4f})", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"change in validation {metric}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curve(history, path=None):
    from . import plot_style as ps
    h = __import__("pandas").DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    axes[0].plot(h.epoch, h.train_loss, label="train", color=ps.SERIES[0])
    axes[0].plot(h.epoch, h.val_loss, label="validation", color=ps.SERIES[1])
    axes[0].set_title("BCE loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(h.epoch, h.val_auc, color=ps.SERIES[1], marker="o", markersize=3)
    best = h.val_auc.idxmax()
    axes[1].scatter([h.epoch[best]], [h.val_auc[best]], s=60, color=ps.INK, zorder=3)
    axes[1].annotate(f"best epoch {int(h.epoch[best])}: {h.val_auc[best]:.4f}",
                     (h.epoch[best], h.val_auc[best]), textcoords="offset points",
                     xytext=(0, -60), ha="center", fontsize=8, color=ps.INK2,
                     arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
    axes[1].set_title("Validation ROC-AUC")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_importances(series_dict, top=12, path=None):
    """series_dict: {title: pd.Series(feature -> importance)} normalised to sum 1."""
    from . import plot_style as ps
    n = len(series_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 0.32 * top + 1.2))
    axes = np.atleast_1d(axes)
    for ax, (title, s) in zip(axes, series_dict.items()):
        s = (s / s.sum()).sort_values(ascending=False).head(top)[::-1]
        ax.barh(s.index, s.values, color=ps.SERIES[0], height=0.6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
