"""Exploratory Data Analysis: descriptive statistics, missing values, distributions,
outliers, categorical effects, correlations and leakage checks."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_style as ps
from .data_loader import CLASS_NAMES, GROUP_COL, RAW_TARGET, TARGET

ps.apply()
LABELS = CLASS_NAMES


def _save(fig, out_dir, name):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    return fig


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, #unique, #missing, %missing, example values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "pct_missing": (100 * df.isna().mean()).round(2),
        "example": [", ".join(map(str, df[c].dropna().unique()[:3])) for c in df.columns],
    })


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").drop(columns=[TARGET], errors="ignore")
    s = num.describe().T
    s["skew"] = num.skew()
    return s.round(2)


def iqr_outliers(df: pd.DataFrame, factor=1.5) -> pd.DataFrame:
    num = df.select_dtypes("number").drop(columns=[TARGET], errors="ignore")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    n_out = ((num < lo) | (num > hi)).sum()
    out = pd.DataFrame({"Q1": q1, "Q3": q3, "IQR": iqr, "lower": lo, "upper": hi,
                        "n_outliers": n_out, "pct_outliers": (100 * n_out / len(num)).round(2),
                        "max": num.max()})
    return out[out.n_outliers > 0].sort_values("n_outliers", ascending=False).round(2)


def plot_missing(df: pd.DataFrame, out_dir=None):
    miss = (100 * df.isna().mean()).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(miss.index, miss.values, color=ps.SERIES[0], height=0.55)
    for b, v, n in zip(bars, miss.values, df[miss.index].isna().sum()):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.2f}%  ({n:,} rows)",
                va="center", fontsize=8, color=ps.INK2)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing values per column (raw data)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "eda_missing.png")


def plot_target(df_raw, df_clean, out_dir=None):
    """Class balance by RainToday, before and after removing the rows without a label."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (title, d) in zip(axes, [("Raw data", df_raw), ("After cleaning", df_clean)]):
        d = d.copy()
        if RAW_TARGET in d:
            d = d[d[RAW_TARGET].notna()]
            d["_y"] = (d[RAW_TARGET] == "Yes").astype(int)
        else:
            d["_y"] = d[TARGET]
        ct = pd.crosstab(d[GROUP_COL].fillna("unknown"), d["_y"])
        x = np.arange(len(ct))
        for k, col in enumerate([0, 1]):
            b = ax.bar(x + (k - 0.5) * 0.38, ct[col], width=0.36, label=LABELS[col],
                       color=[ps.NOT_CANCELED, ps.CANCELED][k])
            ax.bar_label(b, labels=[f"{v:,}" for v in ct[col]], fontsize=7, color=ps.INK2, padding=2)
        ax.set_xticks(x, [f"{GROUP_COL} = {v}" for v in ct.index])
        ax.set_title(f"{title}: {len(d):,} days, {d['_y'].mean():.1%} rainy tomorrow")
        ax.set_ylabel("Days")
        ax.grid(axis="x", visible=False)
    axes[0].legend(title="RainTomorrow")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_target.png")


def plot_numeric_distributions(df, cols, out_dir=None):
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows))
    for ax, c in zip(axes.ravel(), cols):
        for lab in [0, 1]:
            v = df.loc[df[TARGET] == lab, c].dropna()
            hi = df[c].quantile(0.995)
            ax.hist(v.clip(upper=hi), bins=40, alpha=0.55, density=True, label=LABELS[lab],
                    color=[ps.NOT_CANCELED, ps.CANCELED][lab])
        ax.set_title(f"{c}  (skew={df[c].skew():.1f})", fontsize=9)
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend()
    fig.suptitle("Numeric feature distributions by class (clipped at 99.5th percentile)",
                 fontsize=11, fontweight="bold", color=ps.INK)
    fig.tight_layout()
    return _save(fig, out_dir, "eda_numeric_dist.png")


def plot_boxplots(df, cols, out_dir=None):
    fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 3))
    for ax, c in zip(axes, cols):
        data = [df.loc[df[TARGET] == k, c].dropna() for k in [0, 1]]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, flierprops=dict(
            marker="o", markersize=2, alpha=0.3, markeredgecolor=ps.MUTED))
        for patch, col in zip(bp["boxes"], [ps.NOT_CANCELED, ps.CANCELED]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_edgecolor(ps.INK2)
        for m in bp["medians"]:
            m.set_color(ps.INK)
        ax.set_xticks([1, 2], ["No\nrain", "Rain"])
        ax.set_title(c, fontsize=9)
    fig.suptitle("Outliers: box plots by class (1.5 x IQR whiskers)", fontsize=11,
                 fontweight="bold", color=ps.INK)
    fig.tight_layout()
    return _save(fig, out_dir, "eda_boxplots.png")


def plot_categorical_rates(df, cols, out_dir=None, top=10):
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows))
    overall = df[TARGET].mean()
    for ax, c in zip(axes.ravel(), cols):
        s = df[c].fillna("Missing").astype(str)
        g = df.groupby(s)[TARGET].agg(["mean", "size"])
        g = g[g["size"] >= 30].sort_values("size", ascending=False).head(top)
        g = g.sort_values("mean")
        ax.barh(g.index, g["mean"], color=ps.SERIES[1], height=0.6)
        for i, (r, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(r + 0.01, i, f"{r:.0%} (n={n:,})", va="center", fontsize=7, color=ps.INK2)
        ax.axvline(overall, color=ps.INK2, ls="--", lw=1)
        ax.set_xlim(0, 1.25)
        ax.set_title(f"{c}", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Share of rainy next days by category (top {top} by frequency; dashed = overall "
                 f"{overall:.1%})", fontsize=11, fontweight="bold", color=ps.INK)
    fig.tight_layout()
    return _save(fig, out_dir, "eda_categorical_rates.png")


def plot_profiles(df, out_dir=None):
    """Share of rainy next days by calendar month and by 3 pm humidity."""
    g1 = df.groupby("month")[TARGET].agg(["mean", "size"])
    bins = pd.cut(df["Humidity3pm"], [-1, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    g2 = df.groupby(bins, observed=True)[TARGET].agg(["mean", "size"])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    axes[0].bar(months, g1["mean"], color=ps.SERIES[1], width=0.65)
    axes[0].set_title("Rainy next day by month (all stations)")
    axes[0].grid(axis="x", visible=False)
    axes[1].bar([str(i.right) for i in g2.index], g2["mean"], color=ps.SERIES[1], width=0.65)
    axes[1].set_title("Rainy next day by 3 pm humidity (band upper bound, %)")
    axes[1].grid(axis="x", visible=False)
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    return _save(fig, out_dir, "eda_profiles.png")


def plot_correlation(df, out_dir=None):
    num = df.select_dtypes("number")
    corr = num.corr()
    k = max(len(corr) * 0.42, 5)
    fig, ax = plt.subplots(figsize=(k + 1.5, k))
    im = ax.imshow(corr, cmap=ps.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("Correlation matrix of numeric features and the target")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_correlation.png")


def target_correlations(df) -> pd.Series:
    num = df.select_dtypes("number")
    return num.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3)


def missingness_check(df_raw, cols=("Sunshine", "Evaporation", "Cloud3pm", "Cloud9am", "Pressure9am",
                                    "WindGustDir")) -> pd.DataFrame:
    """Is a missing value informative? Share of rainy next days when a column is / is not missing."""
    d = df_raw[df_raw[RAW_TARGET].notna()]
    y = (d[RAW_TARGET] == "Yes").astype(int)
    out = {}
    for c in cols:
        m = d[c].isna()
        out[c] = {"n_missing": int(m.sum()), "pct_missing": round(100 * m.mean(), 1),
                  "rain | missing": round(y[m].mean(), 3), "rain | present": round(y[~m].mean(), 3)}
    return pd.DataFrame(out).T


def station_missing(df_raw, min_pct=5.0) -> pd.DataFrame:
    """Missingness is mostly structural: whole stations do not measure some quantities."""
    cols = [c for c in df_raw.columns if df_raw[c].isna().mean() * 100 >= min_pct]
    g = df_raw.groupby("Location")
    n_loc = df_raw["Location"].nunique()
    rows = {}
    for c in cols:
        never = g[c].apply(lambda s: s.isna().all())
        rows[c] = {"pct_missing": round(100 * df_raw[c].isna().mean(), 1),
                   "stations that never record it": f"{int(never.sum())} of {n_loc}",
                   "share of the gaps from those stations":
                       round(df_raw.loc[df_raw["Location"].isin(never[never].index), c].isna().sum()
                             / max(df_raw[c].isna().sum(), 1), 3)}
    return pd.DataFrame(rows).T
