"""Data download, cleaning and train/val/test splitting.

Dataset: Rain in Australia ("weatherAUS"): about 10 years (2007-2017) of daily weather observations
from 49 Australian weather stations, published by the Australian Bureau of Meteorology and packaged
in the R package `rattle` (G. Williams) / on Kaggle ("Rain in Australia").
Task: predict from today's observations whether it will rain tomorrow (RainTomorrow).
The CSV is downloaded from a public GitHub mirror of that file.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/acakin/weatherAUS/master/weatherAUS.csv"
SOURCE_PAGE = "https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package"
FILENAME = "weatherAUS.csv"

RAW_TARGET = "RainTomorrow"
TARGET = "rain_tomorrow"
CLASS_NAMES = {0: "No rain", 1: "Rain"}
GROUP_COL = "RainToday"          # used by the EDA class-balance plot

# No column is recorded after the label day, so there is no leakage column.
# (An older version of this dataset contained RISK_MM = tomorrow's rainfall amount, which IS the
#  label in disguise; this file does not have it - checked in the notebook.)
LEAKAGE_COLS: list[str] = []
DROP_COLS = ["Date"]             # replaced by the month number (see engineer_features)

CAT_COLS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
NUM_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "month",
]
SKEWED_COLS = ["Rainfall", "Evaporation"]   # zero-inflated / long right tail


def download_data(data_dir: str = "data", url: str = DATA_URL) -> str:
    """Download the raw CSV from the public link (no cloud drive needed)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, FILENAME)
    if not os.path.exists(path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_raw(data_dir: str = "data") -> pd.DataFrame:
    return pd.read_csv(download_data(data_dir))


def clean(df: pd.DataFrame, drop_duplicates: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Remove rows without a label and duplicates; sort by date. Missing feature values are KEPT
    on purpose: imputation is a configurable pipeline step (fit on the training split only)."""
    log = []
    n0 = len(df)
    df = df.copy()
    if drop_duplicates:
        df = df.drop_duplicates()
        log.append(f"exact duplicate rows removed: {n0 - len(df):,}")
    n1 = len(df)
    df = df[df[RAW_TARGET].notna()]
    log.append(f"rows with unknown RainTomorrow removed (no label to learn from): {n1 - len(df):,}")
    df[TARGET] = (df[RAW_TARGET] == "Yes").astype(int)
    df = df.drop(columns=[RAW_TARGET])
    # chronological order (stable): the split below cuts this table by time
    df = df.sort_values(["Date", "Location"], kind="mergesort")
    if verbose:
        print("\n".join(" - " + s for s in log))
        print(f" => {len(df):,} rows remain (from {n0:,}), "
              f"{df['Date'].min()} to {df['Date'].max()}")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only one derived feature: the month (seasonality). The calendar year and the exact date are
    not used - a model must work for future days."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["Date"]).dt.month
    return df.drop(columns=DROP_COLS)


def get_xy(df: pd.DataFrame):
    X = df[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(object)
    y = df[TARGET].astype(int).values
    return X, y


def split(X, y, val_size=0.15, test_size=0.15, seed=42):
    """CHRONOLOGICAL train / validation / test split (70 / 15 / 15): the oldest 70% of the days
    train the model, the next 15% select models, the newest 15% give the final numbers.
    A random split would be optimistic here: consecutive days of one station are strongly
    correlated, and the label of day t (RainTomorrow) is the feature RainToday of day t+1, so
    neighbours of a test day would sit in the training set. `X` must be in date order
    (clean() guarantees it); `seed` is unused and only kept for API compatibility."""
    n = len(y)
    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    n_train = n - n_val - n_test
    s_tr, s_va, s_te = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)
    return X.iloc[s_tr], X.iloc[s_va], X.iloc[s_te], y[s_tr], y[s_va], y[s_te]


def load_dataset(data_dir="data", seed=42, verbose=True):
    """One call: download -> clean -> engineer -> split."""
    raw = load_raw(data_dir)
    df = clean(raw, verbose=verbose)
    dates = df["Date"].values
    df = engineer_features(df)
    X, y = get_xy(df)
    out = split(X, y, seed=seed)
    if verbose:
        n_tr, n_va = len(out[0]), len(out[1])
        print(f" split by date: train up to {dates[n_tr - 1]}, validation up to "
              f"{dates[n_tr + n_va - 1]}, test up to {dates[-1]}")
    return out
