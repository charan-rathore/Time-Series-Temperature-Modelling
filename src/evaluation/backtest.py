"""Rolling-origin backtesting with strong baselines.

One 30-day holdout says little: it is a single weather regime. This module
re-scores every forecaster over several consecutive, non-overlapping test
folds, each trained only on data that precedes it (expanding window), and
reports every model next to simple baselines it has to beat.

Forecaster contract
-------------------
A forecaster is a callable ``f(train_df, fold_df, h) -> np.ndarray`` whose
result has length ``len(fold_df) - h``: element ``i`` is the forecast, made
from information up to and including fold row ``i``, for the temperature at
fold row ``i + h``. This is the same t + h pairing the training evaluation
uses, so backtest numbers are comparable with models/results.json.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

TARGET = "temp_c"
Forecaster = Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray]


def rolling_origin_folds(
    n_rows: int, n_folds: int, fold_len: int, min_train: int
) -> List[Tuple[int, int, int]]:
    """Expanding-window folds as (train_end, test_start, test_end) row bounds.

    Folds tile the END of the series back to back, so the newest data is
    always tested. The train window is rows [0, train_end) and never overlaps
    its own test window, so nothing leaks. Raises if the series is too short.
    """
    if n_folds < 1 or fold_len < 2:
        raise ValueError("need n_folds >= 1 and fold_len >= 2")
    first_test = n_rows - n_folds * fold_len
    if first_test < min_train:
        raise ValueError(
            f"series of {n_rows} rows cannot fit {n_folds} folds of {fold_len} "
            f"after {min_train} training rows"
        )
    folds = []
    for k in range(n_folds):
        start = first_test + k * fold_len
        folds.append((start, start, start + fold_len))
    return folds


# ── baselines ────────────────────────────────────────────────────────────────

def persistence(train_df: pd.DataFrame, fold_df: pd.DataFrame, h: int) -> np.ndarray:
    """Tomorrow equals today."""
    y = fold_df[TARGET].to_numpy(dtype=float)
    return y[: len(y) - h].copy()


def seasonal_naive_7(train_df: pd.DataFrame, fold_df: pd.DataFrame, h: int) -> np.ndarray:
    """The value observed one week before the target day (needs h <= 7)."""
    if h > 7:
        raise ValueError("seasonal_naive_7 needs h <= 7")
    hist = np.concatenate([
        train_df[TARGET].to_numpy(dtype=float), fold_df[TARGET].to_numpy(dtype=float)
    ])
    offset = len(train_df)
    out = []
    for i in range(len(fold_df) - h):
        target_idx = offset + i + h
        out.append(hist[target_idx - 7])  # at most offset + i, i.e. already observed
    return np.asarray(out)


def climatology(train_df: pd.DataFrame, fold_df: pd.DataFrame, h: int) -> np.ndarray:
    """Mean training temperature for the target's day of year (+/- 7 days)."""
    doy_train = pd.to_datetime(train_df["date"]).dt.dayofyear.to_numpy()
    temps = train_df[TARGET].to_numpy(dtype=float)
    fallback = float(np.mean(temps))
    target_dates = pd.to_datetime(fold_df["date"]).iloc[h:]
    out = []
    for d in target_dates.dt.dayofyear.to_numpy():
        gap = np.abs(doy_train - d)
        gap = np.minimum(gap, 365 - gap)
        near = temps[gap <= 7]
        out.append(float(np.mean(near)) if len(near) else fallback)
    return np.asarray(out)


BASELINES: Dict[str, Forecaster] = {
    "persistence": persistence,
    "seasonal_naive_7": seasonal_naive_7,
    "climatology": climatology,
}


def lgbm_forecaster(config: dict) -> Forecaster:
    """LightGBM wrapped to the forecaster contract (imports lightgbm lazily)."""
    def run(train_df: pd.DataFrame, fold_df: pd.DataFrame, h: int) -> np.ndarray:
        from src.models.lgbm_model import LGBMForecastModel

        model = LGBMForecastModel(config, horizon=h)
        model.fit(train_df)  # no val_df: the fold's own data must not steer training
        preds = model.predict(steps=len(fold_df), future_df=fold_df)
        return np.asarray(preds[: len(fold_df) - h], dtype=float)
    return run


# ── scoring ──────────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def run_backtest(
    df: pd.DataFrame,
    forecasters: Dict[str, Forecaster],
    horizons: Sequence[int] = (1, 2, 3),
    n_folds: int = 6,
    fold_len: int = 30,
    min_train: int = 180,
) -> dict:
    """Score every forecaster on every fold and horizon.

    Returns a JSON-safe dict:
      models[name]["dayH"] = {rmse (pooled over all folds), mae, n_observations,
                              fold_rmse[list], fold_rmse_std,
                              skill_vs_persistence, skill_vs_best_baseline}
    Skill = 1 - rmse_model / rmse_reference (positive means better).
    """
    df = df.reset_index(drop=True)
    folds = rolling_origin_folds(len(df), n_folds, fold_len, min_train)
    errs: Dict[str, Dict[int, List[List[float]]]] = {
        name: {h: [] for h in horizons} for name in forecasters
    }
    for train_end, t0, t1 in folds:
        train_df, fold_df = df.iloc[:train_end], df.iloc[t0:t1]
        y = fold_df[TARGET].to_numpy(dtype=float)
        for name, fn in forecasters.items():
            for h in horizons:
                p = np.asarray(fn(train_df, fold_df, h), dtype=float)
                if len(p) != len(y) - h:
                    raise ValueError(f"{name} h={h}: got {len(p)} forecasts, need {len(y) - h}")
                errs[name][h].append((y[h:] - p).tolist())

    models: dict = {}
    for name in forecasters:
        models[name] = {}
        for h in horizons:
            per_fold = [np.asarray(e) for e in errs[name][h]]
            pooled = np.concatenate(per_fold)
            fold_rmse = [float(np.sqrt(np.mean(e ** 2))) for e in per_fold]
            models[name][f"day{h}"] = {
                "rmse": float(np.sqrt(np.mean(pooled ** 2))),
                "mae": float(np.mean(np.abs(pooled))),
                "n_observations": int(len(pooled)),
                "fold_rmse": fold_rmse,
                "fold_rmse_std": float(np.std(fold_rmse)),
            }
    baseline_names = [n for n in forecasters if n in BASELINES]
    for h in horizons:
        key = f"day{h}"
        best_base = min((models[b][key]["rmse"] for b in baseline_names), default=None)
        pers = models["persistence"][key]["rmse"] if "persistence" in models else None
        for name in forecasters:
            r = models[name][key]["rmse"]
            models[name][key]["skill_vs_persistence"] = None if not pers else 1 - r / pers
            models[name][key]["skill_vs_best_baseline"] = None if not best_base else 1 - r / best_base
    return {
        "method": "rolling-origin expanding window",
        "n_folds": n_folds,
        "fold_len": fold_len,
        "min_train": min_train,
        "pairing": "forecast from row i for row i + h",
        "models": models,
    }
