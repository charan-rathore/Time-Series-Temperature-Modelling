"""Data-quality and drift checks for the daily temperature series.

Each check returns plain dicts so results can feed an API health block, a
banner, or a test. Checks flag; they never delete or repair data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def find_gaps(dates: Sequence, max_gap_days: int = 1) -> List[Dict]:
    """Missing calendar days between consecutive observations."""
    d = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    gaps = []
    for prev, cur in zip(d[:-1], d[1:]):
        missing = (cur - prev).days - 1
        if missing >= max_gap_days:
            gaps.append({"after": prev.date().isoformat(), "missing_days": int(missing)})
    return gaps


def find_outliers(values: Sequence[float], z_limit: float = 4.0) -> List[int]:
    """Indices whose day-to-day jump is extreme under a robust (MAD) scale.

    Robust so one bad reading cannot inflate the yardstick that is meant to
    catch it. A spike produces two large jumps (up and back down); both
    neighbours are reported only if they exceed the limit, the spike index
    itself is the one flagged.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 8:
        return []
    jumps = np.diff(v)
    mad = np.median(np.abs(jumps - np.median(jumps)))
    scale = 1.4826 * mad
    if scale == 0:
        return []
    z = np.abs(jumps - np.median(jumps)) / scale
    flagged = []
    for i in range(len(jumps) - 1):
        # point i + 1 is a spike if the jump in and the jump out are both extreme
        if z[i] > z_limit and z[i + 1] > z_limit and np.sign(jumps[i]) != np.sign(jumps[i + 1]):
            flagged.append(i + 1)
    return flagged


def find_stuck_runs(values: Sequence[float], min_run: int = 4) -> List[Dict]:
    """Runs of identical readings (a frozen sensor or a repeated fill value)."""
    v = np.asarray(values, dtype=float)
    runs, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            if i - start >= min_run:
                runs.append({"start_index": int(start), "length": int(i - start),
                             "value": float(v[start])})
            start = i
    return runs


def population_stability_index(reference: Sequence[float], current: Sequence[float],
                               bins: int = 10) -> float:
    """PSI between a reference window and a current window.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def error_drift(recent_errors: Sequence[float], baseline_rmse: float,
                tolerance: float = 1.5) -> Dict:
    """Is recent live error materially worse than the backtest said it would be?"""
    e = np.asarray(recent_errors, dtype=float)
    if len(e) < 7:
        return {"status": "insufficient_data", "n": int(len(e))}
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {"status": "drifting" if rmse > baseline_rmse * tolerance else "ok",
            "recent_rmse": rmse, "baseline_rmse": baseline_rmse, "n": int(len(e))}


def freshness(latest_date, today: Optional[pd.Timestamp] = None, max_age_days: int = 2) -> Dict:
    """Age of the newest observation against an SLO."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    age = (today.normalize() - pd.Timestamp(latest_date).normalize()).days
    return {"latest": pd.Timestamp(latest_date).date().isoformat(), "age_days": int(age),
            "slo_days": max_age_days, "stale": bool(age > max_age_days)}


def summarize(df: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> Dict:
    """One health summary for the processed daily frame."""
    df = df.sort_values("date").reset_index(drop=True)
    temps = df["temp_c"].to_numpy(dtype=float)
    half = len(temps) // 2
    return {
        "rows": int(len(df)),
        "freshness": freshness(df["date"].max(), today),
        "gaps": find_gaps(df["date"]),
        "outlier_count": len(find_outliers(temps)),
        "stuck_runs": find_stuck_runs(temps),
        "temp_psi_first_vs_second_half": population_stability_index(temps[:half], temps[half:]),
    }
