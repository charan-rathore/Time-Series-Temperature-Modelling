"""Independent walk-forward audit of ThermoSense forecasts.

Re-derives every number from data/bundle/daily_merged.csv with code that
shares nothing with the training pipeline, so a leak there cannot leak here.

Protocol
--------
* Origins: every day t in the last ``--test-days`` days (default 365). Each
  forecast for day t + h uses rows <= t only; models are refit every
  ``--refit`` days on rows <= t (expanding window).
* Horizons: 1, 2, 3 days.
* Baselines a model must beat: persistence (y_t), seasonal naive (y_{t+h-7}),
  day-of-year climatology from earlier years, and an anomaly AR(1) that
  decays today's departure from climatology toward climatology.
* Models: ridge and gradient boosting on lagged temperatures, weather lags
  and calendar terms (sklearn, so it runs without LightGBM).
* Uncertainty: 1000-sample moving-block bootstrap (block 14 days) of the
  RMSE difference against persistence; a model "beats" persistence only if
  the 95% interval excludes zero.
* Threshold alerts: a "hot day" is temp >= the 90th percentile of the
  training period before the test window; reports probability of detection,
  false-alarm ratio and Heidke-style hit counts for each forecaster.

Run:  python3 scripts/walk_forward_audit.py [--test-days 365] [--refit 7]
Writes docs/walk-forward-audit.json and prints a Markdown table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 2, 3)
WEATHER = ["humidity_pct", "dewpoint_c", "pressure_hpa", "cloudcover_pct",
           "windspeed_kmh", "solar_radiation_wm2", "precip_mm"]


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/bundle/daily_merged.csv", parse_dates=["date"])
    d = d.sort_values("date").reset_index(drop=True)
    gaps = d.date.diff().dt.days.dropna()
    if (gaps != 1).any():
        raise SystemExit("series has gaps; the audit assumes consecutive days")
    return d


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Row t holds only what is known at the end of day t."""
    f = pd.DataFrame(index=d.index)
    for k in (0, 1, 2, 3, 6, 13):
        f[f"lag{k}"] = d.temp_c.shift(k)
    f["mean7"] = d.temp_c.rolling(7).mean()
    f["mean30"] = d.temp_c.rolling(30).mean()
    f["delta1"] = d.temp_c.diff()
    for c in WEATHER:
        if c in d:
            f[c] = d[c]
            f[c + "_d1"] = d[c].diff()
    doy = d.date.dt.dayofyear
    f["sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def climatology(d: pd.DataFrame, upto: int) -> np.ndarray:
    """Smoothed day-of-year mean from rows < upto, as a 366-length table."""
    past = d.iloc[:upto]
    doy = past.date.dt.dayofyear.values
    tab = np.full(367, np.nan)
    for k in range(1, 367):
        w = np.abs(((doy - k + 183) % 366) - 183) <= 7  # +-7 day window
        if w.any():
            tab[k] = past.temp_c.values[w].mean()
    tab[np.isnan(tab)] = past.temp_c.mean()
    return tab


def run(test_days: int, refit: int, seed: int = 0):
    d = load()
    X = features(d)
    y = d.temp_c.values
    n = len(d)
    first = n - test_days - max(HORIZONS)
    if first < 400:
        raise SystemExit(f"not enough history: first origin at row {first}")
    names = ["persistence", "seasonal_naive_7", "climatology", "anomaly_ar1",
             "ridge", "gbm"]
    pred = {m: {h: [] for h in HORIZONS} for m in names}
    actual = {h: [] for h in HORIZONS}
    models = {}
    clim = None
    for t in range(first, n - max(HORIZONS)):
        if (t - first) % refit == 0:
            clim = climatology(d, t + 1)
            for h in HORIZONS:
                rows = np.arange(30, t + 1 - h)  # targets y[r+h] known at t
                Xr = X.iloc[rows].values
                yr = y[rows + h]
                ok = ~np.isnan(Xr).any(axis=1)
                models[("ridge", h)] = Ridge(alpha=3.0).fit(Xr[ok], yr[ok])
                models[("gbm", h)] = HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                    min_samples_leaf=20, l2_regularization=1.0,
                    random_state=seed).fit(Xr[ok], yr[ok])
                # anomaly persistence coefficient, fitted on the same rows
                doy_r = d.date.dt.dayofyear.values
                a0 = y[rows] - clim[doy_r[rows]]
                ah = yr - clim[doy_r[rows + h]]
                models[("ar1", h)] = float(np.dot(a0, ah) / np.dot(a0, a0))
        xt = X.iloc[[t]].values
        doy = d.date.dt.dayofyear.values
        for h in HORIZONS:
            actual[h].append(y[t + h])
            pred["persistence"][h].append(y[t])
            pred["seasonal_naive_7"][h].append(y[t + h - 7])
            pred["climatology"][h].append(clim[doy[t + h]])
            pred["anomaly_ar1"][h].append(
                clim[doy[t + h]] + models[("ar1", h)] * (y[t] - clim[doy[t]]))
            pred["ridge"][h].append(float(models[("ridge", h)].predict(xt)[0]))
            pred["gbm"][h].append(float(models[("gbm", h)].predict(xt)[0]))
    # hot-day threshold from the period before the test window only
    hot = float(np.percentile(y[:first], 90))
    rng = np.random.default_rng(seed)
    out = {"protocol": {
        "data": "data/bundle/daily_merged.csv",
        "rows": n, "range": f"{d.date.iloc[0].date()} to {d.date.iloc[-1].date()}",
        "origins": n - max(HORIZONS) - first, "refit_days": refit,
        "hot_day_threshold_c": round(hot, 2),
        "sensor_rows": int(d.get("is_sensor_reading", pd.Series(0)).sum()),
    }, "horizons": {}}
    for h in HORIZONS:
        a = np.array(actual[h])
        res = {}
        e_p = np.array(pred["persistence"][h]) - a
        for m in names:
            p = np.array(pred[m][h])
            e = p - a
            rmse = float(np.sqrt(np.mean(e ** 2)))
            # moving-block bootstrap of RMSE(model) - RMSE(persistence)
            L, B = 14, 1000
            starts = rng.integers(0, len(a) - L, size=(B, len(a) // L))
            idx = (starts[..., None] + np.arange(L)).reshape(B, -1)
            diff = np.sqrt((e[idx] ** 2).mean(1)) - np.sqrt((e_p[idx] ** 2).mean(1))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            ev, fc = a >= hot, p >= hot
            hits = int((ev & fc).sum())
            res[m] = {
                "rmse": round(rmse, 3), "mae": round(float(np.mean(np.abs(e))), 3),
                "bias": round(float(e.mean()), 3),
                "skill_vs_persistence": round(1 - rmse / float(np.sqrt(np.mean(e_p ** 2))), 3),
                "rmse_minus_persistence_95ci": [round(float(lo), 3), round(float(hi), 3)],
                "beats_persistence": bool(hi < 0),
                "hot_days": {"events": int(ev.sum()), "hits": hits,
                             "pod": round(hits / max(1, int(ev.sum())), 3),
                             "far": round(int((fc & ~ev).sum()) / max(1, int(fc.sum())), 3),
                             "false_alarms": int((fc & ~ev).sum())},
            }
        out["horizons"][f"day{h}"] = {"n": len(a), "models": res}
    return out


def table(out) -> str:
    lines = []
    for hk, hv in out["horizons"].items():
        lines.append(f"\n**{hk}** (n = {hv['n']})\n")
        lines.append("| Model | RMSE °C | MAE °C | Skill vs persistence | RMSE − persistence, 95% CI | Hot days POD / FAR |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for m, r in hv["models"].items():
            ci = r["rmse_minus_persistence_95ci"]
            mark = " ✓" if r["beats_persistence"] else ""
            hd = r["hot_days"]
            lines.append(f"| {m} | {r['rmse']} | {r['mae']} | {r['skill_vs_persistence']:+.3f}{mark} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {hd['pod']} / {hd['far']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-days", type=int, default=365)
    ap.add_argument("--refit", type=int, default=7)
    args = ap.parse_args()
    o = run(args.test_days, args.refit)
    (ROOT / "docs/walk-forward-audit.json").write_text(json.dumps(o, indent=1))
    print(json.dumps(o["protocol"]))
    print(table(o))
