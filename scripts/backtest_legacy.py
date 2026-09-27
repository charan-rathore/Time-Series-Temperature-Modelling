#!/usr/bin/env python3
"""Leakage-free walk-forward sensor-only baseline; not a commercial-app comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import load_legacy_csv


def backtest(data: pd.DataFrame, warmup: int = 20) -> dict:
    rows = data.sort_values('date').reset_index(drop=True)
    if len(rows) < warmup + 6:
        raise ValueError('Need warmup history and at least six test dates')
    if rows.date.duplicated().any() or not rows.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError('This benchmark needs a complete daily sequence')
    records = {}
    for horizon in (1, 2, 3):
        targets, persistence, trailing_mean = [], [], []
        # A forecast issued at origin i is allowed to use readings only through i.
        for origin in range(warmup - 1, len(rows) - horizon):
            history = rows.temp_c.iloc[:origin + 1].to_numpy(dtype=float)
            targets.append(float(rows.temp_c.iloc[origin + horizon]))
            persistence.append(history[-1])
            trailing_mean.append(float(history[-7:].mean()))
        target = np.asarray(targets)
        records[f'day{horizon}'] = {
            'untouched_targets': len(target),
            'first_target': str(rows.date.iloc[warmup - 1 + horizon].date()),
            'last_target': str(rows.date.iloc[-1].date()),
            'persistence_mae_c': round(float(np.mean(abs(target - persistence))), 3),
            'trailing_7_observation_mean_mae_c': round(float(np.mean(abs(target - trailing_mean))), 3),
            'persistence_rmse_c': round(float(np.sqrt(np.mean((target - persistence) ** 2))), 3),
            'trailing_7_observation_mean_rmse_c': round(float(np.sqrt(np.mean((target - trailing_mean) ** 2))), 3),
        }
    return {
        'kind': 'sensor_only_walk_forward_baseline_not_ThermoSense_model_or_commercial_api',
        'source': 'repo data/legacy/temperature-data-for-TSA.csv; instrumentation, location and forecast issue times unverified',
        'warmup': warmup,
        'input_days': len(rows),
        'horizons': records,
        'warning': 'Each tested forecast uses readings available at issue date only. This benchmarks simple baselines, NOT the published trained models or vendor forecast advantage.',
    }


if __name__ == '__main__':
    print(json.dumps(backtest(load_legacy_csv()), indent=2))
