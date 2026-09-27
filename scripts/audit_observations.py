#!/usr/bin/env python3
"""Reproduce same-time archive-versus-legacy comparisons, never a forecast score."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.fetcher import fetch_historical_open_meteo
from src.data.preprocess import load_legacy_csv


def compare(archive: pd.DataFrame, observations: pd.DataFrame) -> dict:
    archive = archive[["date", "temp_c"]].rename(columns={"temp_c": "archive_9pm_c"})
    observations = observations[["date", "temp_c"]].rename(columns={"temp_c": "legacy_observation_c"})
    paired = archive.merge(observations, on="date", how="inner").sort_values("date")
    if len(paired) < 10:
        raise ValueError("Need at least 10 paired days to report observational comparison")
    diffs = paired.legacy_observation_c.to_numpy() - paired.archive_9pm_c.to_numpy()
    # Later dates form an untouched reporting window, and only earlier observed
    # dates can set the fixed bias. No current-day actual enters calibration.
    split = max(7, int(len(paired) * 0.7))
    if len(paired) - split < 3:
        raise ValueError("Need at least three untouched reporting days")
    offset = float(diffs[:split].mean())
    holdout = diffs[split:]
    return {
        "kind": "same-time_observational_comparison_not_forecast",
        "archive_source": "https://archive-api.open-meteo.com/v1/archive",
        "observation_source": "repo data/legacy/temperature-data-for-TSA.csv; measurement instrument and siting unverified",
        "first_date": str(paired.date.iloc[0].date()),
        "last_date": str(paired.date.iloc[-1].date()),
        "paired_days": len(paired),
        "calibration_days": split,
        "untouched_days": len(holdout),
        "calibration_offset_c": round(offset, 3),
        "holdout_archive_mae_c": round(float(np.mean(np.abs(holdout))), 3),
        "holdout_bias_corrected_same_time_mae_c": round(float(np.mean(np.abs(holdout - offset))), 3),
        "warning": "Same-day archive value is known only retrospectively. This does NOT validate a future forecast, DHT22 capture, or live model performance.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-06-02")
    parser.add_argument("--end", default="2024-07-10")
    parser.add_argument("--offline-raw", type=Path, help="Previously saved raw Open-Meteo response for reproducibility")
    args = parser.parse_args()
    if args.offline_raw:
        from src.data.fetcher import _parse_hourly_to_df, _resample_to_9pm, HOURLY_VARS
        payload = json.loads(args.offline_raw.read_text())
        archive = _resample_to_9pm(_parse_hourly_to_df(payload["hourly"], HOURLY_VARS))
    else:
        archive = fetch_historical_open_meteo(start_date=args.start, end_date=args.end)
    print(json.dumps(compare(archive, load_legacy_csv()), indent=2))


if __name__ == "__main__":
    main()
