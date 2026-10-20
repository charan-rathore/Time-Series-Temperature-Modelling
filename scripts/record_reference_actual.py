#!/usr/bin/env python3
"""Record the latest processed day's actual temperature as reference ground truth.

The leaderboard scores every forecast source against rows in daily_actuals.
Until the Raspberry Pi sensor is live, the Open-Meteo archive value is the
honest reference - stored with source="open_meteo_reference" so it never
poses as a sensor reading. When the Pi reports, store_actual's INSERT OR
REPLACE keeps one row per date and the sensor source wins in sensor views.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.preprocess import load_processed
from src.data.baseline_collector import _DB_PATH, init_database, store_actual


def main() -> None:
    df = load_processed().sort_values("date")
    row = df.iloc[-1]
    day = pd.to_datetime(row["date"]).date()
    humidity = row.get("humidity_pct")
    humidity = float(humidity) if pd.notna(humidity) else None
    init_database(_DB_PATH)
    ok = store_actual(
        _DB_PATH,
        day,
        float(row["temp_c"]),
        humidity,
        api_temp_c=float(row["temp_c_api"]) if pd.notna(row.get("temp_c_api")) else None,
        source="open_meteo_reference",
    )
    print(f"[reference-actual] {day}: {row['temp_c']}°C stored={ok}")


if __name__ == "__main__":
    main()
