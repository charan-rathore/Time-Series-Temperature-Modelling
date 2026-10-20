#!/usr/bin/env python3
"""Record the latest processed day's actual temperature as reference ground truth.

The leaderboard scores every forecast source against rows in daily_actuals.
Until the Raspberry Pi sensor is live, the Open-Meteo archive value is the
honest reference - stored with source="open_meteo_reference" so it never
poses as a sensor reading. A real reading already stored for the day is
never overwritten by the reference.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.preprocess import load_processed
from src.data.baseline_collector import _DB_PATH, init_database, store_actual


REFERENCE_SOURCE = "open_meteo_reference"


def _existing_source(db_path: Path, day) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT source FROM daily_actuals WHERE date = ?", (day.isoformat(),)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def main() -> int:
    df = load_processed().sort_values("date")
    row = df.iloc[-1]
    day = pd.to_datetime(row["date"]).date()
    humidity = row.get("humidity_pct")
    humidity = float(humidity) if pd.notna(humidity) else None
    init_database(_DB_PATH)
    existing = _existing_source(_DB_PATH, day)
    if existing is not None and existing != REFERENCE_SOURCE:
        print(f"[reference-actual] {day}: keeping existing {existing!r} actual")
        return 0
    ok = store_actual(
        _DB_PATH,
        day,
        float(row["temp_c"]),
        humidity,
        api_temp_c=float(row["temp_c_api"]) if pd.notna(row.get("temp_c_api")) else None,
        source=REFERENCE_SOURCE,
    )
    print(f"[reference-actual] {day}: {row['temp_c']}°C stored={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
