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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.preprocess import load_processed
from src.data.baseline_collector import _DB_PATH, init_database


REFERENCE_SOURCE = "open_meteo_reference"


def store_reference(db_path: Path, day, temp_c: float, humidity, api_temp_c) -> bool:
    """Insert the reference for `day`, or refresh an older reference row.

    One SQL statement, so a sensor row that lands at any moment is never
    overwritten: the upsert only updates a row whose source is the reference.
    Returns True when a row was written, False when a real reading was kept.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            INSERT INTO daily_actuals
                (date, sensor_temp_c, sensor_humidity_pct, api_temp_c, recorded_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                sensor_temp_c = excluded.sensor_temp_c,
                sensor_humidity_pct = excluded.sensor_humidity_pct,
                api_temp_c = excluded.api_temp_c,
                recorded_at = excluded.recorded_at
            WHERE daily_actuals.source = ?
            """,
            (
                day.isoformat(),
                temp_c,
                humidity,
                api_temp_c,
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                REFERENCE_SOURCE,
                REFERENCE_SOURCE,
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def main() -> int:
    df = load_processed().sort_values("date")
    row = df.iloc[-1]
    day = pd.to_datetime(row["date"]).date()
    humidity = row.get("humidity_pct")
    humidity = float(humidity) if pd.notna(humidity) else None
    api_temp = float(row["temp_c_api"]) if pd.notna(row.get("temp_c_api")) else None
    init_database(_DB_PATH)
    try:
        written = store_reference(_DB_PATH, day, float(row["temp_c"]), humidity, api_temp)
    except sqlite3.Error as exc:
        print(f"[reference-actual] {day}: write failed: {exc}", file=sys.stderr)
        return 1
    if written:
        print(f"[reference-actual] {day}: {row['temp_c']}\u00b0C stored")
    else:
        print(f"[reference-actual] {day}: keeping the existing sensor reading")
    return 0


if __name__ == "__main__":
    sys.exit(main())
