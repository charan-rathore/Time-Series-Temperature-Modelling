"""Hourly poller: real measured temperature from the HAL airport station (VOBG).

Source: aviationweather.gov METAR API (free, no key). This is an AIRPORT
station reading, not a hyperlocal one, and it is stored under the source label
"airport_vobg_metar" so nothing downstream can mistake it for the Pi sensor.

A real sensor reading for the same date always wins: the poller never
overwrites a row whose source is not the airport label, and it only writes
the one observation closest to the 9 PM IST comparison window.

Usage:  python -m src.data.airport_metar
"""

import logging
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from src.data.baseline_collector import _DB_PATH, init_database, store_actual

logger = logging.getLogger(__name__)

METAR_URL = "https://aviationweather.gov/api/data/metar"
STATION = "VOBG"
SOURCE_LABEL = "airport_vobg_metar"
IST = timezone(timedelta(hours=5, minutes=30))
WINDOW_START = (20, 30)  # 20:30 IST
WINDOW_END = (21, 30)  # 21:30 IST
TEMP_MIN_C, TEMP_MAX_C = -10.0, 55.0


def fetch_latest_metar(station: str = STATION, session=requests) -> Optional[dict]:
    """Return the newest METAR record or None on any failure. Never raises."""
    try:
        r = session.get(
            METAR_URL, params={"ids": station, "format": "json"}, timeout=15,
            headers={"User-Agent": "ThermoSense/1.0 (hourly poll)"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # network, HTTP, bad JSON
        logger.warning("METAR fetch failed: %s", exc)
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    return data[0]


def _in_window(t: datetime) -> bool:
    hm = (t.hour, t.minute)
    return WINDOW_START <= hm <= WINDOW_END


def poll_once(db_path: Path = _DB_PATH, record: Optional[dict] = None) -> str:
    """Poll and store. Returns a short status string (for logs and tests)."""
    record = record if record is not None else fetch_latest_metar()
    if not record:
        return "no_data"
    temp, obs = record.get("temp"), record.get("obsTime")
    if isinstance(temp, bool) or not isinstance(temp, (int, float)):
        return "no_temperature"
    if not isinstance(obs, (int, float)) or isinstance(obs, bool) or not math.isfinite(obs):
        return "no_time"
    temp = float(temp)
    if not (TEMP_MIN_C <= temp <= TEMP_MAX_C):
        return "implausible"
    try:
        t = datetime.fromtimestamp(obs, tz=IST)
    except (OverflowError, OSError, ValueError):
        return "no_time"
    if not _in_window(t):
        return "outside_window"
    d: date = t.date()
    init_database(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT source FROM daily_actuals WHERE date = ?", (d.isoformat(),)
        ).fetchone()
    finally:
        conn.close()
    if row and row[0] != SOURCE_LABEL:
        return "real_sensor_wins"
    humidity = None
    dewp = record.get("dewp")
    if isinstance(dewp, (int, float)) and not isinstance(dewp, bool) and math.isfinite(dewp) and -243.04 < dewp <= temp:
        a, b = 17.625, 243.04
        humidity = round(100 * math.exp(a * dewp / (b + dewp)) / math.exp(a * temp / (b + temp)), 1)
    ok = store_actual(db_path, d, temp, humidity, source=SOURCE_LABEL)
    return "stored" if ok else "store_failed"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(poll_once())
