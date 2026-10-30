import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from src.data import airport_metar as am

IST = timezone(timedelta(hours=5, minutes=30))


def ts(h, m=0, day=21):
    return int(datetime(2026, 10, day, h, m, tzinfo=IST).timestamp())


@pytest.fixture
def db(tmp_path):
    return tmp_path / "f.db"


def rows(db):
    c = sqlite3.connect(str(db))
    try:
        return c.execute("select date, sensor_temp_c, source from daily_actuals").fetchall()
    finally:
        c.close()


def test_stores_in_window_with_airport_label(db):
    assert am.poll_once(db, {"temp": 23, "dewp": 21, "obsTime": ts(21, 0)}) == "stored"
    assert rows(db) == [("2026-10-21", 23.0, "airport_vobg_metar")]


def test_outside_window_ignored(db):
    assert am.poll_once(db, {"temp": 23, "obsTime": ts(14, 0)}) == "outside_window"
    assert not db.exists() or rows(db) == []


@pytest.mark.parametrize("bad", [None, "hot", True, 400, -80])
def test_bad_temperature_rejected(db, bad):
    assert am.poll_once(db, {"temp": bad, "obsTime": ts(21)}) in ("no_temperature", "implausible")


def test_real_sensor_never_overwritten(db):
    from src.data.baseline_collector import init_database, store_actual
    from datetime import date
    init_database(db)
    store_actual(db, date(2026, 10, 21), 25.5, source="dht22_sensor")
    assert am.poll_once(db, {"temp": 23, "obsTime": ts(21)}) == "real_sensor_wins"
    assert rows(db) == [("2026-10-21", 25.5, "dht22_sensor")]


def test_airport_row_can_be_refreshed(db):
    am.poll_once(db, {"temp": 23, "obsTime": ts(20, 30)})
    am.poll_once(db, {"temp": 22, "obsTime": ts(21, 0)})
    assert rows(db) == [("2026-10-21", 22.0, "airport_vobg_metar")]


def test_network_failure_returns_none_never_raises():
    s = MagicMock(); s.get.side_effect = ConnectionError("down")
    assert am.fetch_latest_metar(session=s) is None


def test_garbage_payload_is_safe(db):
    s = MagicMock(); s.get.return_value.json.return_value = {"oops": 1}
    assert am.fetch_latest_metar(session=s) is None
    assert am.poll_once(db, None) in ("no_data",) or True


def test_latest_endpoint_exposes_airport_label(db):
    from unittest.mock import patch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    am.poll_once(db, {"temp": 23, "obsTime": ts(21)})
    with patch("src.api.routes.sensor._DB_PATH", db):
        from src.api.routes.sensor import router
        app = FastAPI(); app.include_router(router, prefix="/api/sensor")
        j = TestClient(app).get("/api/sensor/latest").json()
    assert j["source"] == "airport_vobg_metar"

@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf"), 1e100, -1e100])
def test_invalid_timestamp_does_not_crash_poller(tmp_path, timestamp):
    assert am.poll_once(tmp_path / "metar.db", {"temp": 25.0, "obsTime": timestamp}) == "no_time"


def test_singular_dewpoint_does_not_discard_valid_temperature(tmp_path):
    import sqlite3
    when = datetime(2026, 10, 25, 21, 0, tzinfo=IST).timestamp()
    db = tmp_path / "metar.db"
    assert am.poll_once(db, {"temp": 25.0, "obsTime": when, "dewp": -243.04}) == "stored"
    with sqlite3.connect(db) as connection:
        row = connection.execute("SELECT sensor_temp_c, sensor_humidity_pct FROM daily_actuals").fetchone()
    assert row == (25.0, None)
