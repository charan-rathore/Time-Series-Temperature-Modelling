"""The daily update must not treat a sparse feedback row as a fetched day."""
import argparse
import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pipeline as rp  # noqa: E402


def _full(days, start="2026-10-10"):
    d = pd.date_range(start, periods=days)
    return pd.DataFrame({"date": d, "temp_c": 25.0, "temp_c_api": 24.0,
                         "humidity_pct": 60.0, "is_sensor_reading": False, "gap_filled": False})


def _feedback_row(day, temp=28.0):
    return pd.DataFrame([{"date": pd.Timestamp(day), "temp_c": temp,
                          "is_sensor_reading": True, "gap_filled": False}])


class _Day(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 24)


def _run_daily(existing, fetched):
    calls = []
    import src.data.fetcher as fetcher

    def fake_fetch(**kw):
        calls.append(kw)
        return fetched

    saved = {}
    with patch.object(rp, "date", _Day), patch.object(rp, "load_processed", return_value=existing), \
         patch.object(fetcher, "fetch_historical_open_meteo", side_effect=fake_fetch), \
         patch.object(rp, "save_processed", side_effect=lambda df: saved.setdefault("df", df)), \
         patch.object(rp, "_refresh_bundle"), \
         patch.object(rp, "fetch_forecast_open_meteo", return_value=pd.DataFrame({"a": [1]})), \
         patch.object(pd.DataFrame, "to_parquet"):
        rp.cmd_daily(argparse.Namespace())
    return calls, saved.get("df")


def test_feedback_for_yesterday_does_not_suppress_the_fetch():
    existing = pd.concat([_full(10), _feedback_row("2026-10-23")], ignore_index=True)  # full rows end 10-19
    fetched = _full(4, "2026-10-20")
    calls, saved = _run_daily(existing, fetched)
    assert calls == [{"start_date": "2026-10-20", "end_date": "2026-10-23"}]
    assert saved["date"].dt.date.astype(str).tolist() == [
        str(d.date()) for d in pd.date_range("2026-10-10", "2026-10-23")]


def test_feedback_temperature_is_kept_and_missing_columns_filled():
    existing = pd.concat([_full(10), _feedback_row("2026-10-23", 28.0)], ignore_index=True)
    fetched = _full(4, "2026-10-20")
    _, saved = _run_daily(existing, fetched)
    row = saved[saved["date"] == pd.Timestamp("2026-10-23")].iloc[0]
    assert row["temp_c"] == 28.0
    assert row["temp_c_api"] == 24.0 and row["humidity_pct"] == 60.0
    assert bool(row["is_sensor_reading"]) is True


def test_without_feedback_merge_matches_plain_append():
    existing = _full(10)
    new = _full(4, "2026-10-20")
    old = pd.concat([existing, new], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    pd.testing.assert_frame_equal(rp._merge_daily(existing, new), old, check_dtype=True)
