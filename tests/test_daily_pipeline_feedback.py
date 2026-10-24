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
    # long enough history for the 30-day lag and rolling features
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


def _fetch_shaped(start, end):
    """Match fetch_historical_open_meteo: temp_c and covariates, no temp_c_api."""
    d = pd.date_range(start, end)
    return pd.DataFrame({"date": d, "temp_c": 22.0, "humidity_pct": 70.0, "dewpoint_c": 15.0,
                         "precip_mm": 0.0, "pressure_hpa": 1010.0, "cloudcover_pct": 40.0,
                         "windspeed_kmh": 8.0, "solar_radiation_wm2": 200.0})


def test_consecutive_daily_runs_fetch_one_day_each_and_keep_temp_c_api():
    store = {"df": _full(60, "2026-08-21")}  # full rows through 10-19
    import src.data.fetcher as fetcher
    windows = []

    def fake_fetch(**kw):
        windows.append((kw["start_date"], kw["end_date"]))
        return _fetch_shaped(kw["start_date"], kw["end_date"])

    for day in (21, 22, 23, 24):
        class D(dt.date):
            @classmethod
            def today(cls, _d=day):
                return cls(2026, 10, _d)
        with patch.object(rp, "date", D), patch.object(rp, "load_processed", side_effect=lambda: store["df"]), \
             patch.object(fetcher, "fetch_historical_open_meteo", side_effect=fake_fetch), \
             patch.object(rp, "save_processed", side_effect=lambda df: store.__setitem__("df", df)), \
             patch.object(rp, "_refresh_bundle"), \
             patch.object(rp, "fetch_forecast_open_meteo", return_value=pd.DataFrame({"a": [1]})), \
             patch.object(pd.DataFrame, "to_parquet"):
            rp.cmd_daily(argparse.Namespace())
    assert windows == [("2026-10-20", "2026-10-20"), ("2026-10-21", "2026-10-21"),
                       ("2026-10-22", "2026-10-22"), ("2026-10-23", "2026-10-23")]
    assert store["df"]["temp_c_api"].notna().all()
    # the appended days must reach the model's feature matrix, not be dropped
    from src.features.engineer import build_feature_matrix
    assert build_feature_matrix(store["df"])["date"].max() == pd.Timestamp("2026-10-23")


def test_feedback_row_gets_temp_c_api_from_real_shaped_fetch():
    existing = pd.concat([_full(10), _feedback_row("2026-10-23", 28.0)], ignore_index=True)
    _, saved = _run_daily(existing, _fetch_shaped("2026-10-20", "2026-10-23"))
    row = saved[saved["date"] == pd.Timestamp("2026-10-23")].iloc[0]
    assert row["temp_c"] == 28.0 and row["temp_c_api"] == 22.0
    # fetched API-only flags must only fill empty cells, never overwrite the feedback's
    assert bool(row["is_sensor_reading"]) is True and bool(row["gap_filled"]) is False
    others = saved[saved["date"] != pd.Timestamp("2026-10-23")]
    assert not others["is_sensor_reading"].astype(bool).any()
    assert saved["temp_c_api"].notna().all()
