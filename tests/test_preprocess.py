"""
Tests for src/data/preprocess.py

All file I/O is mocked or directed to tmp_path fixtures.
Covers: legacy CSV loading, merge logic, gap detection, validation, and pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    detect_and_fill_gaps,
    merge_with_legacy,
    run_pipeline,
    save_processed,
    validate_processed,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_api_df(n: int = 10, start: str = "2024-06-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "temp_c": [26.0 + i * 0.1 for i in range(n)],
        "humidity_pct": [70.0] * n,
        "dewpoint_c": [18.0] * n,
        "precip_mm": [0.0] * n,
        "pressure_hpa": [1010.0] * n,
        "cloudcover_pct": [30.0] * n,
        "windspeed_kmh": [10.0] * n,
        "uv_index": [3.0] * n,
    })


def make_legacy_df(n: int = 5, start: str = "2024-06-01") -> pd.DataFrame:
    """Simulates output of load_legacy_csv()."""
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "temp_c": [28.0 + i * 0.5 for i in range(n)],
        "app_pred_day1": [27.0] * n,
        "app_pred_day2": [27.5] * n,
        "app_pred_day3": [27.8] * n,
    })


# ── merge_with_legacy ─────────────────────────────────────────────────────────

def test_merge_sensor_temp_takes_precedence():
    api_df = make_api_df(n=10, start="2024-06-01")
    legacy_df = make_legacy_df(n=5, start="2024-06-01")

    merged = merge_with_legacy(api_df, legacy_df)

    # For the 5 overlapping dates, sensor temp should be used
    overlap = merged[merged["is_sensor_reading"] == True]
    assert len(overlap) == 5

    # Sensor temperatures are 28.0, 28.5, ... (not 26.0, 26.1, ... from API)
    assert overlap["temp_c"].iloc[0] == pytest.approx(28.0)


def test_merge_api_only_rows_have_is_sensor_false():
    api_df = make_api_df(n=10, start="2024-06-01")
    legacy_df = make_legacy_df(n=5, start="2024-06-01")

    merged = merge_with_legacy(api_df, legacy_df)

    api_only = merged[merged["is_sensor_reading"] == False]
    # 10 total - 5 overlap = 5 API-only rows
    assert len(api_only) == 5


def test_merge_no_legacy_overlap():
    api_df = make_api_df(n=5, start="2024-06-01")
    # Legacy CSV covers completely different dates
    legacy_df = make_legacy_df(n=5, start="2024-08-01")

    merged = merge_with_legacy(api_df, legacy_df)

    assert len(merged) == 10  # outer join: 5 + 5
    assert "temp_c" in merged.columns


def test_merge_preserves_app_predictions():
    api_df = make_api_df(n=10)
    legacy_df = make_legacy_df(n=5)

    merged = merge_with_legacy(api_df, legacy_df)

    assert "app_pred_day1" in merged.columns
    assert "app_pred_day2" in merged.columns
    assert "app_pred_day3" in merged.columns


def test_merge_sorted_by_date():
    api_df = make_api_df(n=5)
    legacy_df = make_legacy_df(n=3)

    merged = merge_with_legacy(api_df, legacy_df)

    assert merged["date"].is_monotonic_increasing


# ── detect_and_fill_gaps ──────────────────────────────────────────────────────

def test_detect_gaps_fills_missing_date():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-03"]),  # gap on June 2
        "temp_c": [26.0, 27.0],
        "is_sensor_reading": [True, True],
    })

    filled, n_gaps = detect_and_fill_gaps(df)

    assert n_gaps == 1
    assert len(filled) == 3
    assert pd.to_datetime("2024-06-02") in pd.to_datetime(filled["date"]).values


def test_detect_gaps_marks_inserted_rows():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-03"]),
        "temp_c": [26.0, 27.0],
        "is_sensor_reading": [True, True],
    })

    filled, _ = detect_and_fill_gaps(df)

    # The inserted June 2 row should have gap_filled = True
    gap_row = filled[pd.to_datetime(filled["date"]).dt.date == pd.Timestamp("2024-06-02").date()]
    assert gap_row["gap_filled"].iloc[0] == True


def test_detect_gaps_no_gaps():
    df = pd.DataFrame({
        "date": pd.date_range("2024-06-01", periods=5, freq="D"),
        "temp_c": range(5),
        "is_sensor_reading": [True] * 5,
    })

    filled, n_gaps = detect_and_fill_gaps(df)

    assert n_gaps == 0
    assert len(filled) == 5


def test_detect_gaps_forward_fills_values():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-04"]),
        "temp_c": [26.0, 29.0],
        "is_sensor_reading": [True, True],
    })

    filled, _ = detect_and_fill_gaps(df)

    # June 2 and 3 should be forward-filled from June 1
    assert filled.loc[1, "temp_c"] == pytest.approx(26.0)
    assert filled.loc[2, "temp_c"] == pytest.approx(26.0)


# ── validate_processed ────────────────────────────────────────────────────────

def test_validate_passes_good_df():
    df = pd.DataFrame({
        "date": pd.date_range("2024-06-01", periods=10, freq="D"),
        "temp_c": [26.0] * 10,
    })
    validate_processed(df)  # should not raise


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({"date": pd.date_range("2024-06-01", periods=5, freq="D")})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_processed(df)


def test_validate_raises_on_duplicate_dates():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        "temp_c": [26.0, 27.0],
    })
    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_processed(df)


# ── save_processed / load ─────────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2024-06-01", periods=5, freq="D"),
        "temp_c": [25.0, 26.0, 27.0, 28.0, 27.5],
    })
    path = str(tmp_path / "test.parquet")

    save_processed(df, output_path=path)

    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)
    assert loaded["temp_c"].tolist() == df["temp_c"].tolist()


# ── run_pipeline (integration) ────────────────────────────────────────────────

def test_run_pipeline_integration(tmp_path, monkeypatch):
    """
    Integration test: full pipeline from API df + legacy df → parquet output.
    Monkeypatches load_legacy_csv to return a fixed DataFrame.
    """
    from src.data import preprocess

    api_df = make_api_df(n=10)
    legacy_df = make_legacy_df(n=5)

    monkeypatch.setattr(preprocess, "load_legacy_csv", lambda path=None: legacy_df)

    output_path = str(tmp_path / "output.parquet")
    result = run_pipeline(api_df=api_df, output_path=output_path)

    assert isinstance(result, pd.DataFrame)
    assert "temp_c" in result.columns
    assert "is_sensor_reading" in result.columns
    assert "gap_filled" in result.columns
    assert Path(output_path).exists()
