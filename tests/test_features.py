"""Tests for feature engineering — verifies no data leakage and correct shapes."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_api_bias_feature,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
)


def make_sample_df(n=60):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "temp_c": np.random.uniform(24, 32, n),
        "humidity_pct": np.random.uniform(60, 90, n),
        "pressure_hpa": np.random.uniform(1005, 1015, n),
        "dewpoint_c": np.random.uniform(16, 22, n),
        "precip_mm": np.random.uniform(0, 10, n),
        "cloudcover_pct": np.random.uniform(0, 100, n),
        "windspeed_kmh": np.random.uniform(5, 25, n),
        "uv_index": np.random.uniform(0, 8, n),
        "temp_c_api": np.random.uniform(24, 32, n),
    })


def test_lag_features_no_leakage():
    df = make_sample_df()
    df = add_lag_features(df, col="temp_c", lags=[1, 2])
    # Row 0 should be NaN for lag-1 (no prior data available)
    assert pd.isna(df["temp_c_lag1"].iloc[0])
    # Row 1 should equal row 0's actual temp
    assert df["temp_c_lag1"].iloc[1] == df["temp_c"].iloc[0]


def test_rolling_features_no_leakage():
    df = make_sample_df()
    df = add_rolling_features(df, col="temp_c", windows=[3])
    # Rolling features use shift(1) — row 0 should be NaN
    assert pd.isna(df["temp_c_roll3_mean"].iloc[0])


def test_calendar_features_monsoon():
    df = make_sample_df()
    df["date"] = pd.date_range("2024-06-01", periods=len(df), freq="D")
    df = add_calendar_features(df)
    # June is monsoon season
    assert df["is_monsoon"].iloc[0] == 1


def test_api_bias_feature():
    df = make_sample_df()
    df = add_api_bias_feature(df, sensor_col="temp_c", api_col="temp_c_api")
    assert "api_bias" in df.columns
    assert "api_bias_roll7_mean" in df.columns
    # Bias = sensor - api; verify computation for first non-NaN row
    idx = df["api_bias"].first_valid_index()
    expected = df["temp_c"].iloc[idx] - df["temp_c_api"].iloc[idx]
    assert abs(df["api_bias"].iloc[idx] - expected) < 1e-9


def test_build_feature_matrix_shape():
    df = make_sample_df(n=60)
    result = build_feature_matrix(df, drop_na=True)
    # After dropping NaN rows from lags (max lag=14), should have fewer rows
    assert len(result) < 60
    assert "temp_c" in result.columns
    assert "time_idx" in result.columns
    assert "location" in result.columns
