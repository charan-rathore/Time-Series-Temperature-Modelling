"""
Feature engineering for ThermoSense.

Transforms the merged daily DataFrame into a rich feature matrix
for use by LightGBM, TFT, and the ensemble stacker.

Key innovation: the api_bias feature captures the systematic offset
between our sensor location and the Open-Meteo API's nearest station,
allowing the model to learn hyperlocal microclimate corrections.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7, 14]
MONSOON_MONTHS = {6, 7, 8, 9}


def add_lag_features(df: pd.DataFrame, col: str, lags: List[int] = LAG_DAYS) -> pd.DataFrame:
    """Add autoregressive lag features for a given column."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, col: str, windows: List[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Add rolling mean, std, min, max over specified windows."""
    for w in windows:
        df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w).std()
    df[f"{col}_roll7_max"] = df[col].shift(1).rolling(7).max()
    df[f"{col}_roll7_min"] = df[col].shift(1).rolling(7).min()
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features with cyclic encoding for seasonality."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["day_of_year"] = dates.dt.dayofyear
    df["month"] = dates.dt.month
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["day_of_week"] = dates.dt.dayofweek
    df["is_monsoon"] = dates.dt.month.isin(MONSOON_MONTHS).astype(int)

    # Cyclic encoding - avoids the discontinuity at Dec 31 / Jan 1
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_api_bias_feature(
    df: pd.DataFrame,
    sensor_col: str = "temp_c",
    api_col: str = "temp_c_api",
    bias_window: int = 7,
) -> pd.DataFrame:
    """
    Compute the local sensor bias relative to the Open-Meteo API.

    api_bias = sensor_actual_temp - open_meteo_api_temp

    A positive bias means our location runs warmer than the API's nearest station
    (e.g., urban heat island effect). The rolling mean captures systematic drift.

    This is the key feature that enables hyperlocal correction without retraining
    the underlying NWP model.
    """
    if api_col not in df.columns:
        return df

    df["api_bias"] = df[sensor_col] - df[api_col]
    df["api_bias_roll7_mean"] = df["api_bias"].shift(1).rolling(bias_window).mean()
    df["api_bias_roll7_std"] = df["api_bias"].shift(1).rolling(bias_window).std()
    return df


def add_time_idx(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add an integer time index starting at 0.
    Required by pytorch-forecasting's TimeSeriesDataSet.
    """
    df = df.copy()
    df["time_idx"] = (
        pd.to_datetime(df[date_col]) - pd.to_datetime(df[date_col]).min()
    ).dt.days
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    sensor_col: str = "temp_c",
    api_col: Optional[str] = "temp_c_api",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
        1. Calendar features (cyclic encoding of day/month)
        2. Lag features for temperature and humidity
        3. Rolling statistics for temperature
        4. API bias feature (hyperlocal correction signal)
        5. Integer time index for TFT
        6. Drop rows with NaN from lags (leading rows)

    Args:
        df: Processed daily DataFrame from preprocess.py.
        sensor_col: Column name for sensor/actual temperature.
        api_col: Column name for API-predicted temperature (for bias feature).
        drop_na: Whether to drop rows with NaN from lag/rolling features.

    Returns:
        Feature-rich DataFrame ready for model training.
    """
    LEGACY_COLS = {"app_pred_day1", "app_pred_day2", "app_pred_day3"}

    df = df.copy()
    df = df.drop(columns=[c for c in LEGACY_COLS if c in df.columns])
    df = add_calendar_features(df)
    df = add_lag_features(df, col=sensor_col)

    if "humidity_pct" in df.columns:
        df = add_lag_features(df, col="humidity_pct", lags=[1])
    if "pressure_hpa" in df.columns:
        df = add_lag_features(df, col="pressure_hpa", lags=[1])

    df = add_rolling_features(df, col=sensor_col)
    df = add_api_bias_feature(df, sensor_col=sensor_col, api_col=api_col or "")
    df = add_time_idx(df)
    df["location"] = "sensor"

    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df
