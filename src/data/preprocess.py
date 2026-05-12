"""
Data preprocessing for ThermoSense — Phase 1.

Responsibilities:
  1. Load the legacy 40-day hand-recorded CSV (existing sensor data).
  2. Merge it with Open-Meteo API data — sensor readings take precedence
     for the overlap window since they are real local measurements.
  3. Detect and fill date gaps via forward-fill (flagging them in a column).
  4. Save the merged, clean DataFrame to data/processed/ as parquet.

The run_pipeline() function is the single entry point used by the CLI
script (scripts/run_pipeline.py) and by notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

LEGACY_COL_RENAME = {
    "Actual Temp": "temp_c",
    "Predicted_temp_day1": "app_pred_day1",
    "Predicted_temp_day2": "app_pred_day2",
    "Predicted_temp_day3": "app_pred_day3",
}

PROCESSED_FILENAME = "daily_merged.parquet"


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_legacy_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the original 40-day hand-recorded sensor CSV.

    Renames columns to the project-wide schema and parses dates.
    The 'Date' column in the original CSV uses D/M/YYYY format (dayfirst=True).

    Args:
        csv_path: Path to the CSV. Defaults to data/legacy/temperature-data-for-TSA.csv.

    Returns:
        DataFrame with columns: date, temp_c, app_pred_day1/2/3
    """
    cfg = _load_config()
    if csv_path is None:
        legacy_dir = _PROJECT_ROOT / cfg["data"]["legacy_dir"]
        csv_path = str(legacy_dir / "temperature-data-for-TSA.csv")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in LEGACY_COL_RENAME.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.drop(columns=["Date"], errors="ignore")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[preprocess] Loaded legacy CSV: {len(df)} rows "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def load_processed(path: Optional[str] = None) -> pd.DataFrame:
    """Load the merged processed DataFrame from parquet."""
    cfg = _load_config()
    if path is None:
        path = str(_PROJECT_ROOT / cfg["data"]["processed_dir"] / PROCESSED_FILENAME)
    df = pd.read_parquet(path)
    print(f"[preprocess] Loaded processed data: {len(df)} rows from {path}")
    return df


# ── Core pipeline steps ───────────────────────────────────────────────────────

def merge_with_legacy(
    api_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Open-Meteo historical data with the legacy hand-recorded CSV.

    Merge strategy:
    - Outer join on 'date' so both API-only and sensor-only dates are included.
    - For dates covered by the legacy CSV, the sensor's actual temperature
      (temp_c_sensor) takes precedence over the API temperature (temp_c_api)
      because sensor readings are true local measurements.
    - For all other dates (API only), temp_c_api is used.
    - Meteorological covariates (humidity, pressure, etc.) always come from
      the API since the legacy CSV did not record them.
    - A boolean 'is_sensor_reading' column marks rows with actual observations.

    Args:
        api_df: DataFrame from fetcher.fetch_historical_open_meteo().
        legacy_df: DataFrame from load_legacy_csv().

    Returns:
        Merged DataFrame with columns from both sources.
    """
    api_df = api_df.copy()
    legacy_df = legacy_df.copy()

    api_df["date"] = pd.to_datetime(api_df["date"])
    legacy_df["date"] = pd.to_datetime(legacy_df["date"])

    merged = pd.merge(
        api_df,
        legacy_df,
        on="date",
        how="outer",
        suffixes=("_api", "_sensor"),
    )

    # Prefer sensor temperature for actual readings
    if "temp_c_sensor" in merged.columns and "temp_c_api" in merged.columns:
        merged["is_sensor_reading"] = merged["temp_c_sensor"].notna()
        merged["temp_c"] = merged["temp_c_sensor"].fillna(merged["temp_c_api"])
        merged = merged.drop(columns=["temp_c_sensor", "temp_c_api"])
    elif "temp_c_api" in merged.columns:
        merged = merged.rename(columns={"temp_c_api": "temp_c"})
        merged["is_sensor_reading"] = False
    elif "temp_c_sensor" in merged.columns:
        merged = merged.rename(columns={"temp_c_sensor": "temp_c"})
        merged["is_sensor_reading"] = True
    else:
        merged["is_sensor_reading"] = False

    # Preserve original API temperature for bias feature computation
    if "temp_c" in api_df.columns:
        api_ref = api_df[["date", "temp_c"]].rename(columns={"temp_c": "temp_c_api"})
        merged = pd.merge(merged, api_ref, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    n_sensor = merged["is_sensor_reading"].sum()
    print(f"[preprocess] Merged: {len(merged)} total rows "
          f"({n_sensor} sensor readings, {len(merged) - n_sensor} API-only)")
    return merged


def detect_and_fill_gaps(
    df: pd.DataFrame,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, int]:
    """
    Detect missing calendar dates and forward-fill to bridge gaps.

    A 'gap_filled' boolean column marks rows that were inserted by this step
    so downstream analysis can weight or exclude them appropriately.

    Args:
        df: DataFrame with a 'date' column at daily frequency.
        date_col: Name of the date column.

    Returns:
        Tuple of (filled DataFrame, number of gaps inserted).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    existing = pd.to_datetime(df[date_col])
    missing_dates = full_range.difference(existing)
    n_gaps = len(missing_dates)

    if n_gaps > 0:
        print(f"[preprocess] Filling {n_gaps} missing date(s): "
              f"{[str(d.date()) for d in missing_dates]}")

    df["gap_filled"] = False
    df = (
        df.set_index(date_col)
        .reindex(full_range)
        .ffill()
        .reset_index()
        .rename(columns={"index": date_col})
    )
    # Mark inserted rows
    inserted = ~df[date_col].isin(existing)
    df.loc[inserted, "gap_filled"] = True
    df.loc[inserted, "is_sensor_reading"] = False

    return df, n_gaps


def validate_processed(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the processed DataFrame.

    Raises ValueError on hard failures; prints warnings for soft issues.
    """
    required_cols = ["date", "temp_c"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[preprocess] Missing required columns: {missing}")

    if df["date"].duplicated().any():
        raise ValueError("[preprocess] Duplicate dates found in processed DataFrame.")

    null_temp_pct = df["temp_c"].isna().mean() * 100
    if null_temp_pct > 5:
        print(f"[preprocess] WARNING: {null_temp_pct:.1f}% of temp_c values are null.")

    temp_range = (df["temp_c"].min(), df["temp_c"].max())
    if temp_range[0] < -10 or temp_range[1] > 55:
        print(f"[preprocess] WARNING: temp_c range {temp_range} looks unusual for India.")

    n_sensor = df.get("is_sensor_reading", pd.Series(dtype=bool)).sum()
    print(f"[preprocess] Validation passed — {len(df)} rows, "
          f"{n_sensor} sensor readings, temp range {temp_range[0]:.1f}–{temp_range[1]:.1f}°C")


def save_processed(df: pd.DataFrame, output_path: Optional[str] = None) -> Path:
    """
    Save the processed DataFrame to parquet.

    Args:
        df: Processed DataFrame to save.
        output_path: Override output path. Defaults to data/processed/daily_merged.parquet.

    Returns:
        Path object of the saved file.
    """
    cfg = _load_config()
    if output_path is None:
        out_dir = _PROJECT_ROOT / cfg["data"]["processed_dir"]
        output_path = str(out_dir / PROCESSED_FILENAME)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    try:
        display_path = path.relative_to(_PROJECT_ROOT)
    except ValueError:
        display_path = path
    print(f"[preprocess] Saved → {display_path} ({len(df)} rows)")
    return path


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(
    api_df: pd.DataFrame,
    legacy_csv_path: Optional[str] = None,
    output_path: Optional[str] = None,
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → merge → validate → gap-fill → save.

    This is the single function called by scripts/run_pipeline.py and
    by notebooks/01_eda.ipynb after a fetch.

    Args:
        api_df: Raw daily DataFrame from fetcher.fetch_historical_open_meteo().
        legacy_csv_path: Path to legacy sensor CSV. Defaults to auto-detected path.
        output_path: Where to save the processed parquet. Defaults to auto-detected.
        fill_gaps: Whether to forward-fill missing calendar dates.

    Returns:
        The fully processed and validated DataFrame.
    """
    print("\n[preprocess] === Pipeline start ===")

    legacy_df = load_legacy_csv(legacy_csv_path)
    merged = merge_with_legacy(api_df, legacy_df)

    if fill_gaps:
        merged, n_gaps = detect_and_fill_gaps(merged)
        if n_gaps > 0:
            print(f"[preprocess] Gap-filled {n_gaps} missing days via forward-fill.")

    validate_processed(merged)
    save_processed(merged, output_path)

    print("[preprocess] === Pipeline complete ===\n")
    return merged
