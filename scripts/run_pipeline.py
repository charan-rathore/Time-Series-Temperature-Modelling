#!/usr/bin/env python3
"""
ThermoSense — Data Pipeline CLI (Phase 1)

This script is the main entry point for running the data ingestion and
preprocessing pipeline. It supports two operating modes:

  BACKFILL mode (first-time setup):
    Pulls 365 days of historical data from Open-Meteo archive, merges it
    with the legacy 40-day sensor CSV, fills any date gaps, and writes
    the result to data/processed/daily_merged.parquet.

    Run:
        python scripts/run_pipeline.py --mode backfill

    With custom date range:
        python scripts/run_pipeline.py --mode backfill --start 2023-06-01 --end 2024-07-11

  DAILY mode (scheduled nightly fetch):
    Fetches the latest day's data from Open-Meteo, appends it to the
    existing processed parquet, and refreshes the forecast data for the
    next 7 days.

    Run:
        python scripts/run_pipeline.py --mode daily

Usage summary:
    python scripts/run_pipeline.py --mode [backfill|daily] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # OWM_API_KEY is optional
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on the path when running from any directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.fetcher import backfill, fetch_forecast_open_meteo, fetch_owm_current
from src.data.preprocess import load_processed, run_pipeline, save_processed


def cmd_backfill(args: argparse.Namespace) -> None:
    """
    Full historical backfill pipeline.

    Fetches Open-Meteo archive data, merges with legacy sensor CSV,
    fills gaps, validates, and saves to data/processed/daily_merged.parquet.
    """
    print("=" * 60)
    print("  ThermoSense — BACKFILL mode")
    print("=" * 60)

    api_df = backfill(
        start_date=args.start,
        end_date=args.end,
    )

    processed = run_pipeline(
        api_df=api_df,
        legacy_csv_path=args.legacy_csv,
    )

    print(f"\nBackfill complete: {len(processed)} days of data ready.")
    print(f"  Sensor readings:  {processed['is_sensor_reading'].sum()}")
    print(f"  API-only rows:    {(~processed['is_sensor_reading']).sum()}")
    print(f"  Gap-filled rows:  {processed['gap_filled'].sum()}")
    print(f"  Date range:       {processed['date'].min().date()} → "
          f"{processed['date'].max().date()}")
    print(f"\nOutput: data/processed/daily_merged.parquet")
    print("\nNext step: run Phase 2 (feature engineering)")
    print("  python -c \"from src.features.engineer import build_feature_matrix; "
          "import pandas as pd; df = pd.read_parquet('data/processed/daily_merged.parquet'); "
          "print(build_feature_matrix(df).shape)\"")


def cmd_daily(args: argparse.Namespace) -> None:
    """
    Daily incremental update pipeline.

    Fetches the most recent day from Open-Meteo, appends it to the
    existing processed DataFrame, and pulls a fresh 7-day forecast.
    """
    print("=" * 60)
    print("  ThermoSense — DAILY update mode")
    print("=" * 60)

    processed_path = _PROJECT_ROOT / "data" / "processed" / "daily_merged.parquet"
    if not processed_path.exists():
        print("\nERROR: No processed data found. Run backfill first:")
        print("  python scripts/run_pipeline.py --mode backfill")
        sys.exit(1)

    existing = load_processed()
    last_date = existing["date"].max().date()
    today = date.today()
    yesterday = today - timedelta(days=1)

    if last_date >= yesterday:
        print(f"\nData already up to date (last entry: {last_date}). Nothing to fetch.")
    else:
        fetch_start = (last_date + timedelta(days=1)).isoformat()
        fetch_end = yesterday.isoformat()
        print(f"\nFetching new data: {fetch_start} → {fetch_end}")

        from src.data.fetcher import fetch_historical_open_meteo
        new_rows = fetch_historical_open_meteo(
            start_date=fetch_start,
            end_date=fetch_end,
        )

        import pandas as pd
        updated = pd.concat([existing, new_rows], ignore_index=True)
        updated = updated.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        save_processed(updated)
        print(f"\nAppended {len(new_rows)} new row(s). Total: {len(updated)} rows.")

    # Always refresh the forecast for the next 7 days
    print("\nFetching 7-day forecast for covariate prediction…")
    forecast_df = fetch_forecast_open_meteo(forecast_days=7)
    forecast_path = _PROJECT_ROOT / "data" / "processed" / "forecast_7day.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    print(f"Forecast saved → data/processed/forecast_7day.parquet ({len(forecast_df)} rows)")

    # Optionally fetch OWM baseline
    owm = fetch_owm_current()
    if owm:
        print(f"\nOWM current: {owm['temp_c']:.1f}°C, {owm['description']}")

    print("\nDaily update complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThermoSense Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="backfill",
        help="Pipeline mode: 'backfill' for first-time setup, 'daily' for incremental update.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="(Backfill only) Start date YYYY-MM-DD. Defaults to 365 days ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="(Backfill only) End date YYYY-MM-DD. Defaults to yesterday.",
    )
    parser.add_argument(
        "--legacy-csv",
        dest="legacy_csv",
        type=str,
        default=None,
        help="Path to legacy sensor CSV. Defaults to data/legacy/temperature-data-for-TSA.csv.",
    )

    args = parser.parse_args()

    if args.mode == "backfill":
        cmd_backfill(args)
    elif args.mode == "daily":
        cmd_daily(args)


if __name__ == "__main__":
    main()
