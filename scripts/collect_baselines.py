#!/usr/bin/env python3
"""
Daily Baseline Collection Script

Collects weather forecasts from all sources (Open-Meteo, OpenWeatherMap,
AccuWeather, ThermoSense) and stores them for later comparison.

Run as cron job at 6 PM daily:
    0 18 * * * cd /path/to/project && /path/to/.venv/bin/python scripts/collect_baselines.py

Or run manually:
    python scripts/collect_baselines.py
    python scripts/collect_baselines.py --status
    python scripts/collect_baselines.py --leaderboard
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.baseline_collector import (
    collect_all_baselines,
    get_collection_status,
    get_leaderboard,
    init_database,
)


def main():
    parser = argparse.ArgumentParser(description="ThermoSense Daily Baseline Collection")
    parser.add_argument(
        "--status", action="store_true",
        help="Show collection system status"
    )
    parser.add_argument(
        "--leaderboard", action="store_true",
        help="Show accuracy leaderboard"
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help="Leaderboard window in days (default: 30)"
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Forecast horizon for leaderboard (default: 1)"
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Initialize database only"
    )
    parser.add_argument(
        "--thermosense-url", type=str, 
        default=os.environ.get("THERMOSENSE_API_URL", "http://localhost:8000"),
        help="ThermoSense API URL"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    if args.init:
        path = init_database()
        if args.json:
            print(json.dumps({"initialized": True, "path": str(path)}))
        else:
            print(f"Database initialized: {path}")
        return
    
    if args.status:
        status = get_collection_status()
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print("\n" + "=" * 60)
            print("  BASELINE COLLECTION STATUS")
            print("=" * 60)
            if status.get("initialized"):
                print(f"  Database: {status['database_path']}")
                print(f"  Forecasts stored: {status['forecast_count']}")
                print(f"  Actuals stored: {status['actual_count']}")
                if status.get("date_range"):
                    print(f"  Date range: {status['date_range']['earliest']} to {status['date_range']['latest']}")
                print("\n  Sources:")
                for source, info in status.get("sources", {}).items():
                    print(f"    {source}: last collected {info['last_collected']}, "
                          f"{info['successes']} successes, {info['failures']} failures")
            else:
                print("  Database not initialized. Run with --init first.")
            print("=" * 60 + "\n")
        return
    
    if args.leaderboard:
        board = get_leaderboard(window_days=args.window, horizon=args.horizon)
        if args.json:
            print(json.dumps(board, indent=2))
        else:
            if board:
                print(f"\n{'=' * 65}")
                print(f"  ACCURACY LEADERBOARD (Day-{args.horizon}, Last {args.window} Days)")
                print(f"{'=' * 65}")
                print(f"  {'Rank':<6}{'Source':<20}{'RMSE (°C)':<12}{'MAE (°C)':<12}{'Days':<8}")
                print(f"  {'-' * 58}")
                for row in board:
                    rank = "🥇" if row["rank"] == 1 else "🥈" if row["rank"] == 2 else "🥉" if row["rank"] == 3 else str(row["rank"])
                    print(f"  {rank:<6}{row['source']:<20}{row['rmse']:<12}{row['mae']:<12}{row['n_days']:<8}")
                
                ts = next((r for r in board if r["source"] == "thermosense"), None)
                best_other = next((r for r in board if r["source"] != "thermosense"), None)
                
                if ts and best_other and ts["rmse"] > 0:
                    improvement = (1 - ts["rmse"] / best_other["rmse"]) * 100
                    print(f"\n  ThermoSense vs {best_other['source']}: {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")
                
                print(f"{'=' * 65}\n")
            else:
                print("\nNo leaderboard data available yet.")
                print("Need both forecasts and actuals to compute rankings.\n")
        return
    
    print(f"\n[{date.today()}] Starting baseline collection...")
    results = collect_all_baselines(thermosense_url=args.thermosense_url)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 50)
        print("  COLLECTION SUMMARY")
        print("=" * 50)
        for source, info in results["sources"].items():
            status = "✓" if info.get("success") else "✗"
            if info.get("success"):
                print(f"  {status} {source}: {info['fetched']} fetched, {info['stored']} stored")
            else:
                print(f"  {status} {source}: {info.get('error', 'unknown error')}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
