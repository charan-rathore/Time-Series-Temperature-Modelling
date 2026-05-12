#!/usr/bin/env python3
"""
Weekly Accuracy Report Generator

Generates a comprehensive weekly report comparing ThermoSense performance
against commercial weather services. The report includes:
- Rolling 30-day RMSE per source
- Statistical significance tests (p-values)
- Effect sizes (Cohen's d)
- Sensor data completeness
- Worst prediction days
- Resume-ready statements

Run manually:
    python scripts/generate_report.py

Or as a weekly cron job (Sunday at midnight):
    0 0 * * 0 cd /path/to/project && /path/to/.venv/bin/python scripts/generate_report.py

Output is saved to data/reports/weekly_report_YYYY-MM-DD.json
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.baseline_collector import (
    _DB_PATH,
    get_leaderboard,
    get_collection_status,
    init_database,
)
from src.evaluation.statistical_tests import (
    compare_forecasters,
    diebold_mariano_test,
    compute_skill_score,
    summary_statistics,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPORTS_DIR = _PROJECT_ROOT / "data" / "reports"


def get_errors_for_source(
    db_path: Path,
    source: str,
    horizon: int = 1,
    window_days: int = 30,
) -> List[float]:
    """Get absolute errors for a source."""
    if not db_path.exists():
        return []
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        SELECT ABS(f.predicted_temp_c - a.sensor_temp_c) as error
        FROM daily_forecasts f
        JOIN daily_actuals a ON f.forecast_date = a.date
        WHERE f.source = ? AND f.horizon_days = ? AND f.forecast_date >= ?
        ORDER BY f.forecast_date
        """,
        (source, horizon, cutoff),
    )
    errors = [row[0] for row in cursor.fetchall()]
    conn.close()
    return errors


def get_worst_predictions(
    db_path: Path,
    source: str = "thermosense",
    horizon: int = 1,
    window_days: int = 30,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Get the days with the largest prediction errors."""
    if not db_path.exists():
        return []
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT 
            f.forecast_date,
            f.predicted_temp_c,
            a.sensor_temp_c as actual_temp_c,
            ABS(f.predicted_temp_c - a.sensor_temp_c) as error
        FROM daily_forecasts f
        JOIN daily_actuals a ON f.forecast_date = a.date
        WHERE f.source = ? AND f.horizon_days = ? AND f.forecast_date >= ?
        ORDER BY error DESC
        LIMIT ?
        """,
        (source, horizon, cutoff, limit),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_data_completeness(
    db_path: Path,
    window_days: int = 30,
) -> Dict[str, Any]:
    """Calculate data completeness metrics."""
    if not db_path.exists():
        return {"completeness_pct": 0}
    
    expected_days = window_days
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(db_path))
    
    cursor = conn.execute(
        "SELECT COUNT(*) as count FROM daily_actuals WHERE date >= ?",
        (cutoff,)
    )
    actual_count = cursor.fetchone()[0]
    
    cursor = conn.execute(
        """
        SELECT source, COUNT(DISTINCT forecast_date) as days
        FROM daily_forecasts
        WHERE forecast_date >= ?
        GROUP BY source
        """,
        (cutoff,)
    )
    forecast_counts = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    completeness = {
        "window_days": window_days,
        "sensor_days": actual_count,
        "sensor_completeness_pct": round(actual_count / expected_days * 100, 1),
        "forecast_days_by_source": forecast_counts,
    }
    
    return completeness


def get_available_sources(db_path: Path) -> List[str]:
    """Get list of forecast sources."""
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT DISTINCT source FROM daily_forecasts")
    sources = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sources


def generate_weekly_report(
    db_path: Optional[Path] = None,
    window_days: int = 30,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Generate a comprehensive weekly accuracy report.
    
    Returns a dictionary containing:
    - Summary metrics
    - Leaderboard rankings
    - Statistical comparisons
    - Data quality metrics
    - Worst predictions
    - Resume-ready statements
    """
    db_path = db_path or _DB_PATH
    init_database(db_path)
    
    report = {
        "report_type": "weekly_accuracy_report",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_date": date.today().isoformat(),
        "window_days": window_days,
        "horizon_days": horizon,
    }
    
    report["leaderboard"] = get_leaderboard(db_path, window_days, horizon)
    
    sources = get_available_sources(db_path)
    non_ts_sources = [s for s in sources if s != "thermosense"]
    
    ts_errors = get_errors_for_source(db_path, "thermosense", horizon, window_days)
    
    comparisons = []
    for baseline in non_ts_sources:
        bl_errors = get_errors_for_source(db_path, baseline, horizon, window_days)
        
        if len(bl_errors) < 3 or len(ts_errors) < 3:
            continue
        
        min_len = min(len(ts_errors), len(bl_errors))
        ts_subset = ts_errors[:min_len]
        bl_subset = bl_errors[:min_len]
        
        result = compare_forecasters(ts_subset, bl_subset, alternative="less")
        skill = compute_skill_score(ts_subset, bl_subset)
        
        comparisons.append({
            "baseline": baseline,
            "n_samples": min_len,
            "thermosense_rmse": result.get("thermosense_rmse"),
            "baseline_rmse": result.get("baseline_rmse"),
            "mean_improvement_c": result.get("mean_improvement_c"),
            "pct_improvement": result.get("pct_improvement"),
            "p_value": result.get("p_value"),
            "significant": bool(result.get("significant")),
            "effect_size_d": result.get("effect_size_d"),
            "effect_interpretation": result.get("effect_interpretation"),
            "skill_score": skill,
        })
    
    comparisons.sort(key=lambda x: x.get("p_value") or 999)
    report["statistical_comparisons"] = comparisons
    
    report["data_quality"] = get_data_completeness(db_path, window_days)
    
    report["worst_predictions"] = get_worst_predictions(
        db_path, "thermosense", horizon, window_days, limit=5
    )
    
    source_summaries = {}
    for source in sources:
        errors = get_errors_for_source(db_path, source, horizon, window_days)
        if errors:
            source_summaries[source] = summary_statistics(errors)
    report["source_summaries"] = source_summaries
    
    statements = []
    for comp in comparisons:
        if comp["significant"] and comp["pct_improvement"] and comp["pct_improvement"] > 10:
            p_str = f"{comp['p_value']:.4f}" if comp['p_value'] >= 0.0001 else "< 0.0001"
            stmt = (
                f"ThermoSense achieves {comp['thermosense_rmse']:.2f}°C RMSE on Day-{horizon} forecasts, "
                f"a {comp['pct_improvement']:.0f}% improvement over {comp['baseline'].replace('_', ' ').title()} "
                f"(p = {p_str}, n={comp['n_samples']}, Cohen's d={comp['effect_size_d']:.1f})."
            )
            statements.append({
                "baseline": comp["baseline"],
                "statement": stmt,
            })
    report["resume_ready_statements"] = statements
    
    ts_rank = None
    best_commercial = None
    improvement_vs_best = None
    
    for entry in report["leaderboard"]:
        if entry["source"] == "thermosense":
            ts_rank = entry["rank"]
        elif best_commercial is None:
            best_commercial = entry
    
    if ts_rank and best_commercial:
        ts_entry = next(e for e in report["leaderboard"] if e["source"] == "thermosense")
        if ts_entry["rmse"] > 0 and best_commercial["rmse"] > 0:
            improvement_vs_best = round(
                (1 - ts_entry["rmse"] / best_commercial["rmse"]) * 100, 1
            )
    
    report["summary"] = {
        "thermosense_rank": ts_rank,
        "total_sources": len(report["leaderboard"]),
        "best_commercial": best_commercial["source"] if best_commercial else None,
        "improvement_vs_best_pct": improvement_vs_best,
        "significant_improvements": sum(1 for c in comparisons if c["significant"]),
        "data_completeness_pct": report["data_quality"]["sensor_completeness_pct"],
    }
    
    return report


def save_report(report: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Save report to JSON file."""
    output_dir = output_dir or _REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"weekly_report_{date.today().isoformat()}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return filepath


def print_report_summary(report: Dict[str, Any]):
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 70)
    print("  WEEKLY ACCURACY REPORT")
    print(f"  Generated: {report['generated_at']}")
    print(f"  Window: {report['window_days']} days | Horizon: Day-{report['horizon_days']}")
    print("=" * 70)
    
    summary = report.get("summary", {})
    print(f"\n  ThermoSense Rank: #{summary.get('thermosense_rank', 'N/A')} of {summary.get('total_sources', 0)}")
    print(f"  Best Commercial: {summary.get('best_commercial', 'N/A')}")
    print(f"  Improvement vs Best: {summary.get('improvement_vs_best_pct', 'N/A')}%")
    print(f"  Significant Improvements: {summary.get('significant_improvements', 0)}")
    print(f"  Data Completeness: {summary.get('data_completeness_pct', 0)}%")
    
    print("\n" + "-" * 70)
    print("  LEADERBOARD")
    print("-" * 70)
    print(f"  {'Rank':<6}{'Source':<20}{'RMSE':<12}{'MAE':<12}{'Days':<8}")
    print(f"  {'-'*58}")
    
    for entry in report.get("leaderboard", []):
        rank = "🥇" if entry["rank"] == 1 else "🥈" if entry["rank"] == 2 else "🥉" if entry["rank"] == 3 else str(entry["rank"])
        print(f"  {rank:<6}{entry['source']:<20}{entry['rmse']:<12}{entry['mae']:<12}{entry['n_days']:<8}")
    
    print("\n" + "-" * 70)
    print("  STATISTICAL COMPARISONS")
    print("-" * 70)
    
    for comp in report.get("statistical_comparisons", []):
        sig = "✓" if comp["significant"] else "✗"
        print(f"  {sig} vs {comp['baseline']:<18} "
              f"Improvement: {comp['pct_improvement']:.1f}% "
              f"(p={comp['p_value']:.4f}, d={comp['effect_size_d']:.2f})")
    
    if report.get("resume_ready_statements"):
        print("\n" + "-" * 70)
        print("  RESUME-READY STATEMENTS")
        print("-" * 70)
        for stmt in report["resume_ready_statements"]:
            print(f"\n  > {stmt['statement']}")
    
    if report.get("worst_predictions"):
        print("\n" + "-" * 70)
        print("  WORST PREDICTIONS (for improvement analysis)")
        print("-" * 70)
        for pred in report["worst_predictions"]:
            print(f"  {pred['forecast_date']}: "
                  f"Predicted {pred['predicted_temp_c']:.1f}°C, "
                  f"Actual {pred['actual_temp_c']:.1f}°C, "
                  f"Error {pred['error']:.2f}°C")
    
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Weekly Accuracy Report")
    parser.add_argument(
        "--window", type=int, default=30,
        help="Number of days to include (default: 30)"
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Forecast horizon (default: 1)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report to file"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory for saved reports"
    )
    
    args = parser.parse_args()
    
    report = generate_weekly_report(
        window_days=args.window,
        horizon=args.horizon,
    )
    
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report_summary(report)
    
    if args.save:
        output_dir = Path(args.output_dir) if args.output_dir else None
        filepath = save_report(report, output_dir)
        print(f"Report saved to: {filepath}")


if __name__ == "__main__":
    main()
