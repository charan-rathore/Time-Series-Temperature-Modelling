"""
Statistics API routes - Statistical validation of forecast comparisons

Provides rigorous statistical testing to validate claims like
"ThermoSense beats Google Weather by X%".

Endpoints:
    GET  /api/statistics/compare           - Compare ThermoSense vs a baseline
    GET  /api/statistics/all-comparisons   - Compare ThermoSense vs all sources
    GET  /api/statistics/summary           - Summary statistics for all sources
"""

import sqlite3
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from src.data.baseline_collector import _DB_PATH, init_database
from src.evaluation.statistical_tests import (
    compare_forecasters,
    diebold_mariano_test,
    compute_skill_score,
    summary_statistics,
)


def _to_python_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

router = APIRouter()


def _get_errors_for_source(
    source: str,
    horizon: int = 1,
    window_days: int = 30,
) -> List[float]:
    """Get absolute errors for a source from the database."""
    if not _DB_PATH.exists():
        return []
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(_DB_PATH))
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


def _get_signed_errors_for_source(
    source: str,
    horizon: int = 1,
    window_days: int = 30,
) -> List[float]:
    """Get signed errors (predicted - actual) for a source from the database."""
    if not _DB_PATH.exists():
        return []
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(_DB_PATH))
    cursor = conn.execute(
        """
        SELECT f.predicted_temp_c - a.sensor_temp_c as error
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


def _get_available_sources() -> List[str]:
    """Get list of sources with data in the database."""
    if not _DB_PATH.exists():
        return []
    
    conn = sqlite3.connect(str(_DB_PATH))
    cursor = conn.execute(
        "SELECT DISTINCT source FROM daily_forecasts ORDER BY source"
    )
    sources = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sources


@router.get("/compare")
def compare_to_baseline(
    baseline: str = Query(..., description="Baseline source to compare against (e.g., 'open_meteo')"),
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon"),
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
):
    """
    Compare ThermoSense forecast accuracy against a baseline using statistical tests.
    
    Returns:
    - Paired t-test or Wilcoxon signed-rank test results
    - Effect size (Cohen's d)
    - 95% confidence interval on improvement
    - Diebold-Mariano test results
    - Skill score
    """
    init_database(_DB_PATH)
    
    ts_errors = _get_errors_for_source("thermosense", horizon, window_days)
    bl_errors = _get_errors_for_source(baseline, horizon, window_days)
    
    if len(ts_errors) == 0:
        raise HTTPException(status_code=404, detail="No ThermoSense forecast data found")
    if len(bl_errors) == 0:
        raise HTTPException(status_code=404, detail=f"No {baseline} forecast data found")
    
    min_len = min(len(ts_errors), len(bl_errors))
    ts_errors = ts_errors[:min_len]
    bl_errors = bl_errors[:min_len]
    
    if min_len < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 matched pairs, only have {min_len}"
        )
    
    comparison = _to_python_types(compare_forecasters(ts_errors, bl_errors, alternative="less"))
    
    ts_signed = _get_signed_errors_for_source("thermosense", horizon, window_days)[:min_len]
    bl_signed = _get_signed_errors_for_source(baseline, horizon, window_days)[:min_len]
    dm_result = _to_python_types(diebold_mariano_test(ts_signed, bl_signed, h=horizon))
    
    skill = float(compute_skill_score(ts_errors, bl_errors))
    
    ts_summary = _to_python_types(summary_statistics(ts_errors))
    bl_summary = _to_python_types(summary_statistics(bl_errors))
    
    return {
        "comparison": {
            "thermosense_vs": baseline,
            "horizon_days": horizon,
            "window_days": window_days,
            "n_paired_samples": min_len,
        },
        "statistical_test": {
            "test_used": comparison.get("test_used", "unknown"),
            "alternative_hypothesis": "ThermoSense errors are lower",
            "mean_improvement_c": comparison.get("mean_improvement_c"),
            "ci_95_low": comparison.get("ci_95_low"),
            "ci_95_high": comparison.get("ci_95_high"),
            "t_statistic": comparison.get("t_statistic"),
            "p_value": comparison.get("p_value"),
            "significant_at_005": comparison.get("significant"),
        },
        "effect_size": {
            "cohens_d": comparison.get("effect_size_d"),
            "interpretation": comparison.get("effect_interpretation"),
        },
        "diebold_mariano_test": {
            "dm_statistic": dm_result.get("dm_statistic"),
            "p_value": dm_result.get("p_value"),
            "thermosense_better": dm_result.get("forecaster_1_better"),
        },
        "skill_score": skill,
        "percentage_improvement": comparison.get("pct_improvement"),
        "thermosense_stats": ts_summary,
        "baseline_stats": bl_summary,
        "resume_ready_statement": _generate_resume_statement(
            comparison, baseline, horizon, min_len
        ),
    }


def _generate_resume_statement(comparison: dict, baseline: str, horizon: int, n: int) -> Optional[str]:
    """Generate a resume-ready statement if results are significant."""
    if not comparison.get("significant"):
        return None
    
    ts_rmse = comparison.get("thermosense_rmse", 0)
    pct = comparison.get("pct_improvement", 0)
    p_val = comparison.get("p_value", 1)
    d = comparison.get("effect_size_d", 0)
    
    if pct and ts_rmse and p_val < 0.05:
        return (
            f"ThermoSense achieves {ts_rmse:.2f}°C RMSE on Day-{horizon} forecasts, "
            f"a {pct:.0f}% improvement over {baseline.replace('_', ' ').title()} "
            f"(p < 0.{int(-1 * (p_val * 10000)):04d}, n={n}, Cohen's d={d:.1f})."
        )
    return None


@router.get("/all-comparisons")
def compare_all(
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon"),
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
):
    """
    Compare ThermoSense against all available baseline sources.
    
    Returns comparison statistics for each source.
    """
    init_database(_DB_PATH)
    
    sources = _get_available_sources()
    sources = [s for s in sources if s != "thermosense"]
    
    if not sources:
        raise HTTPException(status_code=404, detail="No baseline sources found")
    
    ts_errors = _get_errors_for_source("thermosense", horizon, window_days)
    if len(ts_errors) == 0:
        raise HTTPException(status_code=404, detail="No ThermoSense forecast data found")
    
    comparisons = []
    
    for baseline in sources:
        bl_errors = _get_errors_for_source(baseline, horizon, window_days)
        
        if len(bl_errors) < 3:
            continue
        
        min_len = min(len(ts_errors), len(bl_errors))
        if min_len < 3:
            continue
        
        ts_subset = ts_errors[:min_len]
        bl_subset = bl_errors[:min_len]
        
        result = _to_python_types(compare_forecasters(ts_subset, bl_subset, alternative="less"))
        skill = float(compute_skill_score(ts_subset, bl_subset))
        
        comparisons.append({
            "baseline": baseline,
            "n_samples": min_len,
            "thermosense_rmse": result.get("thermosense_rmse"),
            "baseline_rmse": result.get("baseline_rmse"),
            "mean_improvement_c": result.get("mean_improvement_c"),
            "pct_improvement": result.get("pct_improvement"),
            "p_value": result.get("p_value"),
            "significant": bool(result.get("significant")) if result.get("significant") is not None else False,
            "effect_size_d": result.get("effect_size_d"),
            "effect_interpretation": result.get("effect_interpretation"),
            "skill_score": skill,
        })
    
    comparisons.sort(key=lambda x: x["p_value"] or 999)
    
    significant_count = sum(1 for c in comparisons if c["significant"])
    
    return {
        "horizon_days": horizon,
        "window_days": window_days,
        "total_comparisons": len(comparisons),
        "significant_improvements": significant_count,
        "comparisons": comparisons,
        "generated_at": date.today().isoformat(),
    }


@router.get("/summary")
def statistics_summary(
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon"),
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
):
    """
    Get summary statistics for all forecast sources.
    
    Returns MAE, RMSE, bias, and other metrics for each source.
    """
    init_database(_DB_PATH)
    
    sources = _get_available_sources()
    
    if not sources:
        return {"sources": [], "message": "No forecast data available"}
    
    summaries = []
    
    for source in sources:
        abs_errors = _get_errors_for_source(source, horizon, window_days)
        signed_errors = _get_signed_errors_for_source(source, horizon, window_days)
        
        if len(abs_errors) < 1:
            continue
        
        stats = _to_python_types(summary_statistics(abs_errors))
        
        bias = float(np.mean(signed_errors)) if signed_errors else None
        
        summaries.append({
            "source": source,
            "n_forecasts": stats["n"],
            "mae": stats["mae"],
            "rmse": stats["rmse"],
            "bias": round(bias, 4) if bias is not None else None,
            "std": stats["std"],
            "median_error": stats["median"],
            "min_error": stats["min"],
            "max_error": stats["max"],
        })
    
    summaries.sort(key=lambda x: x["rmse"])
    
    for i, s in enumerate(summaries, 1):
        s["rank"] = i
    
    return {
        "horizon_days": horizon,
        "window_days": window_days,
        "sources": summaries,
        "generated_at": date.today().isoformat(),
    }
