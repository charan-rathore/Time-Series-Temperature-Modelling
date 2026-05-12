"""
Leaderboard API routes — Live accuracy comparison against commercial weather services

Endpoints:
    GET  /api/leaderboard           - Get accuracy rankings
    GET  /api/leaderboard/comparison - Detailed forecast vs actual data
    GET  /api/leaderboard/status    - Collection system status
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from src.data.baseline_collector import (
    get_leaderboard,
    get_forecast_comparison,
    get_collection_status,
)

router = APIRouter()


@router.get("")
def leaderboard(
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon (1=tomorrow, 2=day after, 3=3 days out)"),
):
    """
    Get the live accuracy leaderboard comparing ThermoSense vs commercial weather services.
    
    Returns sources ranked by RMSE (lower is better), along with MAE and sample count.
    """
    board = get_leaderboard(window_days=window_days, horizon=horizon)
    
    thermosense_rank = next((r["rank"] for r in board if r["source"] == "thermosense"), None)
    best_commercial = next((r for r in board if r["source"] != "thermosense"), None)
    
    improvement = None
    if thermosense_rank and best_commercial:
        ts = next(r for r in board if r["source"] == "thermosense")
        if ts["rmse"] > 0 and best_commercial["rmse"] > 0:
            improvement = round((1 - ts["rmse"] / best_commercial["rmse"]) * 100, 1)
    
    return {
        "generated_at": date.today().isoformat(),
        "window_days": window_days,
        "horizon_days": horizon,
        "rankings": board,
        "thermosense_rank": thermosense_rank,
        "improvement_vs_best_commercial_pct": improvement,
        "best_commercial": best_commercial["source"] if best_commercial else None,
    }


@router.get("/comparison")
def comparison(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get detailed forecast vs actual comparison data.
    
    Returns each date's actual temperature alongside predictions from all sources.
    """
    start = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=30)
    end = date.fromisoformat(end_date) if end_date else date.today()
    
    data = get_forecast_comparison(start_date=start, end_date=end)
    
    by_date = {}
    for row in data:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {
                "date": d,
                "actual_temp_c": row["actual"],
                "forecasts": {},
            }
        if row["source"]:
            horizon_key = f"day{row['horizon_days']}"
            if row["source"] not in by_date[d]["forecasts"]:
                by_date[d]["forecasts"][row["source"]] = {}
            by_date[d]["forecasts"][row["source"]][horizon_key] = {
                "predicted": row["predicted"],
                "error": round(row["error"], 2) if row["error"] else None,
            }
    
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "data": list(by_date.values()),
    }


@router.get("/status")
def status():
    """
    Get the status of the baseline collection system.
    
    Shows database info, forecast counts, and collection history per source.
    """
    return get_collection_status()
