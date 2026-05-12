"""
Locations API routes — Multi-location sensor network management

Supports Phase F expansion to multiple sensor locations with
location-specific bias corrections and aggregated analytics.

Endpoints:
    GET  /api/locations           - List all configured locations
    GET  /api/locations/{id}      - Get details for a specific location
    GET  /api/locations/summary   - Aggregated accuracy across all locations
"""

import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_DB_BASE_PATH = _PROJECT_ROOT / "data" / "baselines"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_location_db_path(location_id: str) -> Path:
    """Get database path for a specific location."""
    if location_id == "default" or location_id == "bangalore_central":
        return _DB_BASE_PATH / "forecasts.db"
    return _DB_BASE_PATH / f"forecasts_{location_id}.db"


def _get_location_stats(
    db_path: Path,
    window_days: int = 30,
    horizon: int = 1,
) -> Optional[Dict[str, Any]]:
    """Get accuracy statistics for a location."""
    if not db_path.exists():
        return None
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute(
        """
        SELECT 
            f.source,
            COUNT(*) as n_days,
            AVG(ABS(f.predicted_temp_c - a.sensor_temp_c)) as mae,
            AVG((f.predicted_temp_c - a.sensor_temp_c) * (f.predicted_temp_c - a.sensor_temp_c)) as mse
        FROM daily_forecasts f
        JOIN daily_actuals a ON f.forecast_date = a.date
        WHERE f.horizon_days = ? AND f.forecast_date >= ?
        GROUP BY f.source
        HAVING n_days >= 3
        ORDER BY mse ASC
        """,
        (horizon, cutoff),
    )
    
    results = []
    for row in cursor.fetchall():
        mse = row["mse"] or 0
        results.append({
            "source": row["source"],
            "rmse": round(mse ** 0.5, 3),
            "mae": round(row["mae"], 3) if row["mae"] else None,
            "n_days": row["n_days"],
        })
    
    cursor = conn.execute(
        "SELECT COUNT(*) as count FROM daily_actuals WHERE date >= ?",
        (cutoff,)
    )
    sensor_days = cursor.fetchone()["count"]
    
    conn.close()
    
    if not results:
        return None
    
    ts = next((r for r in results if r["source"] == "thermosense"), None)
    best_other = next((r for r in results if r["source"] != "thermosense"), None)
    
    improvement = None
    if ts and best_other and ts["rmse"] > 0 and best_other["rmse"] > 0:
        improvement = round((1 - ts["rmse"] / best_other["rmse"]) * 100, 1)
    
    return {
        "rankings": results,
        "thermosense_rank": next((i+1 for i, r in enumerate(results) if r["source"] == "thermosense"), None),
        "improvement_vs_best_pct": improvement,
        "sensor_days": sensor_days,
        "completeness_pct": round(sensor_days / window_days * 100, 1),
    }


@router.get("")
def list_locations():
    """
    List all configured sensor locations.
    
    Returns location details including coordinates, type, and status.
    """
    config = _load_config()
    locations = config.get("locations", [])
    
    if not locations:
        main_loc = config.get("location", {})
        locations = [{
            "id": "default",
            "name": main_loc.get("name", "Default"),
            "lat": main_loc.get("lat"),
            "lon": main_loc.get("lon"),
            "timezone": main_loc.get("timezone"),
            "type": "primary",
            "active": True,
        }]
    
    response_locations = []
    for loc in locations:
        db_path = _get_location_db_path(loc["id"])
        has_data = db_path.exists()
        
        response_locations.append({
            "id": loc["id"],
            "name": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "timezone": loc.get("timezone"),
            "type": loc.get("type", "unknown"),
            "description": loc.get("description"),
            "active": loc.get("active", False),
            "has_data": has_data,
        })
    
    return {
        "locations": response_locations,
        "total": len(response_locations),
        "active": sum(1 for loc in response_locations if loc["active"]),
    }


@router.get("/summary")
def locations_summary(
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon"),
):
    """
    Get aggregated accuracy summary across all active locations.
    
    Returns average improvement and per-location performance breakdown.
    """
    config = _load_config()
    locations = config.get("locations", [])
    
    if not locations:
        main_loc = config.get("location", {})
        locations = [{
            "id": "default",
            "name": main_loc.get("name", "Default"),
            "active": True,
        }]
    
    location_results = []
    total_improvement = 0
    improvement_count = 0
    
    for loc in locations:
        if not loc.get("active", False):
            continue
        
        db_path = _get_location_db_path(loc["id"])
        stats = _get_location_stats(db_path, window_days, horizon)
        
        if stats:
            result = {
                "id": loc["id"],
                "name": loc["name"],
                "thermosense_rank": stats["thermosense_rank"],
                "improvement_vs_best_pct": stats["improvement_vs_best_pct"],
                "sensor_days": stats["sensor_days"],
                "completeness_pct": stats["completeness_pct"],
            }
            
            ts_ranking = next(
                (r for r in stats["rankings"] if r["source"] == "thermosense"),
                None
            )
            if ts_ranking:
                result["thermosense_rmse"] = ts_ranking["rmse"]
            
            location_results.append(result)
            
            if stats["improvement_vs_best_pct"] is not None:
                total_improvement += stats["improvement_vs_best_pct"]
                improvement_count += 1
    
    avg_improvement = None
    if improvement_count > 0:
        avg_improvement = round(total_improvement / improvement_count, 1)
    
    return {
        "window_days": window_days,
        "horizon_days": horizon,
        "locations": location_results,
        "total_active_locations": len(location_results),
        "average_improvement_pct": avg_improvement,
        "generated_at": date.today().isoformat(),
        "resume_statement": _generate_multi_location_statement(
            location_results, avg_improvement
        ),
    }


def _generate_multi_location_statement(
    locations: List[Dict[str, Any]],
    avg_improvement: Optional[float],
) -> Optional[str]:
    """Generate a resume-ready statement for multi-location deployment."""
    if not locations or avg_improvement is None:
        return None
    
    n_locations = len(locations)
    
    if n_locations < 2 or avg_improvement <= 0:
        return None
    
    return (
        f"Deployed ThermoSense to {n_locations} locations; "
        f"average {avg_improvement:.0f}% RMSE improvement over commercial apps."
    )


@router.get("/{location_id}")
def get_location(
    location_id: str,
    window_days: int = Query(30, ge=7, le=365, description="Number of days to include"),
    horizon: int = Query(1, ge=1, le=3, description="Forecast horizon"),
):
    """
    Get detailed information for a specific location.
    
    Includes accuracy statistics and leaderboard for that location.
    """
    config = _load_config()
    locations = config.get("locations", [])
    
    if not locations:
        main_loc = config.get("location", {})
        if location_id == "default":
            locations = [{
                "id": "default",
                "name": main_loc.get("name", "Default"),
                "lat": main_loc.get("lat"),
                "lon": main_loc.get("lon"),
                "timezone": main_loc.get("timezone"),
                "active": True,
            }]
    
    location = next((loc for loc in locations if loc["id"] == location_id), None)
    
    if not location:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")
    
    db_path = _get_location_db_path(location_id)
    stats = _get_location_stats(db_path, window_days, horizon)
    
    return {
        "location": {
            "id": location["id"],
            "name": location["name"],
            "lat": location.get("lat"),
            "lon": location.get("lon"),
            "timezone": location.get("timezone"),
            "type": location.get("type"),
            "description": location.get("description"),
            "active": location.get("active", False),
        },
        "statistics": stats,
        "window_days": window_days,
        "horizon_days": horizon,
        "has_data": stats is not None,
    }
