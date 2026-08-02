"""
Baseline Collector - Fetches daily forecasts from commercial weather APIs

This module collects Day-1, Day-2, and Day-3 temperature forecasts from:
- Open-Meteo (free, already integrated)
- OpenWeatherMap (free tier, 1000 calls/day)
- AccuWeather (free tier, 50 calls/day)
- ThermoSense (our model's predictions)

The forecasts are stored in a SQLite database for later comparison
against actual sensor readings, enabling a fair head-to-head evaluation.

Usage:
    python -m src.data.baseline_collector --collect
    python -m src.data.baseline_collector --status
"""

import json
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_DB_PATH = _PROJECT_ROOT / "data" / "baselines" / "forecasts.db"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def init_database(db_path: Optional[Path] = None) -> Path:
    """
    Initialize the forecasts database with required tables.
    
    Tables:
    - daily_forecasts: Predictions from each source
    - daily_actuals: Actual sensor readings (ground truth)
    - collection_log: Track when each source was collected
    """
    db_path = db_path or _DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_date DATE NOT NULL,
            generated_at TEXT NOT NULL,
            source TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            predicted_temp_c REAL NOT NULL,
            predicted_temp_min_c REAL,
            predicted_temp_max_c REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            raw_response TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(forecast_date, source, horizon_days)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_actuals (
            date DATE PRIMARY KEY,
            sensor_temp_c REAL NOT NULL,
            sensor_humidity_pct REAL,
            api_temp_c REAL,
            recorded_at TEXT NOT NULL,
            source TEXT DEFAULT 'sensor',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            success INTEGER NOT NULL,
            error_message TEXT,
            forecasts_stored INTEGER DEFAULT 0
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_date ON daily_forecasts(forecast_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_source ON daily_forecasts(source)")
    
    conn.commit()
    conn.close()
    
    return db_path


def store_forecast(
    db_path: Path,
    forecast_date: date,
    source: str,
    horizon_days: int,
    predicted_temp_c: float,
    predicted_temp_min_c: Optional[float] = None,
    predicted_temp_max_c: Optional[float] = None,
    confidence_lower: Optional[float] = None,
    confidence_upper: Optional[float] = None,
    raw_response: Optional[str] = None,
) -> bool:
    """Store a single forecast. Returns True if inserted, False if duplicate."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO daily_forecasts 
            (forecast_date, generated_at, source, horizon_days, predicted_temp_c,
             predicted_temp_min_c, predicted_temp_max_c, confidence_lower, 
             confidence_upper, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                forecast_date.isoformat(),
                datetime.utcnow().isoformat() + "Z",
                source,
                horizon_days,
                predicted_temp_c,
                predicted_temp_min_c,
                predicted_temp_max_c,
                confidence_lower,
                confidence_upper,
                raw_response,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def store_actual(
    db_path: Path,
    date_: date,
    sensor_temp_c: float,
    sensor_humidity_pct: Optional[float] = None,
    api_temp_c: Optional[float] = None,
    source: str = "sensor",
) -> bool:
    """Store the actual temperature for a date. Returns True if inserted."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_actuals 
            (date, sensor_temp_c, sensor_humidity_pct, api_temp_c, recorded_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                date_.isoformat(),
                sensor_temp_c,
                sensor_humidity_pct,
                api_temp_c,
                datetime.utcnow().isoformat() + "Z",
                source,
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def log_collection(
    db_path: Path,
    source: str,
    success: bool,
    error_message: Optional[str] = None,
    forecasts_stored: int = 0,
):
    """Log a collection attempt."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO collection_log (source, collected_at, success, error_message, forecasts_stored)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            source,
            datetime.utcnow().isoformat() + "Z",
            1 if success else 0,
            error_message,
            forecasts_stored,
        ),
    )
    conn.commit()
    conn.close()


def fetch_open_meteo_forecast(lat: float, lon: float, timezone: str) -> List[Dict[str, Any]]:
    """
    Fetch daily temperature forecasts from Open-Meteo (free, no API key).
    
    Returns list of forecasts for Day 1, 2, 3.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": timezone,
        "forecast_days": 4,
    }
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    forecasts = []
    daily = data.get("daily", {})
    times = daily.get("time", [])
    means = daily.get("temperature_2m_mean", [])
    mins = daily.get("temperature_2m_min", [])
    maxs = daily.get("temperature_2m_max", [])
    
    today = date.today()
    
    for i, (t, mean, tmin, tmax) in enumerate(zip(times, means, mins, maxs)):
        forecast_date = date.fromisoformat(t)
        horizon = (forecast_date - today).days
        
        if 1 <= horizon <= 3 and mean is not None:
            forecasts.append({
                "forecast_date": forecast_date,
                "horizon_days": horizon,
                "predicted_temp_c": mean,
                "predicted_temp_min_c": tmin,
                "predicted_temp_max_c": tmax,
                "source": "open_meteo",
                "raw": json.dumps({"date": t, "mean": mean, "min": tmin, "max": tmax}),
            })
    
    return forecasts


def fetch_owm_forecast(lat: float, lon: float, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch daily temperature forecasts from OpenWeatherMap.
    
    Uses the 5-day/3-hour forecast endpoint (free tier).
    We extract the ~9 PM reading for each day to match our target hour.
    """
    if not api_key:
        return []
    
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    forecasts = []
    today = date.today()
    
    daily_temps = {}
    
    for item in data.get("list", []):
        dt = datetime.fromtimestamp(item["dt"])
        item_date = dt.date()
        hour = dt.hour
        
        if 18 <= hour <= 22:
            temp = item["main"]["temp"]
            if item_date not in daily_temps:
                daily_temps[item_date] = {
                    "temp": temp,
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "hour": hour,
                }
            elif abs(hour - 21) < abs(daily_temps[item_date]["hour"] - 21):
                daily_temps[item_date] = {
                    "temp": temp,
                    "temp_min": item["main"]["temp_min"],
                    "temp_max": item["main"]["temp_max"],
                    "hour": hour,
                }
    
    for forecast_date, temps in daily_temps.items():
        horizon = (forecast_date - today).days
        
        if 1 <= horizon <= 3:
            forecasts.append({
                "forecast_date": forecast_date,
                "horizon_days": horizon,
                "predicted_temp_c": temps["temp"],
                "predicted_temp_min_c": temps["temp_min"],
                "predicted_temp_max_c": temps["temp_max"],
                "source": "openweathermap",
                "raw": json.dumps(temps),
            })
    
    return forecasts


def fetch_accuweather_forecast(lat: float, lon: float, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch daily temperature forecasts from AccuWeather.
    
    Free tier allows 50 calls/day. We use the 5-day daily forecast endpoint.
    
    Note: AccuWeather requires a location key, which we obtain via their
    geoposition search endpoint first.
    """
    if not api_key:
        return []
    
    location_url = "http://dataservice.accuweather.com/locations/v1/cities/geoposition/search"
    params = {
        "apikey": api_key,
        "q": f"{lat},{lon}",
    }
    
    resp = requests.get(location_url, params=params, timeout=30)
    resp.raise_for_status()
    location_data = resp.json()
    location_key = location_data.get("Key")
    
    if not location_key:
        raise ValueError("Could not get AccuWeather location key")
    
    forecast_url = f"http://dataservice.accuweather.com/forecasts/v1/daily/5day/{location_key}"
    params = {
        "apikey": api_key,
        "metric": "true",
    }
    
    resp = requests.get(forecast_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    forecasts = []
    today = date.today()
    
    for item in data.get("DailyForecasts", []):
        epoch = item["EpochDate"]
        forecast_date = date.fromtimestamp(epoch)
        horizon = (forecast_date - today).days
        
        if 1 <= horizon <= 3:
            temp_min = item["Temperature"]["Minimum"]["Value"]
            temp_max = item["Temperature"]["Maximum"]["Value"]
            temp_mean = (temp_min + temp_max) / 2
            
            forecasts.append({
                "forecast_date": forecast_date,
                "horizon_days": horizon,
                "predicted_temp_c": temp_mean,
                "predicted_temp_min_c": temp_min,
                "predicted_temp_max_c": temp_max,
                "source": "accuweather",
                "raw": json.dumps({
                    "date": forecast_date.isoformat(),
                    "min": temp_min,
                    "max": temp_max,
                }),
            })
    
    return forecasts


def fetch_thermosense_forecast(api_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """
    Fetch forecasts from the ThermoSense API (our model).
    """
    try:
        resp = requests.get(f"{api_url}/api/forecast", params={"days": 3}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []
    
    forecasts = []
    
    for item in data.get("forecasts", []):
        forecast_date = date.fromisoformat(item["date"])
        horizon = item.get("horizon_days", 1)
        
        forecasts.append({
            "forecast_date": forecast_date,
            "horizon_days": horizon,
            "predicted_temp_c": item["predicted_temp_c"],
            "confidence_lower": item.get("lower_bound_c"),
            "confidence_upper": item.get("upper_bound_c"),
            "source": "thermosense",
            "raw": json.dumps(item),
        })
    
    return forecasts


def collect_all_baselines(
    db_path: Optional[Path] = None,
    thermosense_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """
    Collect forecasts from all sources and store in the database.
    
    Returns a summary of the collection run.
    """
    db_path = db_path or _DB_PATH
    init_database(db_path)
    
    config = _load_config()
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    timezone = config["location"]["timezone"]
    
    owm_key = os.environ.get("OWM_API_KEY")
    accuweather_key = os.environ.get("ACCUWEATHER_API_KEY")
    
    results = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "sources": {},
    }
    
    sources = [
        ("open_meteo", lambda: fetch_open_meteo_forecast(lat, lon, timezone)),
        ("openweathermap", lambda: fetch_owm_forecast(lat, lon, owm_key) if owm_key else []),
        ("accuweather", lambda: fetch_accuweather_forecast(lat, lon, accuweather_key) if accuweather_key else []),
        ("thermosense", lambda: fetch_thermosense_forecast(thermosense_url)),
    ]
    
    for source_name, fetch_fn in sources:
        try:
            forecasts = fetch_fn()
            stored = 0
            
            for f in forecasts:
                if store_forecast(
                    db_path,
                    f["forecast_date"],
                    f["source"],
                    f["horizon_days"],
                    f["predicted_temp_c"],
                    f.get("predicted_temp_min_c"),
                    f.get("predicted_temp_max_c"),
                    f.get("confidence_lower"),
                    f.get("confidence_upper"),
                    f.get("raw"),
                ):
                    stored += 1
            
            results["sources"][source_name] = {
                "success": True,
                "fetched": len(forecasts),
                "stored": stored,
            }
            log_collection(db_path, source_name, True, forecasts_stored=stored)
            print(f"[{source_name}] Fetched {len(forecasts)}, stored {stored}")
        
        except Exception as e:
            results["sources"][source_name] = {
                "success": False,
                "error": str(e),
            }
            log_collection(db_path, source_name, False, error_message=str(e))
            print(f"[{source_name}] Error: {e}")
        
        time.sleep(1)
    
    return results


def get_leaderboard(
    db_path: Optional[Path] = None,
    window_days: int = 30,
    horizon: int = 1,
) -> List[Dict[str, Any]]:
    """
    Compute the accuracy leaderboard comparing all sources.
    
    Returns list of sources ranked by RMSE, with MAE and sample count.
    """
    db_path = db_path or _DB_PATH
    
    if not db_path.exists():
        return []
    
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute(
        """
        SELECT 
            f.source,
            COUNT(*) as n_days,
            AVG(ABS(f.predicted_temp_c - a.sensor_temp_c)) as mae,
            AVG((f.predicted_temp_c - a.sensor_temp_c) * (f.predicted_temp_c - a.sensor_temp_c)) as mse,
            AVG(f.predicted_temp_c - a.sensor_temp_c) as mean_error
        FROM daily_forecasts f
        JOIN daily_actuals a ON f.forecast_date = a.date
        WHERE f.horizon_days = ?
          AND f.forecast_date >= ?
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
            "rank": len(results) + 1,
            "source": row["source"],
            "rmse": round(mse ** 0.5, 3),
            "mae": round(row["mae"], 3) if row["mae"] else None,
            "mean_error": round(row["mean_error"], 3) if row["mean_error"] else None,
            "n_days": row["n_days"],
        })
    
    conn.close()
    return results


def get_forecast_comparison(
    db_path: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """
    Get detailed forecast vs actual comparison for a date range.
    """
    db_path = db_path or _DB_PATH
    
    if not db_path.exists():
        return []
    
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    if end_date is None:
        end_date = date.today()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute(
        """
        SELECT 
            a.date,
            a.sensor_temp_c as actual,
            f.source,
            f.horizon_days,
            f.predicted_temp_c as predicted,
            f.predicted_temp_c - a.sensor_temp_c as error
        FROM daily_actuals a
        LEFT JOIN daily_forecasts f ON f.forecast_date = a.date
        WHERE a.date >= ? AND a.date <= ?
        ORDER BY a.date DESC, f.source, f.horizon_days
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_collection_status(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get status of the baseline collection system."""
    db_path = db_path or _DB_PATH
    
    if not db_path.exists():
        return {"initialized": False}
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("SELECT COUNT(*) as count FROM daily_forecasts")
    forecast_count = cursor.fetchone()["count"]
    
    cursor = conn.execute("SELECT COUNT(*) as count FROM daily_actuals")
    actual_count = cursor.fetchone()["count"]
    
    cursor = conn.execute("""
        SELECT source, MAX(collected_at) as last_collected, 
               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
               SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
        FROM collection_log
        GROUP BY source
    """)
    sources = {row["source"]: dict(row) for row in cursor.fetchall()}
    
    cursor = conn.execute("""
        SELECT MIN(forecast_date) as earliest, MAX(forecast_date) as latest
        FROM daily_forecasts
    """)
    date_range = cursor.fetchone()
    
    conn.close()
    
    return {
        "initialized": True,
        "database_path": str(db_path),
        "forecast_count": forecast_count,
        "actual_count": actual_count,
        "date_range": {
            "earliest": date_range["earliest"],
            "latest": date_range["latest"],
        } if date_range["earliest"] else None,
        "sources": sources,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ThermoSense Baseline Collector")
    parser.add_argument("--collect", action="store_true", help="Collect forecasts from all sources")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--leaderboard", action="store_true", help="Show accuracy leaderboard")
    parser.add_argument("--window", type=int, default=30, help="Leaderboard window in days")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (1, 2, or 3)")
    parser.add_argument("--init", action="store_true", help="Initialize database only")
    
    args = parser.parse_args()
    
    if args.init:
        path = init_database()
        print(f"Database initialized: {path}")
    
    elif args.collect:
        results = collect_all_baselines()
        print(json.dumps(results, indent=2))
    
    elif args.status:
        status = get_collection_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.leaderboard:
        board = get_leaderboard(window_days=args.window, horizon=args.horizon)
        if board:
            print(f"\n{'='*60}")
            print(f"  LEADERBOARD (Day-{args.horizon}, Last {args.window} Days)")
            print(f"{'='*60}")
            print(f"{'Rank':<6}{'Source':<18}{'RMSE':<10}{'MAE':<10}{'N':<6}")
            print(f"{'-'*60}")
            for row in board:
                rank = "🥇" if row["rank"] == 1 else "🥈" if row["rank"] == 2 else "🥉" if row["rank"] == 3 else str(row["rank"])
                print(f"{rank:<6}{row['source']:<18}{row['rmse']:<10}{row['mae']:<10}{row['n_days']:<6}")
            print(f"{'='*60}\n")
        else:
            print("No leaderboard data available yet. Need forecasts + actuals.")
    
    else:
        parser.print_help()
