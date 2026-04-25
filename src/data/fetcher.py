"""
Data fetcher for ThermoSense — Phase 1.

Pulls historical and forecast weather data from Open-Meteo (completely free,
no API key required) and optionally from OpenWeatherMap (free tier) for a
commercial-app baseline comparison.

Design principles:
- All API keys are loaded from environment variables only — never hardcoded.
- Raw JSON responses are saved to data/raw/ before any parsing, making the
  pipeline fully reproducible and debuggable.
- Exponential-backoff retry wraps every network call so transient errors
  (rate limits, timeouts) don't abort a long backfill run.
- Config-driven: location, timezone, and variable lists are read from
  config/config.yaml so nothing is hardcoded in this module.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OWM_CURRENT = "https://api.openweathermap.org/data/2.5/weather"

HOURLY_VARS: List[str] = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "precipitation",
    "pressure_msl",
    "cloudcover",
    "windspeed_10m",
    "shortwave_radiation",   # UV proxy — uv_index is not stored in the archive
]

COL_RENAME: Dict[str, str] = {
    "temperature_2m": "temp_c",
    "relativehumidity_2m": "humidity_pct",
    "dewpoint_2m": "dewpoint_c",
    "precipitation": "precip_mm",
    "pressure_msl": "pressure_hpa",
    "cloudcover": "cloudcover_pct",
    "windspeed_10m": "windspeed_kmh",
    "shortwave_radiation": "solar_radiation_wm2",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _get_with_retry(
    url: str,
    params: dict,
    max_retries: int = 4,
    timeout: int = 30,
) -> dict:
    """
    GET request with exponential-backoff retry.

    Retries on connection errors, timeouts, and 429/5xx HTTP responses.
    Raises the last exception if all retries are exhausted.
    """
    delay = 2
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay))
                print(f"[fetcher] Rate-limited. Waiting {wait}s (attempt {attempt})…")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries:
                print(f"[fetcher] Attempt {attempt} failed: {exc}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"[fetcher] All {max_retries} retries failed.") from last_exc


# ── Raw save ──────────────────────────────────────────────────────────────────

def _save_raw(payload: dict, filename: str) -> Path:
    """Persist raw API JSON to data/raw/ for reproducibility and debugging."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = _RAW_DIR / filename
    out.write_text(json.dumps(payload, indent=2))
    return out


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_hourly_to_df(hourly: dict, vars: List[str]) -> pd.DataFrame:
    """Convert Open-Meteo hourly dict to a DataFrame with renamed columns."""
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in vars:
        col = COL_RENAME.get(var, var)
        df[col] = hourly.get(var)
    return df.set_index("time")


def _resample_to_9pm(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter hourly data to the 21:00 (9 PM) row for each date.

    This matches the original dataset's recording convention (readings taken
    between 9–10 PM each evening). Using a single fixed snapshot per day
    makes the merged dataframe directly comparable with the legacy CSV.
    """
    mask = hourly_df.index.hour == 21
    daily = hourly_df[mask].copy()
    daily.index = daily.index.normalize()
    daily.index.name = "date"
    return daily.reset_index()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_historical_open_meteo(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timezone: Optional[str] = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather data from the Open-Meteo archive API.

    No API key is required. Data is available from 1940 to the present day.
    The function resamples hourly data to daily 9 PM snapshots to align
    with the existing sensor dataset.

    Args:
        lat: Latitude. Defaults to config location.
        lon: Longitude. Defaults to config location.
        start_date: Start date (YYYY-MM-DD). Defaults to 365 days ago.
        end_date: End date (YYYY-MM-DD). Defaults to yesterday.
        timezone: IANA timezone string. Defaults to config timezone.
        save_raw: Whether to persist the raw JSON response to data/raw/.

    Returns:
        DataFrame with columns:
            date, temp_c, humidity_pct, dewpoint_c, precip_mm,
            pressure_hpa, cloudcover_pct, windspeed_kmh, uv_index
    """
    cfg = _load_config()
    lat = lat if lat is not None else cfg["location"]["lat"]
    lon = lon if lon is not None else cfg["location"]["lon"]
    timezone = timezone or cfg["location"]["timezone"]

    if start_date is None:
        start_date = (date.today() - timedelta(days=cfg["data"]["history_days"])).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
    }

    print(f"[fetcher] Fetching historical data: {start_date} → {end_date} "
          f"({lat:.4f}, {lon:.4f})")

    payload = _get_with_retry(OPEN_METEO_ARCHIVE, params)

    if save_raw:
        fname = f"open_meteo_historical_{start_date}_{end_date}.json"
        path = _save_raw(payload, fname)
        print(f"[fetcher] Raw response saved → {path.relative_to(_PROJECT_ROOT)}")

    df = _parse_hourly_to_df(payload["hourly"], HOURLY_VARS)
    daily = _resample_to_9pm(df)

    print(f"[fetcher] Parsed {len(daily)} daily rows "
          f"({daily['date'].min().date()} → {daily['date'].max().date()})")
    return daily


def fetch_forecast_open_meteo(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    forecast_days: int = 7,
    timezone: Optional[str] = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    Fetch the upcoming weather forecast from Open-Meteo (up to 16 days, free).

    Used at inference time to obtain future covariate values (humidity, pressure,
    cloud cover, etc.) needed by LightGBM and TFT for forward predictions.

    Args:
        lat: Latitude. Defaults to config location.
        lon: Longitude. Defaults to config location.
        forecast_days: Number of forecast days (1–16).
        timezone: IANA timezone string. Defaults to config timezone.
        save_raw: Whether to persist the raw JSON response.

    Returns:
        DataFrame with same schema as fetch_historical_open_meteo().
    """
    cfg = _load_config()
    lat = lat if lat is not None else cfg["location"]["lat"]
    lon = lon if lon is not None else cfg["location"]["lon"]
    timezone = timezone or cfg["location"]["timezone"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "forecast_days": forecast_days,
    }

    print(f"[fetcher] Fetching {forecast_days}-day forecast ({lat:.4f}, {lon:.4f})")
    payload = _get_with_retry(OPEN_METEO_FORECAST, params)

    if save_raw:
        today_str = date.today().isoformat()
        path = _save_raw(payload, f"open_meteo_forecast_{today_str}.json")
        print(f"[fetcher] Raw response saved → {path.relative_to(_PROJECT_ROOT)}")

    df = _parse_hourly_to_df(payload["hourly"], HOURLY_VARS)
    daily = _resample_to_9pm(df)

    print(f"[fetcher] Parsed {len(daily)} forecast rows")
    return daily


def fetch_owm_current(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    save_raw: bool = True,
) -> Optional[dict]:
    """
    Fetch current conditions from OpenWeatherMap (free tier).

    Used only as a commercial-app baseline for comparison — to measure how much
    our TFT ensemble outperforms raw public weather data at our specific location.

    Requires OWM_API_KEY environment variable (free key from openweathermap.org).
    Returns None gracefully if the key is not set — Open-Meteo is sufficient
    for all forecasting functionality.

    Args:
        lat: Latitude. Defaults to config location.
        lon: Longitude. Defaults to config location.
        save_raw: Whether to persist the raw JSON response.

    Returns:
        Dict with current weather fields, or None if no API key configured.
    """
    api_key = os.environ.get("OWM_API_KEY")
    if not api_key:
        print("[fetcher] OWM_API_KEY not set. Skipping OpenWeatherMap fetch.")
        return None

    cfg = _load_config()
    lat = lat if lat is not None else cfg["location"]["lat"]
    lon = lon if lon is not None else cfg["location"]["lon"]

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    print(f"[fetcher] Fetching OWM current conditions ({lat:.4f}, {lon:.4f})")

    payload = _get_with_retry(OWM_CURRENT, params, timeout=10)

    if save_raw:
        today_str = datetime.utcnow().strftime("%Y-%m-%dT%H%M")
        path = _save_raw(payload, f"owm_current_{today_str}.json")
        print(f"[fetcher] Raw OWM response saved → {path.relative_to(_PROJECT_ROOT)}")

    return {
        "temp_c": float(payload["main"]["temp"]),
        "humidity_pct": float(payload["main"]["humidity"]),
        "pressure_hpa": float(payload["main"]["pressure"]),
        "windspeed_kmh": float(payload["wind"]["speed"]) * 3.6,
        "cloudcover_pct": float(payload["clouds"]["all"]),
        "description": payload["weather"][0]["description"],
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


def backfill(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper to fetch a full historical backfill in one call.

    On first run, pulls `history_days` (default 365) of data from Open-Meteo
    archive to give the deep-learning model sufficient training data.

    Args:
        start_date: Override start date (YYYY-MM-DD).
        end_date: Override end date (YYYY-MM-DD).

    Returns:
        Daily DataFrame ready to be passed into preprocess.run_pipeline().
    """
    cfg = _load_config()
    history_days = cfg["data"]["history_days"]
    if start_date is None:
        start_date = (date.today() - timedelta(days=history_days)).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    print(f"[fetcher] Starting backfill: {history_days} days "
          f"({start_date} → {end_date})")
    return fetch_historical_open_meteo(start_date=start_date, end_date=end_date)
