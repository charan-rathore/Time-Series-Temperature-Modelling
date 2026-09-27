"""
Forecast endpoint for ThermoSense API.

GET /forecast
  Returns 1-3 day temperature predictions with uncertainty intervals.

POST /feedback
  Accepts the actual observed temperature for a past date.
  Appends to the local data store and updates the api_bias rolling feature.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Optional

import os
import hmac
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "daily_merged.parquet"


class ForecastPoint(BaseModel):
    date: str
    predicted_temp_c: float
    lower_bound_c: Optional[float] = None
    upper_bound_c: Optional[float] = None
    horizon_days: int
    confidence: str = "unverified"


class ForecastResponse(BaseModel):
    location: str
    generated_at: str
    model_used: str
    forecasts: List[ForecastPoint]
    availability: str = "trained_model"


class FeedbackRequest(BaseModel):
    date: str = Field(..., description="Date of observation in YYYY-MM-DD format")
    actual_temp_c: float = Field(..., description="Actual observed temperature in Celsius")


class FeedbackResponse(BaseModel):
    status: str
    message: str
    api_bias_updated: bool


@router.get("", response_model=ForecastResponse, summary="Get temperature forecast")
def get_forecast(
    request: Request,
    days: int = 3,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
):
    """
    Return temperature forecasts for the next 1-3 days.

    Uses trained models (ensemble > LightGBM > SARIMA) when available,
    falling back to climatology if no models are trained.
    """
    if days < 1 or days > 3:
        raise HTTPException(status_code=400, detail="days must be between 1 and 3")

    config = request.app.state.config
    location_name = config["location"]["name"]
    today = datetime.now(ZoneInfo(config["location"]["timezone"])).date()

    manager = getattr(request.app.state, "model_manager", None)
    raw_forecasts = manager.forecast(days=days) if manager is not None else []
    model_used = raw_forecasts[0]["model_used"] if raw_forecasts else "none"

    # A fixed fallback is not a weather forecast. Return an explicit unavailable
    # state instead of making the dashboard display an invented 26 C prediction.
    if not raw_forecasts or all(fc.get("model_used") in {"placeholder", "climatology", "fallback"} for fc in raw_forecasts):
        # The external weather service provides a real regional forecast, but
        # that is not a trained hyperlocal model. State provenance explicitly.
        try:
            from src.data.fetcher import fetch_forecast_open_meteo
            regional = fetch_forecast_open_meteo(forecast_days=4, save_raw=False)
            points = []
            for d in range(1, days + 1):
                target = pd.Timestamp(today + timedelta(days=d))
                match = regional.loc[pd.to_datetime(regional["date"]) == target]
                if match.empty or pd.isna(match.iloc[0]["temp_c"]):
                    break
                temp = float(match.iloc[0]["temp_c"])
                points.append(ForecastPoint(date=target.date().isoformat(), predicted_temp_c=round(temp, 2),
                                            horizon_days=d, confidence="none"))
            return ForecastResponse(location=location_name, generated_at=datetime.utcnow().isoformat() + "Z",
                                    model_used="open_meteo_regional", forecasts=points,
                                    availability="regional_forecast_only" if points else "forecast_unavailable")
        except Exception:
            return ForecastResponse(location=location_name, generated_at=datetime.utcnow().isoformat() + "Z",
                                    model_used="none", forecasts=[], availability="forecast_unavailable")

    forecasts = []
    for d in range(1, days + 1):
        forecast_date = (today + timedelta(days=d)).isoformat()
        if d - 1 < len(raw_forecasts):
            fc = raw_forecasts[d - 1]
            forecasts.append(ForecastPoint(
                date=forecast_date,
                predicted_temp_c=fc["predicted_temp_c"],
                lower_bound_c=fc["lower_bound_c"],
                upper_bound_c=fc["upper_bound_c"],
                horizon_days=d,
            ))
        else:
            break

    return ForecastResponse(
        location=location_name,
        generated_at=datetime.utcnow().isoformat() + "Z",
        model_used=model_used,
        forecasts=forecasts,
    )


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit actual temperature")
def post_feedback(payload: FeedbackRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    """
    Accept an actual temperature observation for a past date.

    Appends the observation to the processed data store and marks it
    as a sensor reading so the api_bias feature can be recomputed.
    """
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Feedback storage is unavailable on this deployment")
    expected = os.environ.get("THERMOSENSE_API_KEY")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Feedback requires a configured API key")
    try:
        obs_date = date.fromisoformat(payload.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

    if obs_date > date.today():
        raise HTTPException(status_code=400, detail="Cannot submit feedback for a future date")

    if not PROCESSED_PATH.exists():
        raise HTTPException(503, "No dataset is loaded; observation was not saved")
    bias_updated = False
    try:
        if PROCESSED_PATH.exists():
            df = pd.read_parquet(PROCESSED_PATH)
            obs_dt = pd.Timestamp(obs_date)

            if obs_dt in df["date"].values:
                idx = df.index[df["date"] == obs_dt][0]
                old_temp = df.at[idx, "temp_c"]
                df.at[idx, "temp_c"] = payload.actual_temp_c
                df.at[idx, "is_sensor_reading"] = True
                if "temp_c_api" in df.columns and pd.notna(df.at[idx, "temp_c_api"]):
                    df.at[idx, "api_bias"] = payload.actual_temp_c - df.at[idx, "temp_c_api"]
                    bias_updated = True
            else:
                new_row = pd.DataFrame([{
                    "date": obs_dt,
                    "temp_c": payload.actual_temp_c,
                    "is_sensor_reading": True,
                    "gap_filled": False,
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df = df.sort_values("date").reset_index(drop=True)

            df.to_parquet(PROCESSED_PATH, index=False)
    except Exception as e:
        raise HTTPException(503, "Observation could not be saved") from e

    return FeedbackResponse(
        status="accepted",
        message=f"Observation for {obs_date} recorded: {payload.actual_temp_c}°C",
        api_bias_updated=bias_updated,
    )
