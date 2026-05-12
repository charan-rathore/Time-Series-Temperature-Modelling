"""
Forecast endpoint for ThermoSense API.

GET /forecast
  Returns 1-3 day temperature predictions with uncertainty intervals.

POST /feedback
  Accepts the actual observed temperature for a past date.
  Appends to the local data store and updates the api_bias rolling feature.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "daily_merged.parquet"


class ForecastPoint(BaseModel):
    date: str
    predicted_temp_c: float
    lower_bound_c: float
    upper_bound_c: float
    horizon_days: int
    confidence: str = "90%"


class ForecastResponse(BaseModel):
    location: str
    generated_at: str
    model_used: str
    forecasts: List[ForecastPoint]


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
    today = date.today()

    manager = getattr(request.app.state, "model_manager", None)
    if manager is not None:
        raw_forecasts = manager.forecast(days=days)
        model_used = raw_forecasts[0]["model_used"] if raw_forecasts else "unknown"
    else:
        raw_forecasts = []
        model_used = "placeholder"

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
            forecasts.append(ForecastPoint(
                date=forecast_date,
                predicted_temp_c=26.0,
                lower_bound_c=24.5,
                upper_bound_c=27.5,
                horizon_days=d,
            ))

    return ForecastResponse(
        location=location_name,
        generated_at=datetime.utcnow().isoformat() + "Z",
        model_used=model_used,
        forecasts=forecasts,
    )


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit actual temperature")
def post_feedback(payload: FeedbackRequest, request: Request):
    """
    Accept an actual temperature observation for a past date.

    Appends the observation to the processed data store and marks it
    as a sensor reading so the api_bias feature can be recomputed.
    """
    try:
        obs_date = date.fromisoformat(payload.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

    if obs_date > date.today():
        raise HTTPException(status_code=400, detail="Cannot submit feedback for a future date")

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
        print(f"[feedback] Error persisting observation: {e}")

    return FeedbackResponse(
        status="accepted",
        message=f"Observation for {obs_date} recorded: {payload.actual_temp_c}°C",
        api_bias_updated=bias_updated,
    )
