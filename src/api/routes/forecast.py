"""
Forecast endpoint for ThermoSense API.

GET /forecast
  Returns 1–3 day temperature predictions with uncertainty intervals.
  Predictions come from the ensemble model (champion), with fallback to LightGBM.

POST /feedback
  Accepts the actual observed temperature for a past date.
  Appends to the local database and updates the api_bias rolling feature.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


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
    Return temperature forecasts for the next 1–3 days.

    Uses the ensemble model (SARIMA + LightGBM + TFT) when available,
    falling back to LightGBM for faster cold-start behaviour.

    Query params:
    - days: Number of forecast days (1–3, default 3)
    - lat, lon: Override default location from config (optional)
    """
    if days < 1 or days > 3:
        raise HTTPException(status_code=400, detail="days must be between 1 and 3")

    config = request.app.state.config
    location_name = config["location"]["name"]
    today = date.today()

    # Placeholder — replace with actual model inference once models are trained.
    # See PLAN.md Phase 6 for full implementation.
    forecasts = []
    for d in range(1, days + 1):
        forecast_date = (today + timedelta(days=d)).isoformat()
        forecasts.append(
            ForecastPoint(
                date=forecast_date,
                predicted_temp_c=26.0,   # replace with model output
                lower_bound_c=24.5,
                upper_bound_c=27.5,
                horizon_days=d,
            )
        )

    return ForecastResponse(
        location=location_name,
        generated_at=datetime.utcnow().isoformat() + "Z",
        model_used="ensemble (placeholder)",
        forecasts=forecasts,
    )


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit actual temperature")
def post_feedback(payload: FeedbackRequest, request: Request):
    """
    Accept an actual temperature observation for a past date.

    This closes the prediction loop:
    1. Appends the observation to the processed data store.
    2. Recomputes the api_bias rolling feature for that date.
    3. Queues an incremental model update (online learning).

    The /metrics endpoint will reflect the new observation on next call.
    """
    try:
        obs_date = date.fromisoformat(payload.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

    if obs_date > date.today():
        raise HTTPException(status_code=400, detail="Cannot submit feedback for a future date")

    # Placeholder — replace with actual DB write + bias update logic.
    # See PLAN.md Phase 6 for full implementation.
    return FeedbackResponse(
        status="accepted",
        message=f"Observation for {obs_date} recorded: {payload.actual_temp_c}°C",
        api_bias_updated=True,
    )
