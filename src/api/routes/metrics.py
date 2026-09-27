"""Metrics endpoint for ThermoSense API."""

from typing import Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HorizonMetrics(BaseModel):
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    skill_score: float = 0.0
    coverage_90pct: Optional[float] = None


class MetricsResponse(BaseModel):
    location: str
    evaluation_window_days: int
    n_observations: int
    source: str = "unknown"
    models: Dict[str, Dict[str, HorizonMetrics]]


@router.get("", response_model=MetricsResponse, summary="Live model accuracy metrics")
def get_metrics(request: Request, window_days: int = 30):
    """Return only prospectively validated model metrics (none yet)."""
    config = request.app.state.config
    location_name = config["location"]["name"]

    # No stored training result has verified forecast issuance, matched targets,
    # and an untouched evaluation window. Withhold all historical scores.
    model_metrics: Dict[str, Dict[str, HorizonMetrics]] = {}
    n_obs = 0
    source = "unavailable"

    return MetricsResponse(
        location=location_name,
        evaluation_window_days=window_days,
        n_observations=n_obs,
        source=source,
        models=model_metrics,
    )
