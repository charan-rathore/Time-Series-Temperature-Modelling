"""Metrics endpoint for ThermoSense API."""

from typing import Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HorizonMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float
    skill_score: float
    coverage_90pct: Optional[float] = None
    vs_commercial_app_rmse: Optional[float] = None


class MetricsResponse(BaseModel):
    location: str
    evaluation_window_days: int
    n_observations: int
    ensemble: Dict[str, HorizonMetrics]
    lgbm: Dict[str, HorizonMetrics]
    sarima: Dict[str, HorizonMetrics]
    commercial_app: Dict[str, HorizonMetrics]


@router.get("", response_model=MetricsResponse, summary="Live model accuracy metrics")
def get_metrics(request: Request, window_days: int = 30):
    """
    Return live accuracy metrics over the last N days for all models.

    This endpoint reflects the actual predictive performance on real observations
    submitted via POST /forecast/feedback. Useful for monitoring model drift.

    Query params:
    - window_days: Evaluation window in days (default: 30)
    """
    config = request.app.state.config
    location_name = config["location"]["name"]

    # Placeholder — replace with actual metric computation from stored predictions.
    placeholder_metrics = HorizonMetrics(mae=0.0, rmse=0.0, mape=0.0, skill_score=0.0)

    return MetricsResponse(
        location=location_name,
        evaluation_window_days=window_days,
        n_observations=0,
        ensemble={"day1": placeholder_metrics, "day2": placeholder_metrics, "day3": placeholder_metrics},
        lgbm={"day1": placeholder_metrics, "day2": placeholder_metrics, "day3": placeholder_metrics},
        sarima={"day1": placeholder_metrics, "day2": placeholder_metrics, "day3": placeholder_metrics},
        commercial_app={"day1": placeholder_metrics, "day2": placeholder_metrics, "day3": placeholder_metrics},
    )
