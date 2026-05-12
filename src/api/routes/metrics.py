"""Metrics endpoint for ThermoSense API."""

import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = _PROJECT_ROOT / "models" / "results.json"


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
    models: Dict[str, Dict[str, HorizonMetrics]]


@router.get("", response_model=MetricsResponse, summary="Live model accuracy metrics")
def get_metrics(request: Request, window_days: int = 30):
    """
    Return accuracy metrics for all trained models.

    Reads from the results.json file generated during training.
    After training, metrics are populated with real evaluation data.
    """
    config = request.app.state.config
    location_name = config["location"]["name"]

    model_metrics: Dict[str, Dict[str, HorizonMetrics]] = {}
    n_obs = 0

    manager = getattr(request.app.state, "model_manager", None)
    results = {}
    if manager is not None:
        results = manager.get_results()

    if not results and RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
        except Exception:
            pass

    if results:
        for model_name, horizons in results.items():
            if not isinstance(horizons, dict):
                continue
            model_metrics[model_name] = {}
            for horizon_key, metrics in horizons.items():
                if not isinstance(metrics, dict):
                    continue
                model_metrics[model_name][horizon_key] = HorizonMetrics(
                    mae=metrics.get("mae", 0.0),
                    rmse=metrics.get("rmse", 0.0),
                    mape=metrics.get("mape", 0.0),
                    skill_score=metrics.get("skill_score", 0.0),
                    coverage_90pct=metrics.get("coverage_90pct"),
                )
                n_obs = max(n_obs, 1)

    if not model_metrics:
        placeholder = HorizonMetrics()
        model_metrics = {
            "no_models_trained": {
                "day1": placeholder,
                "day2": placeholder,
                "day3": placeholder,
            }
        }

    return MetricsResponse(
        location=location_name,
        evaluation_window_days=window_days,
        n_observations=n_obs,
        models=model_metrics,
    )
