"""
Forecast endpoint for ThermoSense API.

GET /forecast
  Returns 1-3 day temperature predictions with uncertainty intervals.

POST /feedback
  Accepts the actual observed temperature for a past date.
  Appends to the local data store and updates the api_bias rolling feature.
"""

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = _PROJECT_ROOT / "data" / "bundle" / "daily_merged.csv"
REGISTRY_PATH = _PROJECT_ROOT / "models" / "registry.json"

_UNTRAINED_MODELS = {"climatology", "fallback", "placeholder", "unknown"}
# These models predict `horizon` days after the last row of the processed
# dataset (they are fed features.tail(1) / the feature matrix), so their target
# date is data_through + horizon. SARIMA and the ensemble depend on the SARIMA
# fit end, which is not recorded anywhere, so their target date is unverified.
_ORIGIN_VERIFIED_MODELS = {"lgbm", "tft"}
# Fixed stand-in values with no model behind them: never dated, never current.
_SYNTHETIC_MODELS = {"placeholder", "unknown"}


def _served_artifact_hashes(manager, model_name: str) -> Optional[dict]:
    """Load-time identity of the artifacts behind the served model: the
    hashes ModelManager recorded from the exact bytes it deserialized. A
    file swapped on disk later cannot re-identify the in-memory model."""
    recorded = getattr(manager, "artifact_hashes", None) if manager is not None else None
    if not recorded:
        return None
    if model_name == "lgbm":
        selected = {n: h for n, h in recorded.items() if n.startswith("lgbm_h") and n.endswith(".pkl")}
    elif model_name in ("sarima", "tft", "ensemble"):
        wanted = {"sarima": "sarima.pkl", "tft": "tft.pkl", "ensemble": "ensemble.pkl"}[model_name]
        selected = {n: h for n, h in recorded.items() if n == wanted}
    else:
        return None
    return selected or None


def _artifacts_match(version: dict, loaded_hashes: dict) -> bool:
    """Every loaded artifact must hash to what the version recorded at
    training time. Anything else - replaced file, stale registry, a record
    that never tracked this artifact - means the binding is unverified."""
    recorded = version.get("artifacts") or {}
    if not loaded_hashes:
        return False
    for name, actual in loaded_hashes.items():
        if recorded.get(name) != actual:
            return False
    return True


def _training_provenance(model_name: str, loaded_hashes: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """Training window + timestamp of the version actually serving `model_name`.

    A registry record only earns the claim when its identity is verified
    end to end: the record must be for the served model, selected by the
    champion pointer when one exists (never bare versions[-1], which may be
    an unrelated later record), AND every artifact file the manager loaded
    must hash to what the record registered. Any gap - missing registry,
    ambiguous candidates, wrong-model champion, stale or replaced artifact -
    means the serving version is unknown, reported as no provenance, never
    guessed.
    """
    if not loaded_hashes:
        return None, None
    try:
        registry = json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return None, None
    versions = [v for v in (registry.get("versions") or []) if isinstance(v, dict)]
    candidates = [v for v in versions if v.get("model") == model_name]
    champion = (registry.get("champions") or {}).get(model_name)
    if champion:
        candidates = [v for v in candidates if v.get("version") == champion]
    verified = [v for v in candidates if _artifacts_match(v, loaded_hashes)]
    if len(verified) != 1:
        return None, None
    return verified[0].get("data_range"), verified[0].get("created_at")


class ForecastPoint(BaseModel):
    # Verified calendar date this value is for. None when the model's forecast
    # origin cannot be tied to a date (SARIMA/ensemble fit end is not recorded),
    # so it is never labelled as "tomorrow" by guesswork.
    date: Optional[str] = None
    is_current: bool = True
    # True for the fixed placeholder number: not model output, not a forecast.
    synthetic: bool = False
    predicted_temp_c: float
    lower_bound_c: float
    upper_bound_c: float
    horizon_days: int
    confidence: str = "90%"
    model_used: Optional[str] = None
    trained_data_range: Optional[str] = None
    model_trained_at: Optional[str] = None


class ForecastResponse(BaseModel):
    location: str
    generated_at: str
    model_used: str
    forecasts: List[ForecastPoint]
    trained_data_range: Optional[str] = None
    model_trained_at: Optional[str] = None
    # Last day of the dataset the forecast inputs come from, and whether every
    # point is a real forward-looking forecast (target date after today).
    data_through: Optional[str] = None
    is_current: bool = True


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

    def point_provenance(point_model: str) -> tuple[Optional[str], Optional[str]]:
        if point_model in _UNTRAINED_MODELS:
            return None, None
        return _training_provenance(point_model, _served_artifact_hashes(manager, point_model))

    through_fn = getattr(manager, "data_through", None)
    data_through = through_fn() if callable(through_fn) else None

    def target_for(point_model: str, d: int) -> tuple[Optional[str], bool]:
        if point_model in _SYNTHETIC_MODELS:
            return None, False
        if point_model in _UNTRAINED_MODELS or data_through is None:
            return (today + timedelta(days=d)).isoformat(), True
        if point_model in _ORIGIN_VERIFIED_MODELS:
            target = data_through + timedelta(days=d)
            return target.isoformat(), target > today
        return None, False

    forecasts = []
    for d in range(1, days + 1):
        if d - 1 < len(raw_forecasts):
            fc = raw_forecasts[d - 1]
            # Horizon fallback can serve a different model per row; each
            # point carries the provenance of the model that made it.
            point_model = fc.get("model_used") or "unknown"
            p_range, p_at = point_provenance(point_model)
            point_date, point_current = target_for(point_model, d)
            forecasts.append(ForecastPoint(
                date=point_date,
                is_current=point_current,
                synthetic=point_model in _SYNTHETIC_MODELS,
                predicted_temp_c=fc["predicted_temp_c"],
                lower_bound_c=fc["lower_bound_c"],
                upper_bound_c=fc["upper_bound_c"],
                horizon_days=d,
                model_used=point_model,
                trained_data_range=p_range,
                model_trained_at=p_at,
            ))
        else:
            forecasts.append(ForecastPoint(
                date=None,
                is_current=False,
                synthetic=True,
                predicted_temp_c=26.0,
                lower_bound_c=24.5,
                upper_bound_c=27.5,
                horizon_days=d,
                model_used="placeholder",
            ))

    # Response-level provenance describes the headline (day-1) model, which
    # is what `model_used` reports; per-horizon detail lives on the points.
    trained_range, trained_at = (None, None)
    if model_used not in _UNTRAINED_MODELS:
        trained_range, trained_at = _training_provenance(model_used, _served_artifact_hashes(manager, model_used))

    return ForecastResponse(
        location=location_name,
        generated_at=datetime.utcnow().isoformat() + "Z",
        model_used=model_used,
        forecasts=forecasts,
        trained_data_range=trained_range,
        model_trained_at=trained_at,
        data_through=data_through.isoformat() if data_through else None,
        is_current=all(f.is_current for f in forecasts),
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

    if not math.isfinite(payload.actual_temp_c):
        raise HTTPException(status_code=400, detail="actual_temp_c must be a finite number")

    if obs_date > date.today():
        raise HTTPException(status_code=400, detail="Cannot submit feedback for a future date")

    bias_updated = False
    try:
        if not PROCESSED_PATH.exists():
            raise FileNotFoundError("Processed observation store is unavailable")
        df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
        obs_dt = pd.Timestamp(obs_date)

        if obs_dt in df["date"].values:
            idx = df.index[df["date"] == obs_dt][0]
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

        df.to_csv(PROCESSED_PATH, index=False)
        # Mirror into the parquet store that training and ModelManager
        # read via load_processed(); never fail the receipt over it.
        try:
            parquet_path = (
                PROCESSED_PATH.parent.parent / "processed" / "daily_merged.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
        except Exception as mirror_exc:
            print(f"[feedback] parquet mirror skipped: {mirror_exc}")
    except Exception as e:
        print(f"[feedback] Error persisting observation: {e}")
        raise HTTPException(status_code=503, detail="Observation could not be recorded; please retry") from e

    return FeedbackResponse(
        status="accepted",
        message=f"Observation for {obs_date} recorded: {payload.actual_temp_c}°C",
        api_bias_updated=bias_updated,
    )
