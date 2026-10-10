"""
ThermoSense FastAPI application.

Serves real-time temperature forecasts, historical data, live accuracy metrics,
a feedback endpoint, pipeline/training control, and the React dashboard UI.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

from .routes import forecast, history, metrics, pipeline, leaderboard, sensor, statistics, locations

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Prefer Vercel CDN output (public/), fall back to local CRA build.
_PUBLIC_DIR = _PROJECT_ROOT / "public"
_CRA_BUILD_DIR = _PROJECT_ROOT / "frontend" / "build"
FRONTEND_DIR = _PUBLIC_DIR if (_PUBLIC_DIR / "index.html").exists() else _CRA_BUILD_DIR


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and configuration on startup; clean up on shutdown."""
    try:
        config = load_config()
    except Exception as exc:
        print(f"[ThermoSense API] Config load failed: {exc}")
        config = {}
    app.state.config = config

    try:
        from src.models.loader import ModelManager
        manager = ModelManager()
        manager.load_models(config)
        app.state.model_manager = manager
    except Exception as exc:
        # Keep the API/dashboard up even if ML deps fail on the host (e.g. missing libgomp).
        print(f"[ThermoSense API] Model manager unavailable: {exc}")
        app.state.model_manager = None

    print("[ThermoSense API] Started. Dashboard at /  |  API docs at /docs")
    yield
    print("[ThermoSense API] Shutting down.")


app = FastAPI(
    title="ThermoSense - Hyperlocal Temperature Intelligence",
    description=(
        "REST API for real-time temperature forecasting using SARIMA(X), "
        "LightGBM, and ensemble stacking models. "
        "Beats commercial weather apps by learning hyperlocal sensor bias."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["Leaderboard"])
app.include_router(sensor.router, prefix="/api/sensor", tags=["Sensor"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["Statistics"])
app.include_router(locations.router, prefix="/api/locations", tags=["Locations"])

# Keep legacy routes for backward compatibility
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast (legacy)"], include_in_schema=False)
app.include_router(history.router, prefix="/history", tags=["History (legacy)"], include_in_schema=False)
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics (legacy)"], include_in_schema=False)


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/ready", tags=["Health"])
def readiness():
    """Readiness for model-backed forecasts, not merely a running process."""
    manager = getattr(app.state, "model_manager", None)
    config = getattr(app.state, "config", None)
    location = config.get("location") if isinstance(config, dict) else None
    name = location.get("name") if isinstance(location, dict) else None
    data_ready = False
    try:
        import numpy as np
        import pandas as pd
        from src.data.preprocess import load_processed
        from src.features.engineer import build_feature_matrix
        processed = load_processed()
        if not processed.empty and {"date", "temp_c"}.issubset(processed.columns):
            dates = pd.to_datetime(processed["date"], errors="coerce")
            temperatures = pd.to_numeric(processed["temp_c"], errors="coerce")
            if dates.notna().all() and not dates.dt.normalize().duplicated().any() and np.isfinite(temperatures.dropna()).all():
                processed = processed.copy()
                processed["temp_c"] = temperatures
                features = build_feature_matrix(processed, drop_na=True)
                numeric = features.select_dtypes(include="number")
                required_rows = 1  # LightGBM uses features.tail(1).
                sarima = getattr(manager, "sarima", None)
                if sarima is not None and sarima.is_fitted:
                    required_rows = 3  # SARIMA receives tail(horizon), up to three days.
                tft = getattr(manager, "tft", None)
                if tft is not None and tft.is_fitted:
                    # Use loaded dataset lengths when available, not mutable API config.
                    dataset = getattr(tft, "train_dataset", None)
                    encoder = getattr(dataset, "max_encoder_length", tft.config.get("max_encoder_length", 30))
                    decoder = getattr(dataset, "max_prediction_length", tft.config.get("max_prediction_length", 3))
                    required_rows = max(required_rows, int(encoder) + max(3, int(decoder)))
                data_ready = len(features) >= required_rows and np.isfinite(numeric.to_numpy()).all()
    except Exception:
        # Match inference preparation without leaking file paths or data in the probe.
        pass
    checks = {
        "config": isinstance(name, str) and bool(name.strip()),
        "models": bool(manager and manager.is_loaded),
        "data": bool(data_ready),
    }
    ready = all(checks.values())
    return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)


@app.get("/health", tags=["Health"], include_in_schema=False)
def health_legacy():
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    # Vercel-style public/ output has no static/ folder; mounting it would
    # raise RuntimeError at import time and take the whole API down.
    if (FRONTEND_DIR / "static").is_dir():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        frontend_root = FRONTEND_DIR.resolve()
        file_path = (frontend_root / full_path).resolve()
        if file_path.is_relative_to(frontend_root) and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_root / "index.html"))
else:
    @app.get("/", tags=["Health"])
    def root():
        return {
            "service": "ThermoSense API",
            "version": "2.0.0",
            "dashboard": "Build frontend first: cd frontend && npm run build",
            "docs": "/docs",
            "status": "healthy",
        }
