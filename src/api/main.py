"""
ThermoSense FastAPI application.

Serves real-time temperature forecasts, historical data, live accuracy metrics,
and a feedback endpoint to close the prediction loop with actual observations.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .routes import forecast, history, metrics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and configuration on startup; clean up on shutdown."""
    config = load_config()
    app.state.config = config
    # Model loading is handled lazily in the forecast route on first request.
    # For production, load the champion model from MLflow here:
    #   import mlflow.pyfunc
    #   app.state.model = mlflow.pyfunc.load_model("models:/thermosense-champion/Production")
    print("[ThermoSense API] Started. Docs available at /docs")
    yield
    print("[ThermoSense API] Shutting down.")


app = FastAPI(
    title="ThermoSense — Hyperlocal Temperature Intelligence",
    description=(
        "REST API for real-time temperature forecasting using SARIMA(X), "
        "LightGBM, and Temporal Fusion Transformer models. "
        "Beats commercial weather apps by learning hyperlocal sensor bias."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "ThermoSense API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
