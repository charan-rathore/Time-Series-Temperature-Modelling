"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_config

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def setup_app_state():
    """Ensure app.state is populated before each test."""
    config = load_config()
    app.state.config = config

    from src.models.loader import ModelManager
    if not hasattr(app.state, "model_manager"):
        manager = ModelManager()
        try:
            manager.load_models(config)
        except Exception:
            pass
        app.state.model_manager = manager
    yield


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_default():
    response = client.get("/forecast")
    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    assert len(data["forecasts"]) == 3
    assert "model_used" in data
    assert "location" in data


def test_forecast_single_day():
    response = client.get("/forecast?days=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 1
    assert data["forecasts"][0]["horizon_days"] == 1


def test_forecast_invalid_days():
    response = client.get("/forecast?days=5")
    assert response.status_code == 400


def test_forecast_response_has_bounds():
    response = client.get("/forecast?days=1")
    assert response.status_code == 200
    fc = response.json()["forecasts"][0]
    assert "predicted_temp_c" in fc
    assert "lower_bound_c" in fc
    assert "upper_bound_c" in fc
    assert fc["lower_bound_c"] <= fc["predicted_temp_c"] <= fc["upper_bound_c"]


def test_feedback_valid():
    response = client.post("/forecast/feedback", json={
        "date": "2024-07-11",
        "actual_temp_c": 27.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"


def test_feedback_future_date():
    response = client.post("/forecast/feedback", json={
        "date": "2099-01-01",
        "actual_temp_c": 30.0
    })
    assert response.status_code == 400


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "location" in data
    assert "n_observations" in data


def test_history_endpoint():
    response = client.get("/history?start=2024-06-01&end=2024-07-11")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert "total_records" in data


def test_history_invalid_start():
    response = client.get("/history?start=invalid")
    assert response.status_code == 400


def test_history_invalid_end():
    response = client.get("/history?start=2024-06-01&end=invalid")
    assert response.status_code == 400


def test_pipeline_status():
    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "data_available" in data
    assert "models_available" in data
    assert "active_jobs" in data


def test_pipeline_logs():
    response = client.get("/api/pipeline/logs")
    assert response.status_code == 200
    data = response.json()
    assert "lines" in data
    assert "total" in data


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"
