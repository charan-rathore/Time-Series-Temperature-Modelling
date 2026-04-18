"""Tests for the FastAPI endpoints using httpx AsyncClient."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


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


def test_forecast_single_day():
    response = client.get("/forecast?days=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 1
    assert data["forecasts"][0]["horizon_days"] == 1


def test_forecast_invalid_days():
    response = client.get("/forecast?days=5")
    assert response.status_code == 400


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
    assert "ensemble" in data
    assert "day1" in data["ensemble"]


def test_history_endpoint():
    response = client.get("/history?start=2024-06-01&end=2024-07-11")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
