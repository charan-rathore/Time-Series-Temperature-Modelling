"""
Tests for the sensor API endpoints.

Tests the sensor reading upload and retrieval endpoints.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_forecasts.db"


@pytest.fixture
def client(temp_db_path):
    """Create a test client with mocked database path."""
    with patch("src.api.routes.sensor._DB_PATH", temp_db_path):
        with patch("src.data.baseline_collector._DB_PATH", temp_db_path):
            from src.api.routes.sensor import router
            from fastapi import FastAPI
            
            app = FastAPI()
            app.include_router(router, prefix="/api/sensor")
            
            yield TestClient(app)


class TestUploadReadings:
    def test_accepts_valid_readings(self, client):
        """Should accept valid sensor readings."""
        response = client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {
                        "timestamp": "2026-05-07T21:00:00Z",
                        "temp_c": 27.5,
                        "humidity_pct": 65.0,
                        "source": "dht22_sensor"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 1
    
    def test_filters_non_9pm_readings(self, client):
        """Should only store readings near 9 PM."""
        response = client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {"timestamp": "2026-05-07T12:00:00Z", "temp_c": 30.0},
                    {"timestamp": "2026-05-07T21:00:00Z", "temp_c": 27.0},
                    {"timestamp": "2026-05-07T03:00:00Z", "temp_c": 22.0},
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 1
    
    def test_rejects_empty_batch(self, client):
        """Should reject empty readings batch."""
        response = client.post(
            "/api/sensor/readings",
            json={"readings": []}
        )
        
        assert response.status_code == 422
    
    def test_handles_multiple_days(self, client):
        """Should handle readings from multiple days."""
        response = client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {"timestamp": "2026-05-05T21:00:00Z", "temp_c": 26.0},
                    {"timestamp": "2026-05-06T21:00:00Z", "temp_c": 27.0},
                    {"timestamp": "2026-05-07T21:00:00Z", "temp_c": 28.0},
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 3


class TestLatestReading:
    def test_returns_404_when_empty(self, client):
        """Should return 404 when no readings exist."""
        response = client.get("/api/sensor/latest")
        assert response.status_code == 404
    
    def test_returns_latest_after_upload(self, client):
        """Should return the latest reading after upload."""
        client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {"timestamp": "2026-05-07T21:00:00Z", "temp_c": 27.5}
                ]
            }
        )
        
        response = client.get("/api/sensor/latest")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sensor_temp_c"] == 27.5


class TestSensorHistory:
    def test_returns_empty_list_initially(self, client):
        """Should return empty list when no data."""
        response = client.get("/api/sensor/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["readings"] == []
        assert data["count"] == 0
    
    def test_returns_history_after_upload(self, client):
        """Should return readings after upload."""
        client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {"timestamp": "2026-05-05T21:00:00Z", "temp_c": 26.0},
                    {"timestamp": "2026-05-06T21:00:00Z", "temp_c": 27.0},
                    {"timestamp": "2026-05-07T21:00:00Z", "temp_c": 28.0},
                ]
            }
        )
        
        response = client.get("/api/sensor/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
    
    def test_respects_date_filters(self, client):
        """Should filter by date range."""
        client.post(
            "/api/sensor/readings",
            json={
                "readings": [
                    {"timestamp": "2026-05-01T21:00:00Z", "temp_c": 25.0},
                    {"timestamp": "2026-05-05T21:00:00Z", "temp_c": 26.0},
                    {"timestamp": "2026-05-10T21:00:00Z", "temp_c": 27.0},
                ]
            }
        )
        
        response = client.get(
            "/api/sensor/history",
            params={"start_date": "2026-05-04", "end_date": "2026-05-06"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
