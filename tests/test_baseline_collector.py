"""
Tests for the baseline collector module.

Tests the forecast collection, storage, and leaderboard computation.
"""

import json
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.baseline_collector import (
    init_database,
    store_forecast,
    store_actual,
    get_leaderboard,
    get_collection_status,
    fetch_open_meteo_forecast,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_forecasts.db"
        init_database(db_path)
        yield db_path


class TestDatabaseInit:
    def test_creates_tables(self, temp_db):
        """Database should create all required tables."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        assert "daily_forecasts" in tables
        assert "daily_actuals" in tables
        assert "collection_log" in tables
    
    def test_idempotent(self, temp_db):
        """Multiple init calls should not fail."""
        init_database(temp_db)
        init_database(temp_db)


class TestStoreForecast:
    def test_stores_forecast(self, temp_db):
        """Should store a forecast successfully."""
        result = store_forecast(
            temp_db,
            forecast_date=date(2026, 5, 8),
            source="test_source",
            horizon_days=1,
            predicted_temp_c=25.5,
        )
        assert result is True
    
    def test_rejects_duplicate(self, temp_db):
        """Should reject duplicate forecast (same date, source, horizon)."""
        store_forecast(
            temp_db,
            forecast_date=date(2026, 5, 8),
            source="test_source",
            horizon_days=1,
            predicted_temp_c=25.5,
        )
        
        result = store_forecast(
            temp_db,
            forecast_date=date(2026, 5, 8),
            source="test_source",
            horizon_days=1,
            predicted_temp_c=26.0,
        )
        assert result is False
    
    def test_allows_different_horizons(self, temp_db):
        """Should allow same date/source with different horizons."""
        r1 = store_forecast(temp_db, date(2026, 5, 8), "src", 1, 25.0)
        r2 = store_forecast(temp_db, date(2026, 5, 8), "src", 2, 25.5)
        r3 = store_forecast(temp_db, date(2026, 5, 8), "src", 3, 26.0)
        
        assert r1 and r2 and r3


class TestStoreActual:
    def test_stores_actual(self, temp_db):
        """Should store an actual reading."""
        result = store_actual(
            temp_db,
            date_=date(2026, 5, 7),
            sensor_temp_c=27.3,
            sensor_humidity_pct=65.0,
        )
        assert result is True
    
    def test_replaces_on_conflict(self, temp_db):
        """Should replace existing actual for same date."""
        store_actual(temp_db, date(2026, 5, 7), 27.0)
        store_actual(temp_db, date(2026, 5, 7), 28.0)
        
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT sensor_temp_c FROM daily_actuals WHERE date = '2026-05-07'"
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == 28.0


class TestLeaderboard:
    def test_empty_without_data(self, temp_db):
        """Should return empty list without data."""
        board = get_leaderboard(temp_db)
        assert board == []
    
    def test_computes_rankings(self, temp_db):
        """Should compute correct RMSE rankings."""
        for i in range(10):
            d = date(2026, 5, 1) + timedelta(days=i)
            
            store_actual(temp_db, d, sensor_temp_c=25.0)
            
            store_forecast(temp_db, d, "good_model", 1, 25.5)
            store_forecast(temp_db, d, "bad_model", 1, 27.0)
        
        board = get_leaderboard(temp_db, window_days=30, horizon=1)
        
        assert len(board) == 2
        assert board[0]["source"] == "good_model"
        assert board[1]["source"] == "bad_model"
        assert board[0]["rmse"] < board[1]["rmse"]
    
    def test_requires_minimum_samples(self, temp_db):
        """Should require at least 3 samples per source."""
        store_actual(temp_db, date(2026, 5, 1), 25.0)
        store_forecast(temp_db, date(2026, 5, 1), "sparse", 1, 25.5)
        
        board = get_leaderboard(temp_db)
        assert len(board) == 0


class TestCollectionStatus:
    def test_uninitialized(self):
        """Should report uninitialized for non-existent DB."""
        status = get_collection_status(Path("/nonexistent/path.db"))
        assert status["initialized"] is False
    
    def test_reports_counts(self, temp_db):
        """Should report correct counts."""
        for i in range(5):
            d = date(2026, 5, 1) + timedelta(days=i)
            store_forecast(temp_db, d, "test", 1, 25.0)
        
        for i in range(3):
            d = date(2026, 5, 1) + timedelta(days=i)
            store_actual(temp_db, d, 25.0)
        
        status = get_collection_status(temp_db)
        
        assert status["initialized"] is True
        assert status["forecast_count"] == 5
        assert status["actual_count"] == 3


class TestOpenMeteoFetch:
    @patch("src.data.baseline_collector.requests.get")
    def test_parses_response(self, mock_get):
        """Should parse Open-Meteo response correctly."""
        today = date.today()
        tomorrow = today + timedelta(days=1)
        day_after = today + timedelta(days=2)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "daily": {
                "time": [
                    today.isoformat(),
                    tomorrow.isoformat(),
                    day_after.isoformat(),
                ],
                "temperature_2m_mean": [25.0, 26.0, 27.0],
                "temperature_2m_min": [22.0, 23.0, 24.0],
                "temperature_2m_max": [28.0, 29.0, 30.0],
            }
        }
        mock_get.return_value = mock_response
        
        forecasts = fetch_open_meteo_forecast(12.97, 77.59, "Asia/Kolkata")
        
        assert len(forecasts) >= 1
        
        day1 = next((f for f in forecasts if f["horizon_days"] == 1), None)
        if day1:
            assert day1["source"] == "open_meteo"
            assert day1["predicted_temp_c"] == 26.0
