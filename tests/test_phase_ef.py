"""
Tests for Phase E and F implementations.

Tests the statistical API routes, weekly report generator,
and multi-location support.
"""

import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.data.baseline_collector import (
    init_database,
    store_forecast,
    store_actual,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_forecasts.db"
        init_database(db_path)
        
        np.random.seed(42)
        for i in range(30):
            d = date.today() - timedelta(days=i+1)
            actual = 25.0 + np.random.normal(0, 1)
            store_actual(db_path, d, sensor_temp_c=actual)
            store_forecast(db_path, d, "thermosense", 1, actual + np.random.normal(0, 0.3))
            store_forecast(db_path, d, "open_meteo", 1, actual + np.random.normal(1.0, 0.5))
            store_forecast(db_path, d, "openweathermap", 1, actual + np.random.normal(1.5, 0.6))
        
        yield db_path


class TestStatisticsAPI:
    """Tests for the statistics API routes."""
    
    def test_get_errors_for_source(self, temp_db):
        """Should retrieve errors for a source."""
        from src.api.routes.statistics import _get_errors_for_source
        
        with patch("src.api.routes.statistics._DB_PATH", temp_db):
            errors = _get_errors_for_source("thermosense", 1, 30)
            assert len(errors) > 0
            assert all(e >= 0 for e in errors)
    
    def test_compare_endpoint_returns_stats(self, temp_db):
        """Compare endpoint should return statistical test results."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.statistics import router
        
        with patch("src.api.routes.statistics._DB_PATH", temp_db):
            app = FastAPI()
            app.include_router(router, prefix="/api/statistics")
            client = TestClient(app)
            
            response = client.get("/api/statistics/compare?baseline=open_meteo")
            assert response.status_code == 200
            
            data = response.json()
            assert "statistical_test" in data
            assert "effect_size" in data
            assert "skill_score" in data
            assert "percentage_improvement" in data
    
    def test_all_comparisons_endpoint(self, temp_db):
        """All comparisons endpoint should compare against all baselines."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.statistics import router
        
        with patch("src.api.routes.statistics._DB_PATH", temp_db):
            app = FastAPI()
            app.include_router(router, prefix="/api/statistics")
            client = TestClient(app)
            
            response = client.get("/api/statistics/all-comparisons")
            assert response.status_code == 200
            
            data = response.json()
            assert "comparisons" in data
            assert data["total_comparisons"] >= 2
    
    def test_summary_endpoint(self, temp_db):
        """Summary endpoint should return stats for all sources."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.statistics import router
        
        with patch("src.api.routes.statistics._DB_PATH", temp_db):
            app = FastAPI()
            app.include_router(router, prefix="/api/statistics")
            client = TestClient(app)
            
            response = client.get("/api/statistics/summary")
            assert response.status_code == 200
            
            data = response.json()
            assert "sources" in data
            assert len(data["sources"]) >= 3


class TestWeeklyReport:
    """Tests for the weekly report generator."""
    
    def test_get_errors_for_source(self, temp_db):
        """Should retrieve errors from database."""
        from scripts.generate_report import get_errors_for_source
        
        errors = get_errors_for_source(temp_db, "thermosense", 1, 30)
        assert len(errors) > 0
    
    def test_get_worst_predictions(self, temp_db):
        """Should retrieve worst predictions."""
        from scripts.generate_report import get_worst_predictions
        
        worst = get_worst_predictions(temp_db, "thermosense", 1, 30, limit=5)
        assert len(worst) <= 5
        if len(worst) > 1:
            assert worst[0]["error"] >= worst[1]["error"]
    
    def test_get_data_completeness(self, temp_db):
        """Should compute data completeness metrics."""
        from scripts.generate_report import get_data_completeness
        
        result = get_data_completeness(temp_db, 30)
        assert "sensor_completeness_pct" in result
        assert result["sensor_completeness_pct"] >= 0
    
    def test_generate_weekly_report(self, temp_db):
        """Should generate a complete weekly report."""
        from scripts.generate_report import generate_weekly_report
        
        report = generate_weekly_report(db_path=temp_db, window_days=30, horizon=1)
        
        assert "leaderboard" in report
        assert "statistical_comparisons" in report
        assert "data_quality" in report
        assert "summary" in report
        assert "worst_predictions" in report


class TestLocationsAPI:
    """Tests for the locations API routes."""
    
    def test_list_locations(self):
        """Should list all configured locations."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.locations import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/locations")
        client = TestClient(app)
        
        response = client.get("/api/locations")
        assert response.status_code == 200
        
        data = response.json()
        assert "locations" in data
        assert data["total"] >= 1
    
    def test_locations_summary(self, temp_db):
        """Should return summary across locations."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.locations import router, _DB_BASE_PATH
        
        with patch.object(
            __import__("src.api.routes.locations", fromlist=["_DB_BASE_PATH"]),
            "_DB_BASE_PATH",
            temp_db.parent,
        ):
            app = FastAPI()
            app.include_router(router, prefix="/api/locations")
            client = TestClient(app)
            
            response = client.get("/api/locations/summary")
            assert response.status_code == 200
            
            data = response.json()
            assert "locations" in data
            assert "total_active_locations" in data
    
    def test_get_nonexistent_location(self):
        """Should return 404 for nonexistent location."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.locations import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/locations")
        client = TestClient(app)
        
        response = client.get("/api/locations/nonexistent_location_id")
        assert response.status_code == 404


class TestIntegration:
    """Integration tests for all Phase E and F components."""
    
    def test_full_pipeline(self, temp_db):
        """Test complete pipeline from data to report."""
        from scripts.generate_report import generate_weekly_report
        from src.evaluation.statistical_tests import compare_forecasters
        from src.data.baseline_collector import get_leaderboard
        
        leaderboard = get_leaderboard(temp_db, 30, 1)
        assert len(leaderboard) >= 3
        
        report = generate_weekly_report(db_path=temp_db)
        
        assert report["summary"]["thermosense_rank"] == 1
        
        assert len(report["statistical_comparisons"]) >= 2
        
        significant = sum(1 for c in report["statistical_comparisons"] if c["significant"])
        assert significant >= 1
