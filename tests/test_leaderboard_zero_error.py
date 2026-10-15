"""Zero forecast error is a measured value, never missing data."""
from datetime import date, timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from src.api.routes import leaderboard as route
from src.data import baseline_collector as collector

@pytest.fixture
def client(tmp_path, monkeypatch):
    path = collector.init_database(tmp_path / "forecasts.db")
    monkeypatch.setattr(collector, "_DB_PATH", path)
    app = FastAPI()
    app.include_router(route.router, prefix="/board")
    with TestClient(app) as client:
        yield client, path

def populate(path, ts_error, commercial_error):
    # Three dated real SQL pairs exercise the leaderboard minimum sample gate.
    for n in range(3):
        day = date.today() - timedelta(days=n+1)
        collector.store_actual(path, day, 25.)
        collector.store_forecast(path, day, "thermosense", 1, 25.+ts_error)
        collector.store_forecast(path, day, "open_meteo", 1, 25.+commercial_error)

def test_comparison_retains_exact_zero_error(client):
    api, path = client
    populate(path, 0., 2.)
    response = api.get("/board/comparison")
    assert response.status_code == 200
    for row in response.json()["data"]:
        assert row["forecasts"]["thermosense"]["day1"]["error"] == 0.
        assert row["forecasts"]["open_meteo"]["day1"]["error"] == 2.

def test_perfect_thermosense_has_defined_100_percent_improvement(client):
    api, path = client
    populate(path, 0., 2.)
    response = api.get("/board")
    assert response.status_code == 200
    assert response.json()["improvement_vs_best_commercial_pct"] == 100.

def test_perfect_commercial_baseline_has_undefined_relative_improvement(client):
    api, path = client
    populate(path, 1., 0.)
    assert api.get("/board").json()["improvement_vs_best_commercial_pct"] is None

def test_normal_relative_improvement_unchanged(client):
    api, path = client
    populate(path, 1., 2.)
    assert api.get("/board").json()["improvement_vs_best_commercial_pct"] == 50.

def test_leaderboard_keeps_zero_mae_and_mean_error(client):
    api, path = client
    populate(path, 0., 2.)
    ts = next(row for row in api.get("/board").json()["rankings"] if row["source"] == "thermosense")
    assert ts["mae"] == 0.
    assert ts["mean_error"] == 0.

def test_absent_forecast_keeps_missing_error_state(client):
    api, path = client
    day = date.today() - timedelta(days=1)
    collector.store_actual(path, day, 25.)
    data = api.get("/board/comparison").json()["data"]
    assert len(data) == 1
    assert data[0]["forecasts"] == {}

def test_balanced_nonzero_errors_keep_measured_zero_mean(client):
    api, path = client
    for n, error in enumerate([-2., 0., 2.]):
        day = date.today() - timedelta(days=n+1)
        collector.store_actual(path, day, 25.)
        collector.store_forecast(path, day, "thermosense", 1, 25.+error)
    ts = api.get("/board").json()["rankings"][0]
    assert ts["mae"] == 1.333
    assert ts["mean_error"] == 0.
    assert api.get("/board").json()["improvement_vs_best_commercial_pct"] is None
