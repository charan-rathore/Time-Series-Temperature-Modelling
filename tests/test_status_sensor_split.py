"""Regression tests for the station-readings vs grid-history split in /api/pipeline/status.

The dashboard used to show total grid history rows as "Days recorded", implying
hundreds of station readings that do not exist (the repo bundle holds 871 daily
rows but only 40 real station readings). The status endpoint must report the two
counts separately so the UI cannot blur them again.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_config
import src.api.routes.pipeline as pipeline


@pytest.fixture
def client():
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
    return TestClient(app, raise_server_exceptions=False)


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_status_splits_station_readings_from_grid_history(client, tmp_path, monkeypatch):
    csv_path = tmp_path / "daily_merged.csv"
    _write_csv(csv_path, [
        {"date": "2024-06-01", "temp_c": 20.0, "is_sensor_reading": False},
        {"date": "2024-06-02", "temp_c": 21.0, "is_sensor_reading": True},
        {"date": "2024-06-03", "temp_c": 22.0, "is_sensor_reading": False},
        {"date": "2024-07-11", "temp_c": 23.0, "is_sensor_reading": True},
    ])
    monkeypatch.setattr(pipeline, "PROCESSED_PATH", csv_path)

    data = client.get("/api/pipeline/status").json()
    assert data["data_rows"] == 4
    assert data["sensor_reading_rows"] == 2
    assert data["sensor_date_range"] == "2024-06-02 to 2024-07-11"


def test_status_reports_zero_sensor_rows_when_all_grid(client, tmp_path, monkeypatch):
    csv_path = tmp_path / "daily_merged.csv"
    _write_csv(csv_path, [
        {"date": "2024-06-01", "temp_c": 20.0, "is_sensor_reading": False},
        {"date": "2024-06-02", "temp_c": 21.0, "is_sensor_reading": False},
    ])
    monkeypatch.setattr(pipeline, "PROCESSED_PATH", csv_path)

    data = client.get("/api/pipeline/status").json()
    assert data["data_rows"] == 2
    assert data["sensor_reading_rows"] == 0
    assert data["sensor_date_range"] is None


def test_status_tolerates_missing_sensor_column(client, tmp_path, monkeypatch):
    csv_path = tmp_path / "daily_merged.csv"
    _write_csv(csv_path, [
        {"date": "2024-06-01", "temp_c": 20.0},
        {"date": "2024-06-02", "temp_c": 21.0},
    ])
    monkeypatch.setattr(pipeline, "PROCESSED_PATH", csv_path)

    data = client.get("/api/pipeline/status").json()
    assert data["data_rows"] == 2
    assert data["sensor_reading_rows"] == 0
    assert data["sensor_date_range"] is None


def test_status_counts_string_true_sensor_flags(client, tmp_path, monkeypatch):
    csv_path = tmp_path / "daily_merged.csv"
    _write_csv(csv_path, [
        {"date": "2024-06-01", "temp_c": 20.0, "is_sensor_reading": "true"},
        {"date": "2024-06-02", "temp_c": 21.0, "is_sensor_reading": "1"},
        {"date": "2024-06-03", "temp_c": 22.0, "is_sensor_reading": "false"},
    ])
    monkeypatch.setattr(pipeline, "PROCESSED_PATH", csv_path)

    data = client.get("/api/pipeline/status").json()
    assert data["sensor_reading_rows"] == 2
