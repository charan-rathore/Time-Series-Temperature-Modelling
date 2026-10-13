"""A receipt must mean the observation was actually written."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.routes import forecast

@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "daily_merged.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2024-07-11")], "temp_c": [26.0],
                  "temp_c_api": [25.0], "is_sensor_reading": [False]}).to_parquet(path)
    monkeypatch.setattr(forecast, "PROCESSED_PATH", path)
    return path

def send():
    return TestClient(app, raise_server_exceptions=False).post(
        "/forecast/feedback", json={"date": "2024-07-11", "actual_temp_c": 27.0})

def test_missing_store_does_not_claim_recorded(store):
    store.unlink()
    response = send()
    assert response.status_code == 503
    assert "accepted" not in response.text

def test_failed_write_does_not_claim_recorded(store, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("private/path/disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    response = send()
    assert response.status_code == 503
    assert "private/path" not in response.text
    assert pd.read_parquet(store).loc[0, "temp_c"] == 26.0

def test_success_receipt_follows_persisted_temperature(store):
    response = send()
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    df = pd.read_parquet(store)
    assert df.loc[0, "temp_c"] == 27.0
    assert bool(df.loc[0, "is_sensor_reading"])
    assert df.loc[0, "api_bias"] == 2.0

def test_unreadable_store_does_not_claim_recorded(store):
    store.write_bytes(b"not a parquet file")
    response = send()
    assert response.status_code == 503
    assert "accepted" not in response.text

def test_new_date_receipt_follows_persisted_row(store):
    response = TestClient(app, raise_server_exceptions=False).post(
        "/forecast/feedback", json={"date": "2024-07-12", "actual_temp_c": 29.0})
    assert response.status_code == 200
    df = pd.read_parquet(store)
    row = df.loc[df["date"] == pd.Timestamp("2024-07-12")].iloc[0]
    assert row["temp_c"] == 29.0
    assert bool(row["is_sensor_reading"])
