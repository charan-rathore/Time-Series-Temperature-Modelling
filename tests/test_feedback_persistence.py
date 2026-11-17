"""A receipt must mean the observation was actually written."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.routes import forecast

@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "daily_merged.csv"
    pd.DataFrame({"date": [pd.Timestamp("2024-07-11")], "temp_c": [26.0],
                  "temp_c_api": [25.0], "is_sensor_reading": [False]}).to_csv(path, index=False)
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
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    response = send()
    assert response.status_code == 503
    assert "private/path" not in response.text
    assert pd.read_csv(store, parse_dates=["date"]).loc[0, "temp_c"] == 26.0

def test_success_receipt_follows_persisted_temperature(store):
    response = send()
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    df = pd.read_csv(store, parse_dates=["date"])
    assert df.loc[0, "temp_c"] == 27.0
    assert bool(df.loc[0, "is_sensor_reading"])
    assert df.loc[0, "api_bias"] == 2.0

def test_unreadable_store_does_not_claim_recorded(store):
    store.write_bytes(b"date,temp_c\nnot,a-valid-row\n~~~")
    response = send()
    assert response.status_code == 503
    assert "accepted" not in response.text

def test_new_date_receipt_follows_persisted_row(store):
    response = TestClient(app, raise_server_exceptions=False).post(
        "/forecast/feedback", json={"date": "2024-07-12", "actual_temp_c": 29.0})
    assert response.status_code == 200
    df = pd.read_csv(store, parse_dates=["date"])
    row = df.loc[df["date"] == pd.Timestamp("2024-07-12")].iloc[0]
    assert row["temp_c"] == 29.0
    assert bool(row["is_sensor_reading"])


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_temperature_is_rejected_and_store_unchanged(store, raw):
    before = store.read_bytes()
    response = TestClient(app, raise_server_exceptions=False).post(
        "/forecast/feedback",
        content='{"date": "2024-07-11", "actual_temp_c": %s}' % raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "accepted" not in response.text
    assert store.read_bytes() == before

def test_concurrent_feedback_receipts_do_not_lose_an_observation(store, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    real_read_csv = pd.read_csv
    real_to_csv = pd.DataFrame.to_csv
    write_started = threading.Event()
    competing_read_done = threading.Event()

    def coordinated_read(path, *args, **kwargs):
        frame = real_read_csv(path, *args, **kwargs)
        if path == store:
            competing_read_done.set()
        return frame

    def coordinated_write(self, path, *args, **kwargs):
        if path == store and not write_started.is_set():
            write_started.set()
            # On the unlocked path this lets the competing request read the
            # original CSV before the first write lands. With the fix, the
            # competitor is blocked before read and this bounded wait expires.
            competing_read_done.wait(timeout=0.3)
        return real_to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", coordinated_read)
    monkeypatch.setattr(pd.DataFrame, "to_csv", coordinated_write)

    payloads = [
        {"date": "2024-07-11", "actual_temp_c": 27.0},
        {"date": "2024-07-12", "actual_temp_c": 29.0},
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda payload: TestClient(app, raise_server_exceptions=False).post(
                "/forecast/feedback", json=payload),
            payloads,
        ))
    assert [response.status_code for response in responses] == [200, 200]
    frame = real_read_csv(store, parse_dates=["date"])
    actuals = {
        row.date.strftime("%Y-%m-%d"): row.temp_c
        for row in frame.itertuples(index=False)
    }
    assert actuals["2024-07-11"] == 27.0
    assert actuals["2024-07-12"] == 29.0
