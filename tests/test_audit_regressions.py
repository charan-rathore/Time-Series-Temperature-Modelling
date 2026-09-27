"""Guard sensor provenance, local-time matching, and non-durable demo actions."""
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import sensor, pipeline
from hardware import sensor_daemon, uploader


def client_for_sensor(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMOSENSE_API_KEY", "test-key")
    monkeypatch.delenv("VERCEL", raising=False)
    app = FastAPI()
    app.include_router(sensor.router, prefix="/api/sensor")
    return TestClient(app), tmp_path / "sensor.db"


def test_sensor_requires_auth_and_discards_fake_source(tmp_path, monkeypatch):
    client, db_path = client_for_sensor(tmp_path, monkeypatch)
    with patch.object(sensor, "_DB_PATH", db_path):
        reading = {"timestamp": "2026-09-27T15:30:00Z", "temp_c": 25.0}
        assert client.post("/api/sensor/readings", json={"readings": [reading]}).status_code == 401
        reading["source"] = "simulated"
        response = client.post("/api/sensor/readings", headers={"X-API-Key": "test-key"}, json={"readings": [reading]})
        assert response.json()["accepted"] == 0
        assert response.json()["rejected"] == 1
        assert client.get("/api/sensor/latest").status_code == 404


def test_utc_1530_is_local_9pm_and_out_of_range_is_rejected(tmp_path, monkeypatch):
    client, db_path = client_for_sensor(tmp_path, monkeypatch)
    with patch.object(sensor, "_DB_PATH", db_path):
        readings = [
            {"timestamp": "2026-09-27T15:30:00Z", "temp_c": 25.0},
            {"timestamp": "2026-09-28T21:00:00Z", "temp_c": 25.0},
            {"timestamp": "2026-09-29T15:30:00Z", "temp_c": 125.0},
        ]
        result = client.post("/api/sensor/readings", headers={"X-API-Key": "test-key"}, json={"readings": readings})
        assert result.json()["accepted"] == 1
        assert result.json()["rejected"] == 1
        assert result.json()["acknowledged"] == 0
        assert client.get("/api/sensor/latest").json()["date"] == "2026-09-27"


def test_real_sensor_never_simulates_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(sensor_daemon, "SENSOR_AVAILABLE", False)
    assert sensor_daemon.read_dht22(4) is None


def test_uploader_keeps_batch_on_partial_aggregate_ack(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"accepted": 1}
    monkeypatch.setattr(uploader.requests, "post", lambda *args, **kwargs: Response())
    rows = [{"id": 1, "timestamp": "2026-09-27T15:30:00Z", "temp_c": 25}, {"id": 2, "timestamp": "2026-09-27T15:31:00Z", "temp_c": 26}]
    assert uploader.upload_readings_to_cloud("https://example.test", rows) == (0, [1, 2])
    class Complete(Response):
        def json(self): return {"accepted": 1, "acknowledged": 2, "rejected": 0}
    monkeypatch.setattr(uploader.requests, "post", lambda *args, **kwargs: Complete())
    assert uploader.upload_readings_to_cloud("https://example.test", rows) == (2, [])


def test_pipeline_mutation_requires_key_and_durable_worker(monkeypatch):
    app = FastAPI()
    app.include_router(pipeline.router, prefix="/api/pipeline")
    client = TestClient(app)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("THERMOSENSE_API_KEY", "test-key")
    assert client.post("/api/pipeline/daily").status_code == 401
    monkeypatch.setenv("VERCEL", "1")
    assert client.post("/api/pipeline/daily", headers={"X-API-Key": "test-key"}).status_code == 503


def test_sensor_mutation_unavailable_on_vercel(tmp_path, monkeypatch):
    client, db_path = client_for_sensor(tmp_path, monkeypatch)
    monkeypatch.setenv("VERCEL", "1")
    with patch.object(sensor, "_DB_PATH", db_path):
        result = client.post("/api/sensor/readings", headers={"X-API-Key": "test-key"}, json={"readings": [{"timestamp": "2026-09-27T15:30:00Z", "temp_c": 25}]})
        assert result.status_code == 503


def test_live_regional_forecast_is_labeled_and_not_a_fake_interval(monkeypatch):
    import pandas as pd
    from datetime import date, timedelta
    from src.api.main import app, load_config
    app.state.config = load_config()
    app.state.model_manager = None
    tomorrow = date.today() + timedelta(days=1)
    frame = pd.DataFrame([{"date": pd.Timestamp(tomorrow), "temp_c": 24.75}])
    monkeypatch.setattr("src.data.fetcher.fetch_forecast_open_meteo", lambda **kwargs: frame)
    resp = TestClient(app).get("/api/forecast?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_used"] == "open_meteo_regional"
    assert data["availability"] == "regional_forecast_only"
    assert data["forecasts"][0]["predicted_temp_c"] == 24.75
    assert data["forecasts"][0]["confidence"] == "none"
    assert data["forecasts"][0]["lower_bound_c"] is None


def test_pi_9pm_selection_maps_local_time_to_utc(tmp_path):
    from hardware import sensor_daemon as d
    db = tmp_path / "readings.db"
    d.init_database(db)
    d.store_reading({"timestamp": "2026-09-27T15:30:00Z", "temp_c": 27, "humidity_pct": 60, "source": "dht22_sensor"}, db)
    d.store_reading({"timestamp": "2026-09-27T21:00:00Z", "temp_c": 99, "humidity_pct": 60, "source": "simulated"}, db)
    reading = d.get_9pm_reading(db, "2026-09-27")
    assert reading["temp_c"] == 27


def test_training_refuses_short_or_api_only_history():
    import pandas as pd
    import pytest
    from scripts import train_models
    sample = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40), "temp_c": [25.0]*40, "is_sensor_reading": [False]*40})
    with patch.object(train_models, "load_processed", return_value=sample), patch.object(train_models, "build_feature_matrix", return_value=sample):
        with pytest.raises(ValueError, match="DHT22-tagged site observations"):
            train_models.prepare_data({"evaluation": {"test_split_days": 14}})


def test_ingested_real_pi_actual_enters_feature_source(tmp_path):
    import pandas as pd
    from datetime import date
    from src.data.baseline_collector import init_database, store_actual
    from src.data.preprocess import merge_collected_actuals
    db = tmp_path / "actuals.db"
    init_database(db)
    store_actual(db, date(2026, 9, 27), 28.5, 67, source="dht22_sensor")
    store_actual(db, date(2026, 9, 28), 99, 60, source="simulated")
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-09-27", "2026-09-28"]),
                          "temp_c": [24.0, 23.0], "temp_c_api": [24.0, 23.0],
                          "is_sensor_reading": [False, False], "humidity_pct": [60., 60.],
                          "observation_source": ["open_meteo_archive", "open_meteo_archive"]})
    output = merge_collected_actuals(frame, db)
    assert output.temp_c.tolist() == [28.5, 23.0]
    assert output.is_sensor_reading.tolist() == [True, False]
    assert output.humidity_pct.tolist() == [67., 60.]
    assert output.api_bias.iloc[0] == 4.5
    assert output.observation_source.tolist() == ["dht22_sensor", "open_meteo_archive"]


def test_feedback_requires_key_even_locally(monkeypatch):
    from src.api.main import app
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("THERMOSENSE_API_KEY", "test-key")
    result = TestClient(app).post("/api/forecast/feedback", json={"date": "2024-07-01", "actual_temp_c": 25})
    assert result.status_code == 401


def test_contaminated_model_metrics_never_reach_public_api():
    from src.api.main import app, load_config
    app.state.config = load_config()
    app.state.model_manager = None
    client = TestClient(app)
    metrics = client.get("/api/metrics").json()
    assert metrics["models"] == {}
    assert metrics["n_observations"] == 0
    status = client.get("/api/pipeline/status").json()
    assert "ensemble" not in (status["last_training_results"] or {})
    assert "lgbm" not in (status["last_training_results"] or {})


def test_training_route_refuses_even_with_key(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("THERMOSENSE_API_KEY", "test-key")
    from src.api.main import app
    result = TestClient(app).post("/api/pipeline/train", headers={"X-API-Key": "test-key"}, json={"models": ["sarima"]})
    assert result.status_code == 503


def test_pi_http_unsynced_respects_requested_batch_size(tmp_path):
    import threading
    from http.server import HTTPServer
    import requests
    from hardware import sensor_daemon as d
    db = tmp_path / "readings.db"
    d.init_database(db)
    for i in range(3):
        d.store_reading({"timestamp": f"2026-09-27T15:3{i}:00Z", "temp_c": 27, "humidity_pct": 60, "source": "dht22_sensor"}, db)
    d.SensorHTTPHandler.db_path = db
    server = HTTPServer(("127.0.0.1", 0), d.SensorHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = requests.get(f"http://127.0.0.1:{server.server_port}/unsynced", params={"limit": 2}, timeout=3)
        assert response.json()["count"] == 2
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_archive_values_do_not_masquerade_as_local_bias():
    import pandas as pd
    from src.features.engineer import add_api_bias_feature
    frame = pd.DataFrame({"temp_c": [27., 25.], "temp_c_api": [25., 25.],
                          "observation_source": ["dht22_sensor", "open_meteo_archive"]})
    result = add_api_bias_feature(frame)
    assert result.api_bias.iloc[0] == 2
    assert pd.isna(result.api_bias.iloc[1])
