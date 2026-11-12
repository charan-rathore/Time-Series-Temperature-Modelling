"""Input validation for the pipeline router: dates, tail, limit."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.routes import pipeline


@pytest.fixture(autouse=True)
def clean_state():
    with pipeline._job_lock:
        pipeline._job_store.clear()
    with pipeline._log_lock:
        pipeline._log_lines.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_backfill_rejects_malformed_dates(client):
    response = client.post("/api/pipeline/backfill", json={"start_date": "13-Oct"})
    assert response.status_code == 422


def test_backfill_rejects_impossible_dates(client):
    response = client.post("/api/pipeline/backfill", json={"start_date": "2026-02-30"})
    assert response.status_code == 422


def test_backfill_rejects_inverted_range(client):
    response = client.post(
        "/api/pipeline/backfill",
        json={"start_date": "2026-10-10", "end_date": "2026-10-01"},
    )
    assert response.status_code == 400
    with pipeline._job_lock:
        assert "backfill" not in pipeline._job_store


def test_logs_tail_zero_no_longer_returns_everything(client):
    for i in range(5):
        pipeline._append_log(f"line-{i}")
    assert client.get("/api/pipeline/logs?tail=0").status_code == 422
    assert client.get("/api/pipeline/logs?tail=-3").status_code == 422


def test_logs_tail_is_capped(client):
    assert client.get(f"/api/pipeline/logs?tail={pipeline._LOG_MAX + 1}").status_code == 422
    assert client.get(f"/api/pipeline/logs?tail={pipeline._LOG_MAX}").status_code == 200


def test_mlflow_limit_is_bounded(client):
    assert client.get("/api/pipeline/mlflow?limit=0").status_code == 422
    assert client.get("/api/pipeline/mlflow?limit=101").status_code == 422
    assert client.get("/api/pipeline/mlflow?limit=20").status_code == 200


def test_mlflow_endpoint_reads_existing_file_store(client, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from src.api.routes import pipeline

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("thermosense")
    with mlflow.start_run(run_name="r1"):
        mlflow.log_param("model", "lgbm")
        mlflow.log_metric("rmse", 1.23)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE")
    monkeypatch.setattr(pipeline, "_PROJECT_ROOT", tmp_path)

    body = client.get("/api/pipeline/mlflow").json()
    assert body["total_runs"] == 1
    assert body["runs"][0]["model"] == "lgbm"
