"""Reject invalid model choices before launching expensive background training."""
from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes import pipeline

@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(pipeline.router, prefix="/api/pipeline")
    thread = Mock()
    monkeypatch.setattr(pipeline.threading, "Thread", thread)
    monkeypatch.setattr(pipeline, "_job_store", {})
    return TestClient(app), thread

@pytest.mark.parametrize("models", [["sarima", "typo"], ["lgbm", "all"], ["typo"], []])
def test_invalid_selection_launches_no_job(client, models):
    http, thread = client
    result = http.post("/api/pipeline/train", json={"models": models})
    assert result.status_code == 400
    thread.assert_not_called()

@pytest.mark.parametrize("models", [["sarima"], ["tft", "ensemble", "lgbm", "sarima"]])
def test_valid_selection_retains_order(client, models):
    http, thread = client
    result = http.post("/api/pipeline/train", json={"models": models, "skip_mlflow": True})
    assert result.status_code == 200
    assert thread.call_args.kwargs["args"][1] == [pipeline.sys.executable, "scripts/train_models.py", "--models", *models, "--no-mlflow"]
    thread.return_value.start.assert_called_once()

def test_default_selection_still_starts(client):
    http, thread = client
    result = http.post("/api/pipeline/train", json={})
    assert result.status_code == 200
    assert thread.call_args.kwargs["args"][1][-4:] == ["sarima", "lgbm", "tft", "ensemble"]


def test_running_job_still_rejects_another_launch(client, monkeypatch):
    http, thread = client
    monkeypatch.setattr(pipeline, "_job_store", {"train": {"status": "running"}})
    assert http.post("/api/pipeline/train", json={"models": ["sarima"]}).status_code == 400
    thread.assert_not_called()
