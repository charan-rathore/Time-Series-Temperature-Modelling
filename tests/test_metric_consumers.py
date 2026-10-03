"""Regression checks for the real logging function and metrics HTTP boundary."""
import ast
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.evaluation.metrics import evaluate_all
from src.api.routes import metrics as route

ROOT = Path(__file__).resolve().parents[1]

class NumericMlflow:
    def __init__(self):
        self.metrics = {}
        self.tags = {}
    def set_tracking_uri(self, value): pass
    def set_experiment(self, value): pass
    def start_run(self, **kwargs): return nullcontext()
    def log_param(self, name, value): pass
    def set_tag(self, name, value): self.tags[name] = value
    def log_artifact(self, path): pass
    def log_metric(self, name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError("MLflow requires a finite numeric metric")
        self.metrics[name] = value

def actual_logger(mlflow):
    # Execute the unchanged real function body without optional model imports.
    tree = ast.parse((ROOT / "scripts/train_models.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "log_to_mlflow")
    ns = dict(_MLFLOW_AVAILABLE=True, mlflow=mlflow, _PROJECT_ROOT=ROOT, MODELS_DIR=Path("/tmp/no-model-artifacts"), datetime=datetime, np=np)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual-log-to-mlflow", "exec"), ns)
    return ns["log_to_mlflow"]

def test_real_logger_separates_numeric_metrics_from_status_metadata():
    mlflow = NumericMlflow()
    actual_logger(mlflow)({"sarima": {"day1": evaluate_all(np.array([20.]), np.array([30.]))}}, {"models": {}})
    assert mlflow.metrics["day1_mae"] == 10
    assert mlflow.metrics["day1_n_observations"] == 1
    assert "day1_skill_score" not in mlflow.metrics
    assert mlflow.tags["day1_skill_score_status"] == "undefined_zero_baseline"

def test_real_logger_rejects_nonfinite_metric_values_without_failing_run():
    mlflow = NumericMlflow()
    actual_logger(mlflow)({"sarima": {"day1": {"rmse": float("nan"), "mae": 2., "flag": True}}}, {"models": {}})
    assert mlflow.metrics == {"day1_mae": 2.}

def api_result(monkeypatch, results, disk=False, tmp_path=None):
    app = FastAPI()
    app.state.config = {"location": {"name": "Test site"}}
    if disk:
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        monkeypatch.setattr(route, "RESULTS_PATH", path)
        app.state.model_manager = None
    else:
        app.state.model_manager = SimpleNamespace(get_results=lambda: results)
    app.include_router(route.router, prefix="/metrics")
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    return response.json()

def test_committed_legacy_scores_are_not_relabelled_as_valid(monkeypatch):
    result = api_result(monkeypatch, json.loads((ROOT / "models/results.json").read_text()))
    metric = result["models"]["sarima"]["day1"]
    assert metric["skill_score"] is None
    assert metric["skill_score_status"] == "unverified_legacy"
    assert metric["n_observations"] == 0

def test_disk_legacy_score_is_withheld(monkeypatch, tmp_path):
    result = api_result(monkeypatch, {"sarima": {"day1": {"skill_score": 1.0}}}, True, tmp_path)
    assert result["models"]["sarima"]["day1"]["skill_score"] is None

def test_missing_count_or_status_does_not_establish_availability(monkeypatch):
    for metadata in [{"skill_score_status": "available"}, {"n_observations": 3}, {"skill_score_status": "available", "n_observations": 0}]:
        result = api_result(monkeypatch, {"sarima": {"day1": {"skill_score": 1., **metadata}}})
        assert result["models"]["sarima"]["day1"]["skill_score"] is None

def test_new_valid_and_undefined_scores_keep_their_explicit_status(monkeypatch):
    normal = evaluate_all(np.array([10., 20.]), np.array([11., 19.]))
    undefined = evaluate_all(np.array([20.]), np.array([30.]))
    result = api_result(monkeypatch, {"sarima": {"day1": normal, "day2": undefined}})
    assert result["models"]["sarima"]["day1"]["skill_score"] == normal["skill_score"]
    assert result["models"]["sarima"]["day1"]["skill_score_status"] == "available"
    assert result["models"]["sarima"]["day2"]["skill_score_status"] == "undefined_zero_baseline"
    assert result["n_observations"] == 2

def test_invalid_status_or_count_never_makes_skill_available(monkeypatch):
    for metadata in [
        {"skill_score_status": "available", "n_observations": True},
        {"skill_score_status": "available", "n_observations": -1},
        {"skill_score_status": "other", "n_observations": 2},
    ]:
        result = api_result(monkeypatch, {"sarima": {"day1": {"skill_score": 1., **metadata}}})
        assert result["models"]["sarima"]["day1"]["skill_score"] is None

def test_nonfinite_skill_is_withheld_with_valid_metadata(monkeypatch):
    result = api_result(monkeypatch, {"sarima": {"day1": {"skill_score": float("nan"), "skill_score_status": "available", "n_observations": 3}}})
    assert result["models"]["sarima"]["day1"]["skill_score"] is None
    assert result["models"]["sarima"]["day1"]["skill_score_status"] == "unavailable"
