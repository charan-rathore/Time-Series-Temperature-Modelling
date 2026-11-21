"""An ensemble must never replace a failed required prediction with a mean."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models import loader
from src.models.ensemble import EnsembleStacker


def _raises(**kwargs):
    raise RuntimeError("required model unavailable at prediction time")


@pytest.fixture
def manager(monkeypatch):
    features = pd.DataFrame({"temp_c": [10.0, 20.0, 30.0]})
    monkeypatch.setattr(loader, "load_processed", lambda: features)
    monkeypatch.setattr(loader, "build_feature_matrix", lambda *a, **kw: features)
    m = loader.ModelManager()
    m.is_loaded = True
    m.sarima = SimpleNamespace(
        is_fitted=True,
        predict=lambda **kw: np.full(kw["steps"], 21.0),
        predict_intervals=lambda **kw: {
            "median": np.full(kw["steps"], 21.0),
            "lower": np.full(kw["steps"], 20.0),
            "upper": np.full(kw["steps"], 22.0),
        },
    )
    m.lgbm_models = {
        h: SimpleNamespace(predict=lambda **kw: np.array([22.0]))
        for h in (1, 2, 3)
    }
    m.tft = SimpleNamespace(
        is_fitted=True,
        predict_intervals=lambda **kw: {
            "median": np.full(kw["steps"], 23.0),
            "lower": np.full(kw["steps"], 22.0),
            "upper": np.full(kw["steps"], 24.0),
        },
    )
    m.ensemble = EnsembleStacker({"base_models": ["sarima", "lgbm", "tft"]})
    # Real fitted Ridge models, not a mock stacker that accepts any input shape.
    x = np.array([[10., 11., 12.], [20., 22., 24.], [30., 33., 36.], [40., 41., 42.]])
    m.ensemble.meta_models = {
        h: Ridge().fit(x, np.array([15., 25., 35., 45.]) + h)
        for h in (1, 2, 3)
    }
    m.ensemble.is_fitted = True
    m._results = {"ensemble": {"day1": {"rmse": 0.1, "n_observations": 30}}}
    assert m.get_best_model_name() == "ensemble"
    return m


@pytest.mark.parametrize("failed", ["sarima", "lgbm", "tft"])
def test_failed_required_base_never_serves_mean_fed_ensemble(manager, failed):
    if failed == "sarima":
        manager.sarima.predict = _raises
    elif failed == "lgbm":
        for model in manager.lgbm_models.values():
            model.predict = _raises
    else:
        manager.tft.predict_intervals = _raises
    results = manager.forecast(days=3)
    assert all(row["model_used"] != "ensemble" for row in results)
    expected = "lgbm" if failed == "tft" else "tft"
    assert [row["model_used"] for row in results] == [expected] * 3
    assert [row["predicted_temp_c"] for row in results] == [22.0 if failed == "tft" else 23.0] * 3


def test_missing_horizon_only_falls_back_for_that_horizon(manager):
    del manager.lgbm_models[2]
    results = manager.forecast(days=3)
    assert [row["model_used"] for row in results] == ["ensemble", "tft", "ensemble"]
    assert results[1]["predicted_temp_c"] == 23.0


def test_healthy_required_bases_keep_the_real_stacker_result(manager):
    results = manager.forecast(days=3)
    expected = manager.ensemble.predict(
        steps=3, base_predictions={"sarima": np.array([21.]), "lgbm": np.array([22.]), "tft": np.array([23.])}
    )
    assert [row["model_used"] for row in results] == ["ensemble"] * 3
    assert [row["predicted_temp_c"] for row in results] == [round(float(v), 2) for v in expected]


def test_failure_of_unused_base_does_not_disable_ensemble(manager):
    manager.ensemble.base_model_names = ["sarima"]
    manager.ensemble.meta_models = {
        h: Ridge().fit(np.array([[10.], [20.], [30.]]), np.array([15., 25., 35.]) + h)
        for h in (1, 2, 3)
    }
    manager.tft.predict_intervals = _raises
    for model in manager.lgbm_models.values():
        model.predict = _raises
    assert [row["model_used"] for row in manager.forecast(days=3)] == ["ensemble"] * 3


def test_temporary_required_failure_does_not_poison_later_horizons(manager):
    def sarima_predict(**kwargs):
        if kwargs["steps"] == 2:
            return _raises()
        return np.full(kwargs["steps"], 21.0)
    manager.sarima.predict = sarima_predict
    assert [row["model_used"] for row in manager.forecast(days=3)] == ["ensemble", "tft", "ensemble"]


def test_all_runtime_models_fail_uses_existing_labelled_fallback(manager):
    manager.sarima.predict = _raises
    manager.sarima.predict_intervals = _raises
    manager.tft.predict_intervals = _raises
    for model in manager.lgbm_models.values():
        model.predict = _raises
    results = manager.forecast(days=3)
    assert [row["model_used"] for row in results] == ["fallback"] * 3
    assert [row["predicted_temp_c"] for row in results] == [30.0] * 3


@pytest.mark.parametrize("path", ["/api/forecast", "/forecast"])
def test_api_uses_actual_per_horizon_fallback_identity(manager, monkeypatch, path):
    from datetime import date
    from fastapi.testclient import TestClient
    from src.api.main import app, load_config
    del manager.lgbm_models[2]
    monkeypatch.setattr(manager, "data_through", lambda: date.today())
    monkeypatch.setattr(app.state, "config", load_config(), raising=False)
    monkeypatch.setattr(app.state, "model_manager", manager, raising=False)
    response = TestClient(app).get(path + "?days=3")
    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "ensemble"
    assert [row["model_used"] for row in body["forecasts"]] == ["ensemble", "tft", "ensemble"]
    assert body["forecasts"][1]["predicted_temp_c"] == 23.0
    assert body["forecasts"][1]["synthetic"] is False
