"""Actual serving must select the trained meta-model for each forecast day."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.models import loader
from src.models.ensemble import EnsembleStacker


def test_forecast_uses_each_horizons_meta_model(monkeypatch):
    features = pd.DataFrame({"temp_c": [20., 21., 22.]})
    monkeypatch.setattr(loader, "load_processed", lambda: features)
    monkeypatch.setattr(loader, "build_feature_matrix", lambda *a, **kw: features)
    manager = loader.ModelManager()
    manager.is_loaded = True
    manager.sarima = SimpleNamespace(is_fitted=True, predict=lambda **kw: np.array([20.] * kw["steps"]))
    manager.ensemble = EnsembleStacker({"base_models": ["sarima"]})
    manager.ensemble.is_fitted = True
    manager.ensemble.meta_models = {h: SimpleNamespace(predict=lambda x, value=20.+h: np.array([value])) for h in (1, 2, 3)}
    results = manager.forecast(days=3)
    assert [r["predicted_temp_c"] for r in results] == [21., 22., 23.]


def test_default_multi_horizon_and_explicit_selection():
    stacker = EnsembleStacker({"base_models": ["sarima"]})
    stacker.is_fitted = True
    stacker.meta_models = {h: SimpleNamespace(predict=lambda x, value=h: np.array([value])) for h in (1, 2, 3)}
    inputs = {"sarima": np.array([20.])}
    np.testing.assert_array_equal(stacker.predict(steps=3, base_predictions=inputs), [1., 2., 3.])
    np.testing.assert_array_equal(stacker.predict(steps=1, base_predictions=inputs, start_horizon=2), [2.])
