"""The ensemble is only selectable when every base model its stacker was
trained on actually loaded in this runtime (Vercel has no lightgbm)."""
from types import SimpleNamespace

from src.models.loader import ModelManager


def _manager(lgbm_loaded: bool) -> ModelManager:
    m = ModelManager.__new__(ModelManager)
    m.ensemble = SimpleNamespace(is_fitted=True, base_model_names=["sarima", "lgbm"])
    m.sarima = SimpleNamespace(is_fitted=True)
    m.lgbm_models = {1: object()} if lgbm_loaded else {}
    m.tft = None
    m._results = {
        "ensemble": {"day1": {"rmse": 0.5, "n_observations": 30}},
        "sarima": {"day1": {"rmse": 0.9, "n_observations": 30}},
    }
    m.is_loaded = True
    return m


def test_ensemble_selectable_when_all_bases_loaded():
    assert _manager(lgbm_loaded=True).get_best_model_name() == "ensemble"


def test_ensemble_not_selectable_when_lgbm_missing():
    # The Vercel runtime has no lightgbm: the lgbm artifacts never load,
    # so the pick must fall through rather than serve a mean-fed stacker.
    assert _manager(lgbm_loaded=False).get_best_model_name() == "sarima"
