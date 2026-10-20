"""Saved training metrics must match a fresh evaluation of the committed
model artifacts on the same held-out window.

If models/*.pkl are retrained (or the data changes) without regenerating
models/results.json - or vice versa - the metrics the site shows no longer
describe the model it serves, and this test fails.
"""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _fresh_lgbm_day1():
    from scripts.train_models import prepare_data
    from src.evaluation.metrics import evaluate_all
    from src.models.lgbm_model import LGBMForecastModel

    config = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    _, _, test_df = prepare_data(config)
    model = LGBMForecastModel(config["models"]["lgbm"], horizon=1)
    model.load(str(ROOT / "models" / "lgbm_h1.pkl"))
    preds = model.predict(steps=len(test_df), future_df=test_df)
    n = min(len(test_df), len(preds))
    return evaluate_all(test_df["temp_c"].values[:n], preds[:n])


def test_saved_lgbm_day1_matches_committed_artifact():
    saved = json.load(open(ROOT / "models" / "results.json"))["lgbm"]["day1"]
    fresh = _fresh_lgbm_day1()
    assert fresh["rmse"] == pytest.approx(saved["rmse"], abs=1e-6)
    assert fresh["mae"] == pytest.approx(saved["mae"], abs=1e-6)
    assert saved["n_observations"] == fresh["n_observations"]
