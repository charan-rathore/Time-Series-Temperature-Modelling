"""Saved training metrics must match a fresh evaluation of the committed
model artifacts on the same held-out window.

Contract: models/results.json must describe the committed models/*.pkl
evaluated on the committed data. Retraining artifacts (or changing the
data) without regenerating results.json - or vice versa - fails this test.

Pairing: a horizon-h model labels feature row t with the temperature at
t + h (LGBMForecastModel.fit shifts the target by -self.horizon), so a
prediction from row t is compared with the actual at t + h and lgbm
day-h scores have len(test) - h pairs. SARIMA forecasts from the end of
train+val, so its step-i prediction already pairs with test row i.

The ensemble's published scores come from the training-time holdout
evaluation (which refits its base models), so they cannot be reproduced
from the committed ensemble.pkl alone; this test re-runs that code path
with artifact saves redirected. A known training-side issue - the
ensemble holdout refits LightGBM on train+val while the served stack
uses the lgbm fitted on train only - belongs to the train_models.py
evaluation fix on the publisher's branch; this test follows whatever
train_ensemble produces until then.
"""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _config_and_split():
    from scripts.train_models import prepare_data

    config = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    train_df, val_df, test_df = prepare_data(config)
    return config, train_df, val_df, test_df


def _saved():
    return json.load(open(ROOT / "models" / "results.json"))


def _assert_metrics_equal(saved, fresh, label):
    assert set(fresh) == set(saved), f"{label}: field mismatch {set(fresh) ^ set(saved)}"
    for key, fresh_val in fresh.items():
        saved_val = saved[key]
        if fresh_val is None or saved_val is None:
            assert saved_val == fresh_val, f"{label}.{key}"
        elif key == "n_observations" or isinstance(fresh_val, str):
            assert saved_val == fresh_val, f"{label}.{key}"
        else:
            assert fresh_val == pytest.approx(saved_val, abs=1e-6), f"{label}.{key}"


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_saved_lgbm_metrics_match_committed_artifact(horizon):
    from src.evaluation.metrics import evaluate_all
    from src.models.lgbm_model import LGBMForecastModel

    config, _, _, test_df = _config_and_split()
    model = LGBMForecastModel(config["models"]["lgbm"], horizon=horizon)
    model.load(str(ROOT / "models" / f"lgbm_h{horizon}.pkl"))
    preds = model.predict(steps=len(test_df), future_df=test_df)
    # Row t's prediction targets the actual at t + horizon (see module
    # docstring); drop the tail rows whose target lies beyond the window.
    n = min(len(test_df), len(preds)) - horizon
    assert n > 0, "holdout window too small for this horizon"
    actuals = test_df["temp_c"].values[horizon:horizon + n]
    fresh = evaluate_all(actuals, preds[:n])
    _assert_metrics_equal(_saved()["lgbm"][f"day{horizon}"], fresh, f"lgbm.day{horizon}")


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_saved_sarima_metrics_match_committed_artifact(horizon):
    from src.evaluation.metrics import evaluate_all
    from src.models.sarima_model import SARIMAXModel

    config, _, _, test_df = _config_and_split()
    model = SARIMAXModel(config["models"]["sarima"])
    model.load(str(ROOT / "models" / "sarima.pkl"))
    future_exog = test_df.iloc[:horizon]
    preds = model.predict(steps=horizon, future_df=future_exog)
    actuals = test_df["temp_c"].values[:horizon]
    intervals = model.predict_intervals(steps=horizon, future_df=future_exog)
    fresh = evaluate_all(actuals, preds, intervals["lower"], intervals["upper"])
    _assert_metrics_equal(_saved()["sarima"][f"day{horizon}"], fresh, f"sarima.day{horizon}")


def test_saved_ensemble_scores_match_training_evaluation(tmp_path, monkeypatch):
    """Re-run the training-time ensemble evaluation (same code path,
    artifact saves redirected) and require the committed ensemble metrics
    to match - stale results.json or ensemble.pkl fails."""
    import scripts.train_models as tm

    config, train_df, val_df, test_df = _config_and_split()
    monkeypatch.setattr(tm, "MODELS_DIR", tmp_path)
    fresh = tm.train_ensemble(
        train_df, val_df, test_df, config,
        sarima_results={}, lgbm_results={}, tft_results={},
    )
    saved = _saved()["ensemble"]
    for day, metrics in fresh.items():
        _assert_metrics_equal(saved[day], metrics, f"ensemble.{day}")
