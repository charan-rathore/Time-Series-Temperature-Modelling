"""Training-script follow-ups from the Codex reviews of #22 and #24."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tm(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "train_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "MODELS_DIR", tmp_path)
    return mod


def _frame(start, n):
    idx = range(start, start + n)
    return pd.DataFrame({"temp_c": np.linspace(10, 20, n) + start * 0.01}, index=idx)


class FakeArima:
    """Stands in for the pmdarima model: persistence forecast, updated with actuals."""

    def __init__(self):
        self.last = 0.0

    def predict(self, n_periods, X=None, return_conf_int=False, alpha=0.2, **kw):
        fc = np.full(n_periods, self.last)
        if return_conf_int:
            return fc, np.column_stack([fc - 1, fc + 1])
        return fc

    def update(self, y, X=None):
        self.last = float(y[-1])


def _fake_models(tm, monkeypatch, calls):
    class Sarima:
        def __init__(self, cfg):
            self.model = FakeArima()

        def fit(self, df, **kw):
            calls.append(("sarima", set(df.index), None))
            self.model.last = float(df["temp_c"].iloc[-1])

        def _get_exog(self, df):
            return None

        def predict(self, steps, future_df=None):
            return np.zeros(steps)

        def predict_intervals(self, steps, future_df=None):
            z = np.zeros(steps)
            return {"lower": z - 1, "upper": z + 1}

        def save(self, path):
            pass

    class Lgbm:
        def __init__(self, cfg, horizon=1):
            self.horizon = horizon

        def fit(self, df, val_df=None, **kw):
            calls.append(("lgbm", set(df.index), None if val_df is None else set(val_df.index)))

        def predict(self, steps, future_df=None):
            # row t predicts the temperature of day t + horizon, perfectly
            t = future_df["temp_c"].values
            out = np.full(steps, np.nan)
            out[: len(t) - self.horizon] = t[self.horizon:]
            return out[:steps]

        def feature_importance(self):
            return pd.Series({"a": 1.0})

        def save(self, path):
            pass

    class Stacker:
        def __init__(self, cfg):
            self.meta_models = {}

        def fit_from_oof(self, preds, actuals, horizons):
            self.meta_models = {h: type("M", (), {"predict": staticmethod(lambda X: X.mean(axis=1))})() for h in horizons}

        def save(self, path):
            pass

    monkeypatch.setattr(tm, "SARIMAXModel", Sarima)
    monkeypatch.setattr(tm, "LGBMForecastModel", Lgbm)
    monkeypatch.setattr(tm, "EnsembleStacker", Stacker)


CFG = {"models": {"sarima": {}, "lgbm": {}, "ensemble": {}}}


def _splits():
    return _frame(0, 60), _frame(1000, 20), _frame(2000, 15)


def test_stacker_validation_predictions_do_not_early_stop_on_validation_labels(tm, monkeypatch):
    calls = []
    _fake_models(tm, monkeypatch, calls)
    train_df, val_df, test_df = _splits()
    tm.train_ensemble(train_df, val_df, test_df, CFG, {}, {}, {})
    lgbm = [c for c in calls if c[0] == "lgbm"]
    val_pass = lgbm[:3]  # the stacker's validation-period predictions
    for _, fit_idx, es_idx in val_pass:
        assert es_idx is None or not (es_idx & set(val_df.index))
        assert not (fit_idx & set(val_df.index))


def test_ensemble_test_metrics_use_the_served_base_models(tm, monkeypatch):
    calls = []
    _fake_models(tm, monkeypatch, calls)
    train_df, val_df, test_df = _splits()
    tm.train_ensemble(train_df, val_df, test_df, CFG, {}, {}, {})
    train_idx, val_idx = set(train_df.index), set(val_df.index)
    # served LightGBM (train_lgbm): fit on train, early stop on val
    served_lgbm = [c for c in calls if c[0] == "lgbm"][-3:]
    for _, fit_idx, es_idx in served_lgbm:
        assert fit_idx == train_idx and es_idx == val_idx
    # served SARIMA (train_sarima): fit on train + val
    # (the concat resets the index, so compare sizes)
    assert len([c for c in calls if c[0] == "sarima"][-1][1]) == len(train_df) + len(val_df)


def test_sarima_is_scored_on_every_held_out_day(tm, monkeypatch):
    calls = []
    _fake_models(tm, monkeypatch, calls)
    train_df, val_df, test_df = _splits()
    res = tm.train_sarima(train_df, val_df, test_df, CFG)
    n = len(test_df)
    # horizon h pairs the forecast for day d with the actual of day d, d = h..n-1
    assert res["day1"]["n_observations"] == n - 1
    assert res["day2"]["n_observations"] == n - 2
    assert res["day3"]["n_observations"] == n - 3


def test_rolling_sarima_forecast_uses_only_earlier_actuals(tm):
    class M:
        model = FakeArima()

        def _get_exog(self, df):
            return None

    m = M()
    m.model.last = 5.0
    test_df = pd.DataFrame({"temp_c": [1.0, 2.0, 3.0, 4.0]})
    out = tm.rolling_sarima_forecasts(m, test_df, [1, 2])
    pred1 = out[1][0]
    # persistence: target day d is forecast from the actuals up to day d - h
    assert np.isnan(pred1[0]) and list(pred1[1:]) == [1.0, 2.0, 3.0]
    pred2 = out[2][0]
    assert np.isnan(pred2[:2]).all() and list(pred2[2:]) == [1.0, 2.0]
    assert m.model.last == 5.0  # the saved model is untouched


def test_lgbm_pairs_each_forecast_with_the_actual_of_its_target_day(tm, monkeypatch):
    calls = []
    _fake_models(tm, monkeypatch, calls)
    train_df, val_df, test_df = _splits()
    res = tm.train_lgbm(train_df, val_df, test_df, CFG)
    n = len(test_df)
    for h in (1, 2, 3):
        # the fake forecasts the next-h-day temperature exactly, so a correct pairing scores 0
        assert res[f"day{h}"]["n_observations"] == n - h
        assert res[f"day{h}"]["rmse"] == pytest.approx(0.0)


def test_ensemble_scores_each_horizon_on_n_minus_h_days(tm, monkeypatch):
    calls = []
    _fake_models(tm, monkeypatch, calls)
    train_df, val_df, test_df = _splits()
    res = tm.train_ensemble(train_df, val_df, test_df, CFG, {}, {}, {})
    n = len(test_df)
    for h in (1, 2, 3):
        assert res[f"day{h}"]["n_observations"] == n - h


def test_origin_available_exog_never_reads_after_origin(tm):
    """The helper must return the last h observed exog rows through origin t,
    padding by repeating the earliest row when history is short - never any
    row after t."""
    exog = np.arange(50, dtype=float).reshape(25, 2)
    h = 3
    block = tm._origin_available_exog(exog, t=7, h=h)
    assert block.shape == (h, 2)
    np.testing.assert_array_equal(block, exog[5:8])  # rows t-h+1..t

    short = tm._origin_available_exog(exog, t=1, h=h)  # only rows 0..1 observed
    assert short.shape == (h, 2)
    np.testing.assert_array_equal(short[0], exog[0])  # padded with earliest row
    np.testing.assert_array_equal(short[1:], exog[0:2])

    poisoned = exog.copy()
    poisoned[8:] = -999.0  # anything after the origin must be irrelevant
    np.testing.assert_array_equal(tm._origin_available_exog(poisoned, t=7, h=h), exog[5:8])

    assert tm._origin_available_exog(None, t=7, h=h) is None


def test_rolling_sarima_forecasts_ignores_future_exogenous(tm, monkeypatch):
    """The training evaluation's SARIMA path must not read observed future
    exog either: poisoning exog after origin 0 cannot move the forecast
    issued at origin 0. Fit is made exog-honest by a test-only shim."""
    pmdarima = pytest.importorskip("pmdarima")
    import src.models.sarima_model as sm

    original = sm.pm.auto_arima
    monkeypatch.setattr(
        sm.pm, "auto_arima",
        lambda y, exogenous=None, **kw: original(y, X=exogenous, **kw),
    )

    rng = np.random.default_rng(3)
    days = 230
    hum = 60 + 10 * rng.normal(0, 1, days)
    prs = 1013 + 3 * rng.normal(0, 1, days)
    temp = 0.4 * hum - 0.1 * prs + 120.0 + rng.normal(0, 0.5, days)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "temp_c": temp, "humidity_pct": hum, "pressure_hpa": prs,
    })
    config = {"seasonal": True, "m": 7, "information_criterion": "aic", "stepwise": True}
    model = sm.SARIMAXModel(config)
    model.fit(df.iloc[:200])
    assert model.model.arima_res_.model.exog is not None  # shim really flowed

    test_df = df.iloc[200:]
    h = 2
    base = tm.rolling_sarima_forecasts(model, test_df, [h])[h][0][h]  # origin 0 -> row h
    poisoned = test_df.copy()
    poisoned.loc[poisoned.index[1]:, "humidity_pct"] += 50.0
    poisoned.loc[poisoned.index[1]:, "pressure_hpa"] -= 60.0
    alt = tm.rolling_sarima_forecasts(model, poisoned, [h])[h][0][h]
    assert base == pytest.approx(alt, abs=1e-9)
