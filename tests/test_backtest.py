"""Rolling-origin backtest: no leakage, correct pairing, honest skill."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    BASELINES, climatology, persistence, rolling_origin_folds,
    run_backtest, sarima_forecaster, seasonal_naive_7,
)


def _series(values, start="2024-01-01"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(values)),
                         "temp_c": np.asarray(values, dtype=float)})


def test_folds_tile_the_end_and_never_leak():
    folds = rolling_origin_folds(n_rows=400, n_folds=4, fold_len=30, min_train=100)
    assert len(folds) == 4
    assert folds[-1][2] == 400                       # newest data is tested
    for (train_end, t0, t1), nxt in zip(folds, folds[1:] + [None]):
        assert train_end == t0                        # train stops where test starts
        assert t1 - t0 == 30
        if nxt:
            assert nxt[1] == t1                       # folds are contiguous, no overlap
    assert [f[0] for f in folds] == sorted(f[0] for f in folds)  # expanding window


def test_too_short_series_is_refused():
    with pytest.raises(ValueError):
        rolling_origin_folds(n_rows=150, n_folds=4, fold_len=30, min_train=100)


def test_persistence_on_a_linear_ramp_misses_by_h_slopes():
    df = _series(np.arange(300) * 0.1)
    out = run_backtest(df, {"persistence": persistence}, horizons=(1, 3),
                       n_folds=3, fold_len=30, min_train=100)
    assert out["models"]["persistence"]["day1"]["rmse"] == pytest.approx(0.1)
    assert out["models"]["persistence"]["day3"]["rmse"] == pytest.approx(0.3)


def test_seasonal_naive_is_exact_on_a_weekly_cycle():
    df = _series(np.tile([20, 21, 23, 25, 24, 22, 20], 60))
    fold = df.iloc[300:330]
    p = seasonal_naive_7(df.iloc[:300], fold, 2)
    np.testing.assert_allclose(p, fold["temp_c"].to_numpy()[2:])


def test_seasonal_naive_uses_only_already_observed_days():
    # The forecast for the target at fold row i + h comes from row i + h - 7,
    # which must be <= i (observed at the origin) for every h <= 7.
    df = _series(np.arange(100))
    fold = df.iloc[60:90]
    p = seasonal_naive_7(df.iloc[:60], fold, 7)
    # h = 7: the forecast for row i + 7 is the value at row i, the origin itself.
    np.testing.assert_allclose(p, fold["temp_c"].to_numpy()[: len(fold) - 7])


def test_climatology_never_sees_the_test_window():
    df = _series(np.concatenate([np.full(200, 10.0), np.full(30, 99.0)]))
    p = climatology(df.iloc[:200], df.iloc[200:230], 1)
    assert np.all(p == 10.0)                          # the 99s are in the future


def test_wrong_length_forecaster_is_rejected():
    df = _series(np.arange(300) * 0.1)
    with pytest.raises(ValueError):
        run_backtest(df, {"bad": lambda tr, fo, h: np.zeros(3)}, n_folds=2, fold_len=30, min_train=100)


def test_skill_is_relative_to_baselines_and_json_safe():
    import json
    rng = np.random.default_rng(0)
    df = _series(25 + 3 * np.sin(np.arange(400) * 2 * np.pi / 7) + rng.normal(0, .3, 400))
    perfect_weekly = BASELINES["seasonal_naive_7"]
    out = run_backtest(df, dict(BASELINES), n_folds=4, fold_len=30, min_train=100)
    m = out["models"]
    assert m["seasonal_naive_7"]["day1"]["rmse"] < m["persistence"]["day1"]["rmse"]
    assert m["seasonal_naive_7"]["day1"]["skill_vs_persistence"] > 0
    assert m["persistence"]["day1"]["skill_vs_persistence"] == pytest.approx(0.0)
    assert len(m["persistence"]["day1"]["fold_rmse"]) == 4
    assert m["persistence"]["day1"]["n_observations"] == 4 * 29
    json.dumps(out)                                   # serialisable


def test_sarima_forecaster_meets_the_contract():
    """sarima_forecaster fits on the training window only, then returns one
    forecast per (fold row i, target row i + h) pair: length len(fold) - h,
    all finite, and identical across repeated calls with the same inputs
    (per-fold fit is cached, not refit from fold data)."""
    pmdarima = pytest.importorskip("pmdarima")
    rng = np.random.default_rng(7)
    days = 260
    values = 24 + 3 * np.sin(2 * np.pi * np.arange(days) / 7) + rng.normal(0, 0.3, days)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "temp_c": values,
        "humidity_pct": 60 + 10 * rng.normal(0, 1, days),
        "pressure_hpa": 1013 + 3 * rng.normal(0, 1, days),
    })
    config = {"seasonal": True, "m": 7, "information_criterion": "aic", "stepwise": True}
    train, fold = df.iloc[:200], df.iloc[200:230]
    for h in (1, 2, 3):
        forecaster = sarima_forecaster(config)
        preds = forecaster(train, fold, h)
        assert len(preds) == len(fold) - h
        assert np.all(np.isfinite(preds))
        # Same closure, same inputs: cached fit, identical output.
        again = forecaster(train, fold, h)
        assert np.array_equal(again, preds)


def test_sarima_forecaster_cache_isolated_by_training_content(monkeypatch):
    """One closure across horizons reuses the per-fold fit, but a same-length
    training window with different content must NOT reuse it: the cache key
    is the training window's content identity plus config, not its length."""
    pytest.importorskip("pmdarima")
    from src.models.sarima_model import SARIMAXModel

    train, fold, config = _sarima_fixture()
    fits = {"n": 0}
    original_fit = SARIMAXModel.fit

    def counting_fit(self, train_df, **kwargs):
        fits["n"] += 1
        return original_fit(self, train_df, **kwargs)

    monkeypatch.setattr(SARIMAXModel, "fit", counting_fit)
    forecaster = sarima_forecaster(config)
    forecaster(train, fold, 1)
    forecaster(train, fold, 2)  # same training window, another horizon
    assert fits["n"] == 1

    same_len_different_data = train.copy()
    same_len_different_data["temp_c"] = same_len_different_data["temp_c"] + 1.0
    forecaster(same_len_different_data, fold, 1)
    assert fits["n"] == 2  # new content: refit, no silent reuse

    forecaster(train, fold, 1)  # original window still cached
    assert fits["n"] == 2


def _sarima_fixture(days=260):
    rng = np.random.default_rng(11)
    values = 24 + 3 * np.sin(2 * np.pi * np.arange(days) / 7) + rng.normal(0, 0.3, days)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "temp_c": values,
        "humidity_pct": 60 + 10 * rng.normal(0, 1, days),
        "pressure_hpa": 1013 + 3 * rng.normal(0, 1, days),
    })
    config = {"seasonal": True, "m": 7, "information_criterion": "aic", "stepwise": True}
    return df.iloc[:200], df.iloc[200:230], config


def _honest_exog_fit(monkeypatch):
    """Test-only shim: route the 1.x ``exogenous=`` kwarg that SARIMAXModel.fit
    passes to pmdarima 2.x's ``X=``. Production under the installed pmdarima
    silently drops the exogenous block; these tests exercise the predict
    boundary for the day that is fixed, without changing production."""
    import src.models.sarima_model as sm
    original = sm.pm.auto_arima

    def honest(y, exogenous=None, **kw):
        return original(y, X=exogenous, **kw)

    monkeypatch.setattr(sm.pm, "auto_arima", honest)


def _assert_exog_reached_fit():
    """Guard: the shim must really put exog into the fitted model, otherwise
    the poisoning tests below could pass vacuously."""
    from src.models.sarima_model import SARIMAXModel
    train, _, config = _sarima_fixture()
    m = SARIMAXModel(config)
    m.fit(train)
    assert m.model.arima_res_.model.exog is not None


def test_sarima_forecaster_ignores_future_targets():
    """Changing temp_c after origin 0 must not change the forecast issued at
    origin 0 for target row h (the first returned prediction)."""
    pytest.importorskip("pmdarima")
    train, fold, config = _sarima_fixture()
    for h in (1, 2, 3):
        base = sarima_forecaster(config)(train, fold, h)
        poisoned = fold.copy()
        poisoned.loc[poisoned.index[1]:, "temp_c"] += 40.0
        alt = sarima_forecaster(config)(train, poisoned, h)
        assert base[0] == pytest.approx(alt[0], abs=1e-9)


def test_sarima_forecaster_ignores_future_exogenous(monkeypatch):
    """Changing humidity/pressure after origin 0 must not change the forecast
    issued at origin 0: future exogenous values are unknown at issue time, so
    the forecaster must stand in the last observed rows (serving parity).
    temp_c is exog-driven and the fit is made exog-honest (test-only shim),
    so a forecaster reading observed future exog fails this test."""
    pytest.importorskip("pmdarima")
    _honest_exog_fit(monkeypatch)
    _assert_exog_reached_fit()
    train, fold, config = _sarima_fixture()
    rng = np.random.default_rng(5)
    for frame in (train, fold):
        frame["temp_c"] = (
            0.4 * frame["humidity_pct"] - 0.1 * frame["pressure_hpa"]
            + 120.0 + rng.normal(0, 0.5, len(frame))
        )
    for h in (1, 2, 3):
        base = sarima_forecaster(config)(train, fold, h)
        poisoned = fold.copy()
        poisoned.loc[poisoned.index[1]:, "humidity_pct"] += 50.0
        poisoned.loc[poisoned.index[1]:, "pressure_hpa"] -= 60.0
        alt = sarima_forecaster(config)(train, poisoned, h)
        assert base[0] == pytest.approx(alt[0], abs=1e-9)
