"""Rolling-origin backtest: no leakage, correct pairing, honest skill."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    BASELINES, climatology, persistence, rolling_origin_folds,
    run_backtest, seasonal_naive_7,
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
