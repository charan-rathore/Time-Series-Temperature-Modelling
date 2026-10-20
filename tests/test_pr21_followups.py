"""Regression tests for review findings on the sidebar/real-data change."""
import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import baseline_collector
from src.models import loader

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- reference actual must not overwrite sensor truth -----------------------

def _run_reference(tmp_path, monkeypatch, existing_source=None):
    db = tmp_path / "f.db"
    mod = _load_script("record_reference_actual")
    day = date(2026, 10, 1)
    df = pd.DataFrame({"date": [pd.Timestamp(day)], "temp_c": [30.0], "humidity_pct": [50.0],
                       "temp_c_api": [29.0]})
    monkeypatch.setattr(mod, "load_processed", lambda: df)
    monkeypatch.setattr(mod, "_DB_PATH", db)
    baseline_collector.init_database(db)
    if existing_source:
        assert baseline_collector.store_actual(db, day, 22.5, 40.0, source=existing_source)
    return mod, db, day


def _row(db, day):
    import sqlite3
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT sensor_temp_c, source FROM daily_actuals WHERE date=?",
                            (day.isoformat(),)).fetchone()
    finally:
        conn.close()


def test_reference_does_not_overwrite_sensor_reading(tmp_path, monkeypatch):
    mod, db, day = _run_reference(tmp_path, monkeypatch, existing_source="raspberry_pi")
    assert mod.main() == 0
    assert _row(db, day) == (22.5, "raspberry_pi")


def test_reference_is_stored_when_no_actual_exists(tmp_path, monkeypatch):
    mod, db, day = _run_reference(tmp_path, monkeypatch)
    assert mod.main() == 0
    assert _row(db, day) == (30.0, "open_meteo_reference")


def test_reference_exits_nonzero_when_write_fails(tmp_path, monkeypatch):
    import sqlite3
    mod, db, day = _run_reference(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mod, "store_reference", boom)
    assert mod.main() == 1


def test_reference_write_is_one_atomic_statement(tmp_path, monkeypatch):
    """A sensor row that exists at write time survives; an older reference row is refreshed."""
    mod, db, day = _run_reference(tmp_path, monkeypatch, existing_source="raspberry_pi")
    assert mod.store_reference(db, day, 30.0, 50.0, 29.0) is False
    assert _row(db, day) == (22.5, "raspberry_pi")

    other = tmp_path / "other"
    other.mkdir()
    mod2, db2, day2 = _run_reference(other, monkeypatch)
    assert mod2.store_reference(db2, day2, 30.0, 50.0, 29.0) is True
    assert mod2.store_reference(db2, day2, 31.0, 50.0, 29.0) is True
    assert _row(db2, day2) == (31.0, "open_meteo_reference")


# --- sensor endpoints keep arbitrary uploader sources -----------------------

def test_sensor_endpoints_keep_non_sensor_named_sources(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    db = tmp_path / "f.db"
    with patch("src.api.routes.sensor._DB_PATH", db):
        baseline_collector.init_database(db)
        baseline_collector.store_actual(db, date(2026, 10, 1), 22.5, 40.0, source="raspberry_pi")
        baseline_collector.store_actual(db, date(2026, 10, 2), 30.0, 50.0, source="open_meteo_reference")
        from src.api.routes.sensor import router
        app = FastAPI()
        app.include_router(router, prefix="/api/sensor")
        client = TestClient(app)
        latest = client.get("/api/sensor/latest")
        assert latest.status_code == 200
        assert latest.json()["source"] == "raspberry_pi"
        history = client.get("/api/sensor/history?start_date=2026-10-01&end_date=2026-10-31").json()
        text = str(history)
        assert "raspberry_pi" in text and "open_meteo_reference" not in text


# --- forecast serves the selected best model --------------------------------

def _manager(best, monkeypatch):
    features = pd.DataFrame({"temp_c": [20.0, 21.0, 22.0]})
    monkeypatch.setattr(loader, "load_processed", lambda: features)
    monkeypatch.setattr(loader, "build_feature_matrix", lambda *a, **kw: features)
    manager = loader.ModelManager()
    manager.is_loaded = True
    manager.tft = None
    manager.ensemble = None
    manager.lgbm_models = {1: SimpleNamespace(predict=lambda **kw: np.array([50.0]))}
    manager.sarima = SimpleNamespace(
        is_fitted=True,
        predict_intervals=lambda **kw: {"median": np.array([10.0]), "lower": np.array([9.0]),
                                         "upper": np.array([11.0])},
    )
    monkeypatch.setattr(manager, "get_best_model_name", lambda: best)
    return manager


def test_forecast_serves_selected_sarima_over_loaded_lgbm(monkeypatch):
    result = _manager("sarima", monkeypatch).forecast(days=1)[0]
    assert (result["model_used"], result["predicted_temp_c"]) == ("sarima", 10.0)


def test_forecast_serves_selected_lgbm(monkeypatch):
    result = _manager("lgbm", monkeypatch).forecast(days=1)[0]
    assert (result["model_used"], result["predicted_temp_c"]) == ("lgbm", 50.0)


def test_forecast_falls_back_when_selected_model_fails(monkeypatch):
    manager = _manager("lgbm", monkeypatch)
    manager.lgbm_models = {1: SimpleNamespace(predict=lambda **kw: (_ for _ in ()).throw(RuntimeError("x")))}
    result = manager.forecast(days=1)[0]
    assert result["model_used"] == "sarima"


def test_fallback_at_one_horizon_does_not_change_the_next_horizons_choice(monkeypatch):
    manager = _manager("lgbm", monkeypatch)
    ok = lambda **kw: np.array([50.0])
    manager.lgbm_models = {1: SimpleNamespace(predict=ok), 3: SimpleNamespace(predict=ok)}
    used = [r["model_used"] for r in manager.forecast(days=3)]
    assert used == ["lgbm", "sarima", "lgbm"]


# --- ensemble training: horizon alignment and no TFT KeyError ---------------

class _FakeSarima:
    def __init__(self, cfg):
        pass

    def fit(self, df):
        return self

    def predict(self, steps, future_df=None):
        return np.zeros(steps)  # uninformative: only aligned LightGBM can score well


class _FakeLgbm:
    def __init__(self, cfg, horizon=1):
        self.h = horizon

    def fit(self, df, val_df=None):
        return self

    def predict(self, steps=1, future_df=None):
        t = future_df["temp_c"].values.astype(float)
        out = np.zeros(len(t))
        out[: len(t) - self.h] = t[self.h:]  # row t predicts day t + h
        return out[:steps]


def _frames():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(size=60)) + 20
    mk = lambda a: pd.DataFrame({"temp_c": a})
    return mk(series[:30]), mk(series[30:45]), mk(series[45:])


def test_ensemble_scores_each_horizon_against_its_own_target_day(monkeypatch, tmp_path):
    mod = _load_script("train_models")
    monkeypatch.setattr(mod, "SARIMAXModel", _FakeSarima)
    monkeypatch.setattr(mod, "LGBMForecastModel", _FakeLgbm)
    monkeypatch.setattr(mod, "MODELS_DIR", tmp_path)
    train, val, test = _frames()
    config = {"models": {"ensemble": {"alpha": 1e-9}, "sarima": {}, "lgbm": {}}}
    results = mod.train_ensemble(train, val, test, config, {}, {}, {})
    # The fake LightGBM is exact for its own target day (row t predicts day
    # t + h). Scored against the right day the stacked error is ~0; scored
    # against the unshifted row it is large.
    for h in (1, 2, 3):
        assert results[f"day{h}"]["rmse"] < 1e-3, h


def test_ensemble_does_not_raise_when_tft_results_exist(monkeypatch, tmp_path):
    mod = _load_script("train_models")
    monkeypatch.setattr(mod, "SARIMAXModel", _FakeSarima)
    monkeypatch.setattr(mod, "LGBMForecastModel", _FakeLgbm)
    monkeypatch.setattr(mod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_TFT_AVAILABLE", True)
    train, val, test = _frames()
    config = {"models": {"ensemble": {"alpha": 1.0}, "sarima": {}, "lgbm": {}}}
    results = mod.train_ensemble(train, val, test, config, {}, {}, {"day1": {"rmse": 1.0}})
    assert set(results) == {"day1", "day2", "day3"}
