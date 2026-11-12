"""Forecast responses must carry the served model's training provenance.

A forecast number without its training window hides that the model learned
from grid history ending 2026-10-19 (not from recent station readings).
Provenance is earned only by a verified identity chain: registry record for
the served model (champion pointer when present, never bare versions[-1])
whose recorded artifact hashes match the files the manager actually loaded.
Anything unverified - missing registry, ambiguous records, wrong-model
champion, replaced/stale artifact - is reported as no provenance, never
guessed.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_config
from src.models.registry import file_sha256
import src.api.routes.forecast as forecast_route

MODELS = forecast_route._PROJECT_ROOT / "models"


@pytest.fixture
def client():
    config = load_config()
    app.state.config = config
    from src.models.loader import ModelManager
    if not hasattr(app.state, "model_manager"):
        manager = ModelManager()
        try:
            manager.load_models(config)
        except Exception:
            pass
        app.state.model_manager = manager
    return TestClient(app, raise_server_exceptions=False)


def _real_artifacts(model="lgbm"):
    """The load-time artifact identity the manager recorded for `model`."""
    hashes = forecast_route._served_artifact_hashes(app.state.model_manager, model)
    assert hashes, f"no served artifacts for {model}"
    return dict(hashes)


def _disk_artifacts(model="lgbm"):
    """Artifact hashes straight from disk; valid in unit tests that never
    build the app fixture (the files are what a load would have read)."""
    if model == "lgbm":
        names = ["lgbm_h1.pkl", "lgbm_h2.pkl", "lgbm_h3.pkl"]
    else:
        names = [f"{model}.pkl"]
    return {n: file_sha256(MODELS / n)[:16] for n in names}


def _write_registry(tmp_path, champions, versions):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"champions": champions, "versions": versions}))
    return path


def _lgbm_registry(tmp_path, artifacts=None):
    return _write_registry(
        tmp_path,
        {"lgbm": "lgbm-test"},
        [{
            "model": "lgbm",
            "version": "lgbm-test",
            "created_at": "2026-10-20T15:35:25+00:00",
            "data_range": "2024-06-01 to 2026-10-19",
            "artifacts": artifacts if artifacts is not None else _real_artifacts("lgbm"),
        }],
    )


def test_trained_forecast_carries_training_window(client, tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", _lgbm_registry(tmp_path))
    data = client.get("/api/forecast?days=1").json()
    assert data["model_used"] not in forecast_route._UNTRAINED_MODELS
    assert data["trained_data_range"] == "2024-06-01 to 2026-10-19"
    assert data["model_trained_at"] == "2026-10-20T15:35:25+00:00"


def test_missing_registry_means_no_provenance(client, tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", tmp_path / "nope.json")
    data = client.get("/api/forecast?days=1").json()
    assert data["trained_data_range"] is None
    assert data["model_trained_at"] is None


def test_untrained_mode_never_claims_provenance(client, tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", _lgbm_registry(tmp_path))
    original = app.state.model_manager
    app.state.model_manager = None  # forces the placeholder path
    try:
        data = client.get("/api/forecast?days=1").json()
    finally:
        app.state.model_manager = original
    assert data["model_used"] == "placeholder"
    assert data["trained_data_range"] is None
    assert data["model_trained_at"] is None


def test_later_unrelated_record_does_not_hijack_provenance(client, tmp_path, monkeypatch):
    """Served lgbm must not inherit a later sarima record's window (versions[-1] bug)."""
    path = _write_registry(
        tmp_path,
        {"lgbm": "lgbm-test"},
        [
            {
                "model": "lgbm",
                "version": "lgbm-test",
                "created_at": "2026-10-20T15:35:25+00:00",
                "data_range": "2024-06-01 to 2026-10-19",
                "artifacts": _real_artifacts("lgbm"),
            },
            {
                "model": "sarima",
                "version": "sarima-later",
                "created_at": "2091-01-01T00:00:00+00:00",
                "data_range": "2090-01-01 to 2090-12-31",
                "artifacts": {"sarima.pkl": "deadbeefdeadbeef"},
            },
        ],
    )
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", path)
    data = client.get("/api/forecast?days=1").json()
    assert data["trained_data_range"] == "2024-06-01 to 2026-10-19"
    assert data["model_trained_at"] == "2026-10-20T15:35:25+00:00"
    assert "2090" not in (data["trained_data_range"] or "")
    assert "2091" not in (data["model_trained_at"] or "")


def test_replaced_or_stale_artifact_means_unknown(client, tmp_path, monkeypatch):
    """Champion record whose artifact hash does not match the file on disk
    (retrain without registry update, stale registry): no provenance."""
    bogus = {n: "0000000000000000" for n in _real_artifacts("lgbm")}
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", _lgbm_registry(tmp_path, artifacts=bogus))
    data = client.get("/api/forecast?days=1").json()
    assert data["model_used"] not in forecast_route._UNTRAINED_MODELS
    assert data["trained_data_range"] is None
    assert data["model_trained_at"] is None


def test_record_without_artifact_coverage_means_unknown(client, tmp_path, monkeypatch):
    """A record that never tracked the loaded artifact cannot vouch for it."""
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", _lgbm_registry(tmp_path, artifacts={}))
    data = client.get("/api/forecast?days=1").json()
    assert data["trained_data_range"] is None


def test_wrong_model_champion_means_unknown(tmp_path):
    """Champion pointer for lgbm naming a sarima record: never vouch."""
    path = _write_registry(
        tmp_path,
        {"lgbm": "sarima-1"},
        [{
            "model": "sarima",
            "version": "sarima-1",
            "created_at": "2026-09-01T00:00:00+00:00",
            "data_range": "sarima-range",
            "artifacts": {"sarima.pkl": file_sha256(MODELS / "sarima.pkl")[:16]},
        }],
    )
    original = forecast_route.REGISTRY_PATH
    forecast_route.REGISTRY_PATH = path
    try:
        hashes = _disk_artifacts("lgbm")
        assert forecast_route._training_provenance("lgbm", hashes) == (None, None)
    finally:
        forecast_route.REGISTRY_PATH = original


def test_champion_selects_its_own_version_among_several(tmp_path):
    """Two lgbm records; only the champion's artifacts match the files on disk."""
    path = _write_registry(
        tmp_path,
        {"lgbm": "lgbm-old"},
        [
            {"model": "lgbm", "version": "lgbm-old", "created_at": "2026-01-01T00:00:00+00:00",
             "data_range": "old-range", "artifacts": _disk_artifacts("lgbm")},
            {"model": "lgbm", "version": "lgbm-new", "created_at": "2026-11-01T00:00:00+00:00",
             "data_range": "new-range", "artifacts": {n: "0000000000000000" for n in _disk_artifacts("lgbm")}},
        ],
    )
    original = forecast_route.REGISTRY_PATH
    forecast_route.REGISTRY_PATH = path
    try:
        hashes = _disk_artifacts("lgbm")
        assert forecast_route._training_provenance("lgbm", hashes) == ("old-range", "2026-01-01T00:00:00+00:00")
    finally:
        forecast_route.REGISTRY_PATH = original


def test_unknown_identity_is_not_guessed(tmp_path):
    original = forecast_route.REGISTRY_PATH
    try:
        # No champion, two records both matching the artifacts: ambiguous.
        path = _write_registry(tmp_path, {}, [
            {"model": "lgbm", "version": "a", "data_range": "r1", "created_at": "t1", "artifacts": _disk_artifacts("lgbm")},
            {"model": "lgbm", "version": "b", "data_range": "r2", "created_at": "t2", "artifacts": _disk_artifacts("lgbm")},
        ])
        forecast_route.REGISTRY_PATH = path
        hashes = _disk_artifacts("lgbm")
        assert forecast_route._training_provenance("lgbm", hashes) == (None, None)

        # Champion pointing at a version the registry does not record.
        path = _write_registry(tmp_path, {"lgbm": "ghost"}, [
            {"model": "lgbm", "version": "a", "data_range": "r1", "created_at": "t1", "artifacts": _disk_artifacts("lgbm")},
        ])
        forecast_route.REGISTRY_PATH = path
        assert forecast_route._training_provenance("lgbm", hashes) == (None, None)

        # No record for the served model at all.
        path = _write_registry(tmp_path, {}, [
            {"model": "sarima", "version": "s", "data_range": "r", "created_at": "t",
             "artifacts": {"sarima.pkl": file_sha256(MODELS / "sarima.pkl")[:16]}},
        ])
        forecast_route.REGISTRY_PATH = path
        assert forecast_route._training_provenance("lgbm", hashes) == (None, None)

        # No champion, exactly one artifact-verified record: unambiguous.
        sarima_hash = file_sha256(MODELS / "sarima.pkl")[:16]
        assert forecast_route._training_provenance("sarima", {"sarima.pkl": sarima_hash}) == ("r", "t")
    finally:
        forecast_route.REGISTRY_PATH = original


def test_load_then_replace_artifact_and_advance_registry_means_unknown(client, tmp_path, monkeypatch):
    """Publisher repro: the manager loaded the real lgbm, THEN the disk
    artifacts were replaced and the registry advanced to a new version.
    The in-memory model still serves its old predictions, so it must not
    inherit the new record's training window: identity was bound at load
    time, and the new record's hashes do not match it."""
    manager_hashes = _real_artifacts("lgbm")  # captured at load, before "replacement"
    advanced = {n: ("f" * 16 if not h.startswith("f") else "e" * 16) for n, h in manager_hashes.items()}
    path = _write_registry(
        tmp_path,
        {"lgbm": "lgbm-retrained"},
        [{
            "model": "lgbm",
            "version": "lgbm-retrained",
            "created_at": "2091-01-01T00:00:00+00:00",
            "data_range": "2090-01-01 to 2090-12-31",
            "artifacts": advanced,
        }],
    )
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", path)
    data = client.get("/api/forecast?days=1").json()
    assert data["model_used"] not in forecast_route._UNTRAINED_MODELS
    assert data["trained_data_range"] is None
    assert data["model_trained_at"] is None
    assert "2090" not in json.dumps(data)


class _MixedHorizonManager:
    """Day 1 served by lgbm, day 2 falls back to sarima."""

    artifact_hashes = {
        "lgbm_h1.pkl": file_sha256(MODELS / "lgbm_h1.pkl")[:16],
        "sarima.pkl": file_sha256(MODELS / "sarima.pkl")[:16],
    }

    def forecast(self, days: int = 3):
        rows = [
            {"model_used": "lgbm", "predicted_temp_c": 26.0, "lower_bound_c": 24.0, "upper_bound_c": 28.0},
            {"model_used": "sarima", "predicted_temp_c": 26.5, "lower_bound_c": 24.5, "upper_bound_c": 28.5},
        ]
        return rows[:days]


def test_horizon_fallback_carries_per_point_provenance(client, tmp_path, monkeypatch):
    path = _write_registry(
        tmp_path,
        {"lgbm": "lgbm-test"},
        [
            {"model": "lgbm", "version": "lgbm-test", "created_at": "2026-10-20T00:00:00+00:00",
             "data_range": "lgbm-range",
             "artifacts": {"lgbm_h1.pkl": file_sha256(MODELS / "lgbm_h1.pkl")[:16]}},
            {"model": "sarima", "version": "sarima-1", "created_at": "2026-09-01T00:00:00+00:00",
             "data_range": "sarima-range",
             "artifacts": {"sarima.pkl": file_sha256(MODELS / "sarima.pkl")[:16]}},
        ],
    )
    monkeypatch.setattr(forecast_route, "REGISTRY_PATH", path)
    original = app.state.model_manager
    app.state.model_manager = _MixedHorizonManager()
    try:
        data = client.get("/api/forecast?days=2").json()
    finally:
        app.state.model_manager = original
    # Headline (day-1) model keeps response-level provenance.
    assert data["model_used"] == "lgbm"
    assert data["trained_data_range"] == "lgbm-range"
    # Each point carries the provenance of the model that actually served it.
    day1, day2 = data["forecasts"]
    assert day1["model_used"] == "lgbm"
    assert day1["trained_data_range"] == "lgbm-range"
    assert day2["model_used"] == "sarima"
    assert day2["trained_data_range"] == "sarima-range"
    assert day2["model_trained_at"] == "2026-09-01T00:00:00+00:00"
