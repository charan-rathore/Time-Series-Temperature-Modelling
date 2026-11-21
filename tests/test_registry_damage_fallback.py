"""A damaged optional registry must not take usable forecast output down."""
import json
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_config
import src.api.routes.forecast as route


_HASHES = {"sarima.pkl": "0123456789abcdef"}


def _valid():
    return {
        "champions": {"sarima": "sarima-ok"},
        "versions": [{"model": "sarima", "version": "sarima-ok", "artifacts": dict(_HASHES),
                      "data_range": "2024-06-01 to 2026-11-20", "created_at": "2026-11-21T00:00:00Z"}],
    }


def _broken(kind):
    if kind == "list root": return []
    if kind == "null root": return None
    if kind == "scalar root": return 12
    value = _valid()
    if kind == "scalar versions": value["versions"] = 5
    elif kind == "null versions": value["versions"] = None
    elif kind == "empty champions": value["champions"] = []
    elif kind == "null champions": value["champions"] = None
    elif kind == "dict versions": value["versions"] = {"unexpected": "shape"}
    elif kind == "list champions": value["champions"] = ["sarima-ok"]
    elif kind == "string champions": value["champions"] = "sarima-ok"
    elif kind == "list artifacts": value["versions"][0]["artifacts"] = ["sarima.pkl"]
    elif kind == "string artifacts": value["versions"][0]["artifacts"] = "0123456789abcdef"
    elif kind == "list data range": value["versions"][0]["data_range"] = ["2024-06-01", "2026-11-20"]
    elif kind == "dict created at": value["versions"][0]["created_at"] = {"timestamp": "2026-11-21"}
    return value


@pytest.fixture
def client(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        artifact_hashes=dict(_HASHES), data_through=lambda: date(2026, 11, 20),
        forecast=lambda days: [{"model_used": "sarima", "predicted_temp_c": 23.0,
                                "lower_bound_c": 22.0, "upper_bound_c": 24.0} for _ in range(days)],
    )
    monkeypatch.setattr(app.state, "model_manager", manager, raising=False)
    monkeypatch.setattr(app.state, "config", load_config(), raising=False)
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(route, "REGISTRY_PATH", registry)
    return TestClient(app, raise_server_exceptions=False), registry


@pytest.mark.parametrize("kind", ["list root", "null root", "scalar root", "scalar versions", "null versions", "dict versions", "empty champions", "null champions", "list champions", "string champions", "list artifacts", "string artifacts", "list data range", "dict created at"])
def test_malformed_optional_registry_means_unknown_not_500(client, kind):
    http, registry = client
    registry.write_text(json.dumps(_broken(kind)))
    response = http.get('/api/forecast?days=1')
    assert response.status_code == 200
    body = response.json()
    assert body['model_used'] == 'sarima'
    assert body['forecasts'][0]['predicted_temp_c'] == 23.0
    for value in (body, body['forecasts'][0]):
        assert value['trained_data_range'] is None
        assert value['model_trained_at'] is None


def test_valid_optional_registry_still_binds_provenance(client):
    http, registry = client
    registry.write_text(json.dumps(_valid()))
    body = http.get('/api/forecast?days=1').json()
    assert body['trained_data_range'] == '2024-06-01 to 2026-11-20'
    assert body['model_trained_at'] == '2026-11-21T00:00:00Z'
    assert body['forecasts'][0]['trained_data_range'] == body['trained_data_range']


def test_invalid_json_optional_registry_still_means_unknown(client):
    http, registry = client
    registry.write_text('{')
    response = http.get('/api/forecast?days=1')
    assert response.status_code == 200
    assert response.json()['trained_data_range'] is None
