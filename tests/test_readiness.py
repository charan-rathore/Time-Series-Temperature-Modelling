"""Actual HTTP readiness is distinct from process liveness."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.mark.parametrize("config,manager,ready", [({},None,False),({"location":{}},None,False),({"location":{}},SimpleNamespace(is_loaded=False),False),({"location":{}},SimpleNamespace(is_loaded=True),True)])
def test_readiness_checks_config_and_loaded_model(monkeypatch,config,manager,ready):
    monkeypatch.setattr(app.state,"config",config,raising=False)
    monkeypatch.setattr(app.state,"model_manager",manager,raising=False)
    client=TestClient(app)
    response=client.get("/api/ready")
    assert response.status_code == (200 if ready else 503)
    assert response.json()["ready"] is ready
    assert response.json()["checks"] == {"config":bool(config),"models":bool(manager and manager.is_loaded)}
    assert client.get("/api/health").status_code == 200
    assert client.get("/health").json() == {"status":"ok"}
