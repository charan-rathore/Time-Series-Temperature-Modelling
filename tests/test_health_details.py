import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


def test_health_stays_a_cheap_liveness_probe():
    with TestClient(app) as c:
        assert c.get("/api/health").json() == {"status": "ok", "version": "2.0.0"}


def test_details_reports_freshness_serving_and_quality():
    with TestClient(app) as c:
        body = c.get("/api/health/details").json()
    assert body["status"] in ("ok", "degraded")
    assert {"rows", "freshness", "gaps", "stuck_runs"} <= set(body["data_quality"])
    assert body["serving"]["models_loaded"] is True
    assert body["serving"]["selected_model"] in ("sarima", "lgbm", "ensemble", "tft")


def test_stale_data_degrades_status(monkeypatch):
    import src.data.preprocess as pp
    old = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=60), "temp_c": [25.0 + i % 3 for i in range(60)]})
    monkeypatch.setattr(pp, "load_processed", lambda *a, **k: old)
    with TestClient(app) as c:
        body = c.get("/api/health/details").json()
    assert body["status"] == "degraded" and "data_stale" in body["problems"]


def test_unreadable_data_is_reported_not_raised(monkeypatch):
    import src.data.preprocess as pp
    def boom(*a, **k):
        raise OSError("nope")
    monkeypatch.setattr(pp, "load_processed", boom)
    with TestClient(app) as c:
        r = c.get("/api/health/details")
    assert r.status_code == 200 and "data_unreadable" in r.json()["problems"]
