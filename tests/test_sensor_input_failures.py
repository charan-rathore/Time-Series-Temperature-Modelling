from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from src.api.routes import sensor

@pytest.mark.parametrize("query",["start_date=not-a-date","end_date=2026-99-99","start_date=2026-02-02&end_date=2026-01-01"])
def test_invalid_history_range_is_client_error(monkeypatch,tmp_path,query):
    path=tmp_path/"readings.db";sensor.init_database(path);monkeypatch.setattr(sensor,"_DB_PATH",path)
    app=FastAPI();app.include_router(sensor.router,prefix="/sensor")
    assert TestClient(app,raise_server_exceptions=False).get('/sensor/history?'+query).status_code==422

