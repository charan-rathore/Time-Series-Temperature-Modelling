from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from src.api.routes import sensor

@pytest.mark.parametrize("query",["start_date=not-a-date","end_date=2026-99-99","start_date=2026-02-02&end_date=2026-01-01"])
def test_invalid_history_range_is_client_error(monkeypatch,tmp_path,query):
    path=tmp_path/"readings.db";sensor.init_database(path);monkeypatch.setattr(sensor,"_DB_PATH",path)
    app=FastAPI();app.include_router(sensor.router,prefix="/sensor")
    assert TestClient(app,raise_server_exceptions=False).get('/sensor/history?'+query).status_code==422

@pytest.mark.parametrize("temp",["NaN","Infinity","-Infinity"])
def test_nonfinite_sensor_temperature_is_rejected(temp):
    app=FastAPI();app.include_router(sensor.router,prefix="/sensor")
    response=TestClient(app,raise_server_exceptions=False).post('/sensor/readings',json={"readings":[{"timestamp":"2026-10-05T21:00:00Z","temp_c":temp}]})
    assert response.status_code==422
