"""Use the actual API/database boundary to choose the closest 9 PM reading."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes import sensor
import pytest

@pytest.mark.parametrize("times,temps,expected",[
    (["20:59:00","21:50:00"],[25.,30.],25.),
    (["21:50:00","21:01:00"],[30.,25.],25.),
    (["21:01:00","21:50:00"],[25.,30.],25.),
    (["20:59:59","21:00:10"],[25.,30.],25.),
])
def test_daily_actual_is_closest_to_nine_pm(monkeypatch,tmp_path,times,temps,expected):
    monkeypatch.setattr(sensor,"_DB_PATH",tmp_path/"readings.db")
    app=FastAPI();app.include_router(sensor.router,prefix="/sensor");client=TestClient(app)
    payload={"readings":[{"timestamp":f"2026-10-06T{t}Z","temp_c":v} for t,v in zip(times,temps)]}
    result=client.post("/sensor/readings",json=payload)
    assert result.status_code==200
    assert result.json()["accepted"]==1
    assert client.get("/sensor/latest").json()["sensor_temp_c"]==expected
