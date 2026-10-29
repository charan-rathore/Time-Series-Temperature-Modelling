import json
import time
from types import SimpleNamespace
import pandas as pd
from src.api.routes import history


def test_actual_history_handler_keeps_every_response_field_and_row(tmp_path, monkeypatch):
    path = tmp_path / 'history.csv'
    dates = pd.date_range('2000-01-01', periods=10000)
    pd.DataFrame({'date': dates, 'temp_c': [20.125] * 10000,
                  'temp_c_api': [21.25] * 10000, 'humidity_pct': [51.5] * 10000,
                  'pressure_hpa': [1012.25] * 10000, 'is_sensor_reading': [True] * 10000}).to_csv(path,index=False)
    monkeypatch.setattr(history,'PROCESSED_PATH',path)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config={'location':{'name':'fixture'}})))
    started=time.perf_counter()
    result=history.get_history(request,start='2000-01-01',end='2040-01-01')
    elapsed=time.perf_counter()-started
    expected={'date':'2000-01-01','actual_temp_c':20.12,'api_temp_c':21.25,'api_bias':None,
              'humidity_pct':51.5,'pressure_hpa':1012.25,'is_sensor_reading':True}
    assert result.total_records == 10000
    assert result.records[0].model_dump() == expected
    assert result.records[-1].date == dates[-1].date().isoformat()
    assert elapsed < .35, elapsed


def test_actual_fastapi_history_response_bytes(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    path=tmp_path/'history.csv'
    path.write_text('date,temp_c,temp_c_api,api_bias,humidity_pct,pressure_hpa,is_sensor_reading\n2025-01-02,20.125,21.25,-1.125,,1012.25,True\n2025-01-03,22.5,21.2,1.3,55.2,,False\n')
    monkeypatch.setattr(history,'PROCESSED_PATH',path)
    app=FastAPI();app.state.config={'location':{'name':'fixture'}}
    app.include_router(history.router,prefix='/history')
    response=TestClient(app).get('/history?start=2025-01-02&end=2025-01-03')
    assert response.status_code==200
    assert response.json()=={'location':'fixture','start_date':'2025-01-02','end_date':'2025-01-03','total_records':2,'records':[
      {'date':'2025-01-02','actual_temp_c':20.12,'api_temp_c':21.25,'api_bias':-1.12,'humidity_pct':None,'pressure_hpa':1012.25,'is_sensor_reading':True},
      {'date':'2025-01-03','actual_temp_c':22.5,'api_temp_c':21.2,'api_bias':1.3,'humidity_pct':55.2,'pressure_hpa':None,'is_sensor_reading':False}]}
