from datetime import date,timedelta
import sqlite3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.data.baseline_collector import init_database,store_forecast,store_actual
from src.api.routes import statistics as routes

@pytest.fixture
def database(tmp_path,monkeypatch):
    path=tmp_path/'stats.sqlite';init_database(path)
    monkeypatch.setattr(routes,'_DB_PATH',path)
    return path


def populate(path,ts_days,baseline_days):
    for day in sorted(set(ts_days)|set(baseline_days)):
        d=date.today()-timedelta(days=day)
        store_actual(path,d,sensor_temp_c=20.)
        if day in ts_days:store_forecast(path,d,'thermosense',1,20.+day)
        if day in baseline_days:store_forecast(path,d,'reference',1,20.-day/2)


def client():
    app=FastAPI();app.include_router(routes.router,prefix='/statistics');return TestClient(app)


def test_disjoint_dates_not_claimed_matched(database):
    populate(database,[1,2,3],[4,5,6])
    response=client().get('/statistics/compare?baseline=reference')
    assert response.status_code==400
    assert 'only have 0' in response.json()['detail']
    assert client().get('/statistics/all-comparisons').json()['total_comparisons']==0


def test_partial_overlap_uses_exact_dates_for_absolute_and_signed_errors(database,monkeypatch):
    populate(database,[1,2,3,4],[2,3,4,5])
    absolute=[];signed=[]
    real=routes.compare_forecasters;real_dm=routes.diebold_mariano_test
    def comparison(a,b,**kw):absolute.append((list(a),list(b)));return real(a,b,**kw)
    def dm(a,b,**kw):signed.append((list(a),list(b)));return real_dm(a,b,**kw)
    monkeypatch.setattr(routes,'compare_forecasters',comparison)
    monkeypatch.setattr(routes,'diebold_mariano_test',dm)
    response=client().get('/statistics/compare?baseline=reference')
    assert response.status_code==200
    assert response.json()['comparison']['n_paired_samples']==3
    assert absolute==[([4.,3.,2.],[2.,1.5,1.])]
    assert signed==[([4.,3.,2.],[-2.,-1.5,-1.])]
    absolute.clear()
    out=client().get('/statistics/all-comparisons').json()
    assert out['comparisons'][0]['n_samples']==3
    assert absolute==[([4.,3.,2.],[2.,1.5,1.])]


def test_overlap_below_minimum_rejected(database):
    populate(database,[1,2,3],[2,3,4])
    assert client().get('/statistics/compare?baseline=reference').status_code==400
    assert client().get('/statistics/all-comparisons').json()['total_comparisons']==0


def test_horizon_cutoff_and_actual_required(database):
    populate(database,[1,2,3],[1,2,3])
    # Other horizon and old dates must not expand the paired sample.
    old=date.today()-timedelta(days=60)
    store_actual(database,old,sensor_temp_c=20.)
    for source in ('thermosense','reference'):
        store_forecast(database,old,source,1,25.)
        store_forecast(database,date.today()-timedelta(days=4),source,1,30.) # no actual
        store_forecast(database,date.today()-timedelta(days=1),source,2,40.)
    ours,other=routes._get_paired_signed_errors('reference',1,30)
    assert ours==[3.,2.,1.] and other==[-1.5,-1.,-.5]
    ours,other=routes._get_paired_signed_errors('reference',2,30)
    assert ours==[20.] and other==[20.]


def test_missing_source_404_and_empty_reader(database):
    populate(database,[1,2,3],[])
    assert client().get('/statistics/compare?baseline=absent').status_code==404
    with sqlite3.connect(database) as conn:
        conn.execute("DELETE FROM daily_forecasts")
    assert client().get('/statistics/compare?baseline=reference').status_code==404


def test_signed_absolute_reader_compatibility(database):
    populate(database,[1,2,3],[1,2,3])
    assert routes._get_errors_for_source('reference',1,30)==[1.5,1.,.5]
    assert routes._get_signed_errors_for_source('reference',1,30)==[-1.5,-1.,-.5]


def test_missing_database_does_not_create_it(tmp_path,monkeypatch):
    path=tmp_path/'absent.sqlite';monkeypatch.setattr(routes,'_DB_PATH',path)
    assert routes._get_paired_signed_errors('reference',1,30)==([],[])
    assert not path.exists()
