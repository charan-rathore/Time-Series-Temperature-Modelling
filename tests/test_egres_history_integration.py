"""Opt-in real-release tests. Set THERMOSENSE_TEST_EGRES_DSN to disposable loopback DB only."""
import os
import uuid
import pytest
from concurrent.futures import ThreadPoolExecutor
from src.history.egres import EgresHistory, HistoryConflict, HistoryLimit
from src.history.observations import Observation

@pytest.fixture
def db():
    dsn=os.getenv('THERMOSENSE_TEST_EGRES_DSN')
    if not dsn:pytest.skip('requires isolated real Egres instance')
    db=EgresHistory(dsn);db.initialize();return db

def sample(location,t='2026-10-28T08:00:00Z',temp=25.):
    return Observation.create(location=location,source='test_sensor',observed_at=t,temperature_c=temp)

def test_append_read_replay_conflict(db):
    loc='test-'+uuid.uuid4().hex
    a=sample(loc)
    assert db.append(a) is True
    assert db.append(a) is False
    with pytest.raises(HistoryConflict):db.append(sample(loc,temp=99.))
    rows=db.read(location=loc,start='2026-10-28',end='2026-10-28',source='test_sensor')
    assert len(rows)==1 and rows[0].payload()==a.payload()

def test_out_of_order_source_and_cap(db):
    loc='test-'+uuid.uuid4().hex
    db.append(sample(loc,'2026-10-28T09:00:00Z'));db.append(sample(loc))
    rows=db.read(location=loc,start='2026-10-28',end='2026-10-28',source='test_sensor')
    assert rows[0].observed_at<rows[1].observed_at
    assert db.read(location=loc,start='2026-10-28',end='2026-10-28',source='other')==[]
    with pytest.raises(HistoryLimit):db.read(location=loc,start='2026-10-28',end='2026-10-28',source='test_sensor',max_rows=1)

def test_concurrent_replay(db):
    a=sample('test-'+uuid.uuid4().hex)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(db.append,[a]*4))
    assert sum(result)==1

def test_http_to_real_egres(db,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.observations import router
    monkeypatch.setenv('THERMOSENSE_HISTORY_DSN',db.dsn)
    monkeypatch.setenv('THERMOSENSE_HISTORY_TOKEN','integration-test-only')
    app=FastAPI();app.include_router(router);client=TestClient(app)
    row=dict(location='http-'+uuid.uuid4().hex,source='sensor',observed_at='2026-10-28T08:00:00Z',temperature_c=25.)
    headers={'Authorization':'Bearer integration-test-only'}
    assert client.post('/readings',json={'readings':[row]},headers=headers).json()['stored']==1
    assert client.post('/readings',json={'readings':[row]},headers=headers).json()['duplicates']==1
    assert client.post('/readings',json={'readings':[dict(row,temperature_c=99.)]},headers=headers).status_code==409
    response=client.get('/readings',params=dict(location=row['location'],start='2026-10-28',end='2026-10-28',source='sensor'),headers=headers)
    assert response.status_code==200 and len(response.json()['readings'])==1


def test_seeded_sqlite_differential(db):
    import random,sqlite3
    from datetime import datetime,timedelta,timezone
    rng=random.Random(28012026);loc='fuzz-'+uuid.uuid4().hex
    reference=sqlite3.connect(':memory:')
    reference.execute('CREATE TABLE history (id TEXT PRIMARY KEY, payload TEXT)')
    origin=datetime(2026,10,25,tzinfo=timezone.utc)
    samples=[]
    for index in rng.sample(range(600),80):
        when=(origin+timedelta(minutes=5*index)).isoformat()
        a=Observation.create(location=loc,source='test_sensor',observed_at=when,temperature_c=rng.uniform(-10,45))
        assert db.append(a)
        reference.execute('INSERT INTO history VALUES(?,?)',(a.id,a.payload()));samples.append(a)
    rng.shuffle(samples)
    for a in samples[:20]:assert db.append(a) is False
    rows=db.read(location=loc,start='2026-10-25',end='2026-10-28',source='test_sensor')
    expected=sorted(payload for (payload,) in reference.execute('SELECT payload FROM history'))
    assert sorted(a.payload() for a in rows)==expected
    reference.close()
