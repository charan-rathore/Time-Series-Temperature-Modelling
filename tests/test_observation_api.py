import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes.observations import router,store
from src.history.egres import HistoryConflict

@pytest.fixture
def client():
    app=FastAPI();app.include_router(router)
    return TestClient(app)

def test_fail_closed(client,monkeypatch):
    monkeypatch.delenv('THERMOSENSE_HISTORY_TOKEN',raising=False)
    assert client.get('/readings',params=dict(location='x',start='x',end='x',source='x')).status_code==503

def test_auth_rejected(client,monkeypatch):
    monkeypatch.setenv('THERMOSENSE_HISTORY_TOKEN','test-only-token')
    monkeypatch.setenv('THERMOSENSE_HISTORY_DSN','not-a-real-dsn')
    assert client.post('/readings',json={'readings':[]}).status_code==401

def test_post_and_read_preserve_timestamp(client):
    class Fake:
        def __init__(self):self.rows=[]
        def append(self,r):self.rows.append(r);return True
        def read(self,**kw):return self.rows
    fake=Fake();client.app.dependency_overrides[store]=lambda:fake
    row=dict(location='site',source='sensor',observed_at='2026-10-28T13:30:00+05:30',temperature_c=25.)
    assert client.post('/readings',json={'readings':[row]}).json()['stored']==1
    out=client.get('/readings',params=dict(location='site',start='2026-10-28',end='2026-10-28',source='sensor')).json()
    assert out['readings'][0]['original_observed_at']==row['observed_at']
    assert out['summary']['observed_change_c'] is None
    assert client.post('/readings',json={'readings':[dict(row,observed_at='2026-10-28T00:00:00')]}).status_code==422
    assert len(fake.rows)==1

def test_invalid_whole_batch_has_no_partial_writes(client):
    fake=type('Fake',(),{'append':lambda self,r: (_ for _ in ()).throw(AssertionError('must not write'))})()
    client.app.dependency_overrides[store]=lambda:fake
    row=dict(location='site',source='sensor',observed_at='2026-10-28T08:00:00Z',temperature_c=25.)
    assert client.post('/readings',json={'readings':[row,dict(row,observed_at='bad')]}).status_code==422

def test_storage_error_does_not_claim_success(client):
    def broken(self,r):raise OSError('private storage details')
    client.app.dependency_overrides[store]=lambda:type('Fake',(),{'append':broken})()
    row=dict(location='site',source='sensor',observed_at='2026-10-28T08:00:00Z',temperature_c=25.)
    response=client.post('/readings',json={'readings':[row]})
    assert response.status_code==503 and 'private storage' not in response.text

def test_non_ascii_bearer_is_unauthorized_not_server_error(client,monkeypatch):
    monkeypatch.setenv('THERMOSENSE_HISTORY_TOKEN','test-only-token')
    monkeypatch.setenv('THERMOSENSE_HISTORY_DSN','not-a-real-dsn')
    response=client.get('/readings',params=dict(location='x',start='x',end='x',source='x'),
                        headers=[(b'authorization',b'Bearer invalid-\xe9')])
    assert response.status_code==401
