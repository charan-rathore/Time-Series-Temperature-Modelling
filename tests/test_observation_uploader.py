import importlib.util
import sqlite3
from pathlib import Path
from unittest.mock import Mock,patch
import pytest
spec=importlib.util.spec_from_file_location('history_uploader',Path(__file__).parents[1]/'scripts/upload_observation_history.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)

@pytest.fixture
def db(tmp_path):
    path=tmp_path/'readings.db'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE sensor_readings(id integer primary key,timestamp text,temp_c real,humidity_pct real,source text)')
        c.execute("INSERT INTO sensor_readings VALUES(1,'2026-10-28T10:00:00',25,60,'sensor')")
    return path

def test_naive_requires_explicit_zone(db):
    with pytest.raises(ValueError,match='source-timezone'):u.upload(db,'https://example.invalid/readings','test-token','site')

def test_acknowledged_and_no_legacy_flags(db):
    response=Mock(status_code=200);response.json.return_value={'stored':0,'duplicates':1}
    with patch.object(u.requests,'post',return_value=response) as post:
        out=u.upload(db,'https://example.invalid/readings','test-token','site','Asia/Kolkata')
        assert out==dict(acknowledged=1,last_id=1)
        assert post.call_args.kwargs['allow_redirects'] is False
        assert post.call_args.kwargs['json']['readings'][0]['observed_at'].endswith('+05:30')
    with sqlite3.connect(db) as c:assert c.execute('SELECT COUNT(*) FROM sensor_readings').fetchone()[0]==1

@pytest.mark.parametrize('result',[{}, {'stored':True,'duplicates':0},{'stored':0,'duplicates':0},{'stored':2,'duplicates':-1}])
def test_invalid_ack(db,result):
    response=Mock(status_code=200);response.json.return_value=result
    with patch.object(u.requests,'post',return_value=response),pytest.raises(ValueError):
        u.upload(db,'https://example.invalid/readings','test-token','site','Asia/Kolkata')

def test_token_cannot_follow_redirect(db):
    response=Mock(status_code=302)
    with patch.object(u.requests,'post',return_value=response),pytest.raises(ValueError):
        u.upload(db,'https://example.invalid/readings','test-token','site','Asia/Kolkata')

def test_remote_http_rejected(db):
    with pytest.raises(ValueError):u.upload(db,'http://example.invalid/readings','test-token','site','Asia/Kolkata')
