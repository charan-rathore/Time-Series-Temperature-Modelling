from unittest.mock import Mock
from hardware import uploader

def test_cycle_stops_when_full_batch_makes_no_progress(monkeypatch):
    batch=Mock(return_value={"fetched":2,"uploaded":0,"marked_synced":0,"failed":2})
    monkeypatch.setattr(uploader,"upload_batch",batch)
    # Bound a buggy loop without hanging the test.
    batch.side_effect=[batch.return_value,AssertionError("repeated failed batch")]
    result=uploader.run_once("sensor","cloud",None,2)
    assert result["batches"]==1

def test_cycle_stops_when_acknowledgement_fails(monkeypatch):
    batch=Mock(side_effect=[{"fetched":2,"uploaded":2,"marked_synced":0,"failed":0},AssertionError("repeated unacked batch")])
    monkeypatch.setattr(uploader,"upload_batch",batch)
    assert uploader.run_once("sensor","cloud",None,2)["batches"]==1

def test_fetch_respects_batch_limit(monkeypatch):
    get=Mock();get.return_value.json.return_value={"readings":[]}
    monkeypatch.setattr(uploader.requests,"get",get)
    uploader.fetch_unsynced_readings("sensor",limit=2)
    assert get.call_args.kwargs["params"]=={"limit":2}

def test_uploader_limit_is_supported_by_actual_sensor_http_route(monkeypatch,tmp_path):
    from hardware import sensor_daemon as daemon
    from http.server import HTTPServer
    import threading
    from datetime import datetime
    path=tmp_path/'sensor.db';daemon.init_database(path)
    for n in range(3):daemon.store_reading({"timestamp":f"2026-10-05T21:00:0{n}","temp_c":25.,"humidity_pct":50.,"source":"test"},path)
    monkeypatch.setattr(daemon.SensorHTTPHandler,'db_path',path)
    server=HTTPServer(('127.0.0.1',0),daemon.SensorHTTPHandler)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        rows=uploader.fetch_unsynced_readings(f'http://127.0.0.1:{server.server_port}',2)
        assert len(rows)==2
    finally:server.shutdown();thread.join();server.server_close()
