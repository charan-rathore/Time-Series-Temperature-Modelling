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
