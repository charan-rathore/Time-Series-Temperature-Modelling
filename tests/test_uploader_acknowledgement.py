"""Never discard local readings based on ambiguous cloud acknowledgements."""
from unittest.mock import Mock
import pytest
from hardware import uploader

@pytest.mark.parametrize("reply",[{"accepted":1,"rejected":1},{},{"accepted":2,"rejected":1},{"accepted":"2"},{"accepted":True}])
def test_ambiguous_cloud_reply_keeps_local_readings_unsynced(monkeypatch,reply):
    rows=[{"id":1,"timestamp":"2026-10-06T21:00:00Z","temp_c":25.},{"id":2,"timestamp":"2026-10-06T21:01:00Z","temp_c":26.}]
    response=Mock();response.json.return_value=reply
    monkeypatch.setattr(uploader.requests,"post",Mock(return_value=response))
    monkeypatch.setattr(uploader,"fetch_unsynced_readings",Mock(return_value=rows))
    mark=Mock(return_value=True);monkeypatch.setattr(uploader,"mark_readings_synced",mark)
    stats=uploader.upload_batch("sensor","cloud",None,2)
    mark.assert_not_called()
    assert stats["failed"]==2

def test_complete_cloud_reply_allows_local_ack(monkeypatch):
    rows=[{"id":1,"timestamp":"2026-10-06T21:00:00Z","temp_c":25.}]
    response=Mock();response.json.return_value={"accepted":1,"rejected":0}
    monkeypatch.setattr(uploader.requests,"post",Mock(return_value=response))
    assert uploader.upload_readings_to_cloud("cloud",rows)==(1,[])
