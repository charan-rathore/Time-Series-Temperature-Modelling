from pathlib import Path
import json
import pytest
from src.data import fetcher


def test_failed_raw_write_preserves_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "_RAW_DIR", tmp_path)
    target = fetcher._save_raw({"version": "previous"}, "response.json")
    original = Path.write_text
    def interrupted(path, text, *args, **kwargs):
        original(path, text[:7], *args, **kwargs)
        raise OSError("simulated disk write interruption")
    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError, match="interruption"):
        fetcher._save_raw({"version": "replacement", "records": list(range(20))}, "response.json")
    assert json.loads(target.read_text()) == {"version": "previous"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["response.json"]


def test_successful_raw_replace_keeps_exact_json_and_returned_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "_RAW_DIR", tmp_path)
    payload = {"weather": [25, None, "évidence"]}
    target = fetcher._save_raw(payload, "response.json")
    assert target == tmp_path / "response.json"
    assert target.read_text() == json.dumps(payload, indent=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["response.json"]


def test_failed_raw_replace_preserves_previous_and_cleans_temporary(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "_RAW_DIR", tmp_path)
    target = fetcher._save_raw({"version": "previous"}, "response.json")
    def refused(*args):
        raise PermissionError("simulated replace refused")
    monkeypatch.setattr(fetcher.os, "replace", refused)
    with pytest.raises(PermissionError, match="refused"):
        fetcher._save_raw({"version": "replacement"}, "response.json")
    assert json.loads(target.read_text()) == {"version": "previous"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["response.json"]


def test_replacement_preserves_existing_file_permissions(tmp_path, monkeypatch):
    import stat
    monkeypatch.setattr(fetcher, "_RAW_DIR", tmp_path)
    target = fetcher._save_raw({"version": 1}, "response.json")
    target.chmod(0o640)
    fetcher._save_raw({"version": 2}, "response.json")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
