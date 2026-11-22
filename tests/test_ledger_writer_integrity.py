"""Ledger writers must serialize identity checks and chain assignment."""
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from src.evaluation import ledger as ledger


def observation(value=21.0, supersedes=None):
    record = {
        "kind": "observation", "source": ledger.OBS_SOURCE,
        "target": ledger.TARGET, "date": "2026-11-01", "value": value,
        "observed_at": "2026-11-01T15:30:00Z", "issued_at": "2026-11-02T10:00:00Z",
    }
    if supersedes is not None:
        record["supersedes"] = supersedes
    return record


def test_simultaneous_independent_appends_keep_one_chain(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    original_read = ledger.read
    original_records = ledger._read_records

    def overlapping_read(stream):
        snapshot = original_records(stream)
        time.sleep(0.03)
        return snapshot

    monkeypatch.setattr(ledger, "_read_records", overlapping_read)
    records = [dict(observation(), source=f"station-{i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda record: ledger.append(record, path), records))
    stored = original_read(path)
    assert len(stored) == 8
    assert ledger.verify_records(stored) == []


def test_simultaneous_duplicate_appends_keep_first_write(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    original_read = ledger.read
    original_records = ledger._read_records

    def overlapping_read(stream):
        snapshot = original_records(stream)
        time.sleep(0.03)
        return snapshot

    monkeypatch.setattr(ledger, "_read_records", overlapping_read)

    def append_once(_):
        try:
            ledger.append(observation(), path)
            return True
        except ledger.LedgerError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append_once, range(8)))
    assert results.count(True) == 1
    assert len(original_read(path)) == 1
    assert ledger.verify_records(original_read(path)) == []


def test_correction_cannot_branch_from_already_superseded_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = ledger.append(observation(), path)
    second = ledger.append(observation(22.0, first["hash"]), path)
    with pytest.raises(ledger.LedgerError, match="superseded"):
        ledger.append(observation(23.0, first["hash"]), path)
    latest = ledger.append(observation(24.0, second["hash"]), path)
    assert ledger.current_observations(ledger.read(path))["2026-11-01"] == latest
    assert ledger.verify(path) == []


def test_verifier_rejects_validly_hashed_branched_correction():
    records = []
    for record in [observation(), observation(22.0), observation(23.0)]:
        if records:
            record["supersedes"] = records[0]["hash"]
        record["prev"] = records[-1]["hash"] if records else ledger.GENESIS
        record["hash"] = ledger._hash(record["prev"], record)
        records.append(record)
    assert any("superseded" in error for error in ledger.verify_records(records))


def _process_append(args):
    path, source = args
    ledger.append(dict(observation(), source=source), path)
    return source


def test_separate_process_writers_share_the_same_lock(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "ledger.jsonl"
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert len(list(pool.map(_process_append, [(path, f"process-{i}") for i in range(16)]))) == 16
    assert len(ledger.read(path)) == 16
    assert ledger.verify(path) == []


def test_failed_validation_releases_writer_lock(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append(observation(), path)
    with pytest.raises(ledger.LedgerError):
        ledger.append(observation(), path)
    ledger.append(dict(observation(), source="another-station"), path)
    assert ledger.verify(path) == []


def test_short_writes_are_completed_before_return(tmp_path, monkeypatch):
    original_write = ledger.os.write

    def short_write(fd, payload):
        return original_write(fd, payload[:7])

    monkeypatch.setattr(ledger.os, "write", short_write)
    path = tmp_path / "ledger.jsonl"
    ledger.append(observation(), path)
    assert len(ledger.read(path)) == 1
    assert ledger.verify(path) == []


def test_failed_partial_write_restores_the_previous_complete_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger.append(observation(), path)
    before = path.read_bytes()
    original_write = ledger.os.write
    calls = 0

    def broken_write(fd, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(fd, payload[:11])
        raise OSError("simulated disk write failure")

    with monkeypatch.context() as patch:
        patch.setattr(ledger.os, "write", broken_write)
        with pytest.raises(ledger.LedgerWriteError, match="rollback completed") as caught:
            ledger.append(dict(observation(), source="second-station"), path)
        assert isinstance(caught.value.__cause__, OSError)
        assert "simulated" in str(caught.value.__cause__)
    assert path.read_bytes() == before
    assert ledger.verify(path) == []
    ledger.append(dict(observation(), source="third-station"), path)
    assert ledger.verify(path) == []
