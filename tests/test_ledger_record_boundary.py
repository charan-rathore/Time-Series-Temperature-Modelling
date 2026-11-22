"""Appending must not concatenate a new record onto an unterminated prefix."""
from pathlib import Path

import pytest

from src.evaluation import ledger as L


def observation(day: int):
    return {
        "kind": "observation", "source": L.OBS_SOURCE, "target": L.TARGET,
        "date": f"2026-11-{day:02d}", "value": 22.0,
        "issued_at": f"2026-11-{day + 1:02d}T10:00:00Z",
    }


@pytest.mark.parametrize("suffix", [b"", b" ", b"\t", b"\r", b" " * 8193])
def test_refuses_unterminated_record_without_changing_verified_prefix(tmp_path: Path, suffix: bytes):
    path = tmp_path / "ledger.jsonl"
    L.append(observation(1), path)
    path.write_bytes(path.read_bytes().rstrip(b"\n") + suffix)
    before = path.read_bytes()
    assert L.verify(path) == []
    assert len(L.read(path)) == 1
    with pytest.raises(L.LedgerError, match="newline"):
        L.append(observation(2), path)
    assert path.read_bytes() == before
    assert L.verify(path) == []


@pytest.mark.parametrize("suffix", [b"\n", b"\r\n", b"\n \t", b"\n" + b" " * 8193])
def test_terminated_record_and_trailing_blank_whitespace_remain_appendable(tmp_path: Path, suffix: bytes):
    path = tmp_path / "ledger.jsonl"
    L.append(observation(1), path)
    path.write_bytes(path.read_bytes().rstrip(b"\n") + suffix)
    L.append(observation(2), path)
    assert len(L.read(path)) == 2
    assert L.verify(path) == []


@pytest.mark.parametrize("prefix", [b"", b" \t", b" " * 8193])
def test_empty_or_whitespace_only_prefix_accepts_first_record(tmp_path: Path, prefix: bytes):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(prefix)
    L.append(observation(1), path)
    assert len(L.read(path)) == 1
    assert L.verify(path) == []
