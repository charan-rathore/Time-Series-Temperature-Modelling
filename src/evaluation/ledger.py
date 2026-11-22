"""Append-only, hash-chained ledger of forecasts and observations.

Why: a forecast comparison is only fair if every forecast was written down
before its target day, never edited afterwards, and scored against the same
quantity it predicted. The SQLite baseline table allows UPDATE and DELETE and
overwrites actuals in place; this ledger does not.

Runtime contract: cooperating processes on one local POSIX filesystem, with
exclusive ownership of the ledger pathname. Do not rotate, replace, restore,
unlink, or modify that path while readers/writers are active. Inode checks
detect observed replacement, but cannot make noncooperating tools atomic.
Parent directories must already be durable. File and parent-directory fsync
cover normal appends, not crash recovery after an interrupted partial write.
An uncertain-commit or recovery error requires inspection before any retry.

Format: one JSON object per line (JSONL), committed to git. Each record
carries `prev` (the previous record's hash) and `hash` =
sha256(prev + canonical JSON of the record without `hash`). Editing,
reordering or deleting any earlier line breaks every later hash, and git
history gives an outside timestamp for each appended line.

Rules enforced on append and re-checked by `verify`:
- a forecast's `issued_at` (UTC) is before its target day starts in IST;
- first write wins: one forecast per (source, target, target_date, horizon);
- observations are never edited; a correction is a new record whose
  `supersedes` names the hash it replaces.

Target: `temp_c_2100_ist`, the air temperature at 21:00 IST, because the
only measured series (the VOBG airport METAR poller) records the reading
closest to 21:00 IST. Daily-mean forecasts are not scored against it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))
TARGET = "temp_c_2100_ist"
OBS_SOURCE = "airport_vobg_metar"
GENESIS = "0" * 64
ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "data" / "ledger" / "ledger.jsonl"


class LedgerError(ValueError):
    pass


class LedgerWriteError(LedgerError):
    """The storage write failed, and the prior ledger prefix was restored."""


class LedgerCommitUncertainError(LedgerError):
    """The record was written, but durable completion could not be confirmed."""


class LedgerRecoveryError(LedgerError):
    """A failed append could not establish a durable rollback."""

    def __init__(self, write_error: BaseException, recovery_error: BaseException):
        super().__init__("append failed and recovery could not be confirmed; stop writes and inspect the ledger")
        self.write_error = write_error
        self.recovery_error = recovery_error


def _check_path(fd: int, path: Path) -> None:
    """Detect replacement while open; external writers must keep the inode stable."""
    try:
        current = path.stat()
    except FileNotFoundError as exc:
        raise LedgerError("ledger pathname was removed or replaced while open") from exc
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise LedgerError("ledger pathname was replaced while open")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)



def _canonical(rec: Dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(prev: str, rec: Dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canonical(rec)).encode("utf-8")).hexdigest()


def _read_records(stream) -> List[Dict[str, Any]]:
    out = []
    for n, line in enumerate(stream, 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LedgerError(f"line {n}: not JSON ({exc})") from exc
    return out


def read(path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    try:
        stream = open(path, encoding="utf-8")
    except FileNotFoundError:
        return []
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        _check_path(stream.fileno(), path)
        records = _read_records(stream)
        _check_path(stream.fileno(), path)
        return records


def _utc(ts: str) -> datetime:
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise LedgerError(f"timestamp without timezone: {ts}")
    return t.astimezone(timezone.utc)


def target_day_start(target_date: str) -> datetime:
    """Midnight IST at the start of the target day, in UTC."""
    return datetime.combine(date.fromisoformat(target_date), time(0), IST).astimezone(timezone.utc)


def _forecast_key(r: Dict[str, Any]) -> Tuple:
    return (r["source"], r["target"], r["target_date"], int(r["horizon_days"]))


class _ValidationState:
    """Semantic indexes contain accepted records only, not invalid predecessors.

    A rejected record still makes verification fail. Secondary diagnostics may
    differ from a raw-prefix scan; they do not bless any already-invalid ledger.
    State is rebuilt per scan and never trusted across calls.
    """

    def __init__(self):
        self.last_issued = None
        self.forecasts = set()
        self.observations = {}
        self.observation_keys = set()
        self.superseded = set()

    def check(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            raise LedgerError("record must be an object")
        kind = rec.get("kind")
        if kind not in ("forecast", "observation"):
            raise LedgerError(f"unknown kind {kind!r}")
        for field in ("source", "target", "issued_at"):
            if not isinstance(rec.get(field), str) or not rec[field]:
                raise LedgerError(f"{kind} needs {field} as a nonempty string")
        value = rec.get("value")
        try:
            finite = math.isfinite(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else False
        except OverflowError:
            finite = False
        if not finite:
            raise LedgerError("value must be a finite number")
        try:
            issued = _utc(rec["issued_at"])
            if self.last_issued is not None and issued < self.last_issued:
                raise LedgerError("issued_at goes backwards; records are appended in time order")
            if kind == "forecast":
                horizon = rec.get("horizon_days")
                if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                    raise LedgerError("forecast needs an integer horizon_days >= 1")
                if issued >= target_day_start(rec["target_date"]):
                    raise LedgerError("a forecast must be issued before its target day starts (IST)")
                if (date.fromisoformat(rec["target_date"]) - issued.astimezone(IST).date()).days != horizon:
                    raise LedgerError("horizon_days does not match issue day and target date (IST)")
                key = _forecast_key(rec)
                if key in self.forecasts:
                    raise LedgerError(f"duplicate forecast {key}; first write wins")
            else:
                date.fromisoformat(rec["date"])
                key = (rec["source"], rec["target"], rec["date"])
                supersedes = rec.get("supersedes")
                if supersedes is not None:
                    if not isinstance(supersedes, str):
                        raise LedgerError("supersedes must be a hash string")
                    if supersedes in self.superseded:
                        raise LedgerError("supersedes names an already superseded observation; correct the live record")
                    old = self.observations.get(supersedes)
                    if old is None or old != key:
                        raise LedgerError("supersedes must name an earlier observation of the same source, target and date")
                elif key in self.observation_keys:
                    raise LedgerError("observation already recorded; correct it with a superseding record")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(f"invalid {kind} fields: {exc}") from exc

    def add(self, rec: Dict[str, Any]) -> None:
        self.last_issued = _utc(rec["issued_at"])
        if rec["kind"] == "forecast":
            self.forecasts.add(_forecast_key(rec))
        else:
            key = (rec["source"], rec["target"], rec["date"])
            self.observation_keys.add(key)
            hash_value = rec.get("hash")
            if isinstance(hash_value, str):
                self.observations[hash_value] = key
            if rec.get("supersedes") is not None:
                self.superseded.add(rec["supersedes"])


def _check(rec: Dict[str, Any], prior: List[Dict[str, Any]]) -> None:
    """Check a new record against a prefix already accepted by verify_records."""
    state = _ValidationState()
    for previous in prior:
        state.add(previous)
    state.check(rec)


def append(rec: Dict[str, Any], path: Path = LEDGER_PATH) -> Dict[str, Any]:
    """Validate, chain and append one record. Returns the stored record."""
    # All writers lock the same ledger inode, not a process-local mutex. Keep
    # validation and hash assignment inside the lock so two processes cannot
    # both accept the same identity or append from the same previous hash.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _check_path(fd, path)
        with os.fdopen(os.dup(fd), encoding="utf-8") as stream:
            prior = _read_records(stream)
        problems = verify_records(prior)
        if problems:
            raise LedgerError("ledger is already inconsistent: " + problems[0])
        rec = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
        _check(rec, prior)
        rec["prev"] = prior[-1]["hash"] if prior else GENESIS
        rec["hash"] = _hash(rec["prev"], rec)
        payload = (json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        original_size = os.fstat(fd).st_size
        # A valid JSON value at EOF need not have a newline, but appending
        # another value there would corrupt its physical JSONL record. Blank
        # trailing whitespace after a newline is safe; never rewrite old bytes.
        boundary = original_size
        while boundary:
            start = max(0, boundary - 4096)
            tail = os.pread(fd, boundary - start, start)
            last_line = tail.rsplit(b"\n", 1)[-1]
            if last_line.strip(b" \t\r"):
                raise LedgerError("ledger final record needs a newline before appending; prefix unchanged")
            if b"\n" in tail:
                break
            boundary = start
        _check_path(fd, path)
        try:
            while payload:
                written = os.write(fd, payload)
                if not written:
                    raise OSError("ledger write made no progress")
                payload = payload[written:]
            os.fsync(fd)
        except BaseException as write_error:
            try:
                os.ftruncate(fd, original_size)
                os.fsync(fd)
            except BaseException as recovery_error:
                raise LedgerRecoveryError(write_error, recovery_error) from write_error
            if isinstance(write_error, OSError):
                raise LedgerWriteError("ledger append failed; rollback completed") from write_error
            raise
        try:
            _check_path(fd, path)
        except (LedgerError, OSError) as exc:
            raise LedgerCommitUncertainError("record written but ledger pathname could not be confirmed; inspect before retrying") from exc
        # File fsync alone does not make a newly created directory entry durable.
        # Failure here is an uncertain durable commit, not permission to retry.
        try:
            _sync_directory(path.parent)
        except OSError as exc:
            raise LedgerCommitUncertainError("record written but directory durability is unconfirmed; inspect before retrying") from exc
    finally:
        os.close(fd)
    return rec


def verify_records(records: List[Dict[str, Any]]) -> List[str]:
    problems: List[str] = []
    prev = GENESIS
    state = _ValidationState()
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            problems.append(f"record {i}: record must be an object")
            prev = ""
            continue
        if rec.get("prev") != prev:
            problems.append(f"record {i}: chain broken (prev does not match record {i - 1})")
        elif rec.get("hash") != _hash(prev, rec):
            problems.append(f"record {i}: content changed after it was written (hash mismatch)")
        try:
            state.check(rec)
            state.add(rec)
        except LedgerError as exc:
            problems.append(f"record {i}: {exc}")
        prev = rec.get("hash", "")
        if not isinstance(prev, str):
            problems.append(f"record {i}: hash must be a string")
            prev = ""
    return problems


def verify(path: Path = LEDGER_PATH) -> List[str]:
    return verify_records(read(path))


def current_observations(
    records: Iterable[Dict[str, Any]], target: str = TARGET, source: Optional[str] = OBS_SOURCE
) -> Dict[str, Dict[str, Any]]:
    """The live observation per date: superseded records drop out, so a
    correction replaces what it corrects. Only `source` when given."""
    obs = [
        r
        for r in records
        if r["kind"] == "observation" and r["target"] == target and (source is None or r["source"] == source)
    ]
    superseded = {r["supersedes"] for r in obs if r.get("supersedes")}
    out: Dict[str, Dict[str, Any]] = {}
    for r in obs:
        if r.get("hash") not in superseded:
            out[r["date"]] = r
    return out


def _block_ci(diffs: List[float], block: int = 7, reps: int = 2000, seed: int = 7) -> Tuple[float, float]:
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        sample: List[float] = []
        while len(sample) < n:
            s = rng.randrange(0, max(1, n - block + 1))
            sample.extend(diffs[s : s + block])
        means.append(sum(sample[:n]) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def score(
    records: List[Dict[str, Any]],
    target: str = TARGET,
    baseline: str = "persistence",
    min_n: int = 60,
    obs_source: Optional[str] = OBS_SOURCE,
) -> Dict[str, Any]:
    """MAE/RMSE per source and horizon on dates where the source AND the
    baseline both forecast and an observation exists. A source may be called
    better than the baseline only with n >= min_n and a 95% block-bootstrap
    interval of the squared-error difference entirely below zero."""
    obs = current_observations(records, target, obs_source)
    fc: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in records:
        if r["kind"] == "forecast" and r["target"] == target:
            fc.setdefault((r["source"], r["horizon_days"]), {})[r["target_date"]] = float(r["value"])
    out: Dict[str, Any] = {
        "target": target,
        "observed_by": obs_source,
        "baseline": baseline,
        "min_n": min_n,
        "rows": [],
    }
    for (source, h), by_date in sorted(fc.items()):
        base = fc.get((baseline, h), {})
        dates = sorted(d for d in by_date if d in obs and (source == baseline or d in base))
        if not dates:
            continue
        err = [by_date[d] - obs[d]["value"] for d in dates]
        row: Dict[str, Any] = {
            "source": source,
            "horizon_days": h,
            "n": len(dates),
            "mae": sum(abs(e) for e in err) / len(err),
            "rmse": math.sqrt(sum(e * e for e in err) / len(err)),
            "first": dates[0],
            "last": dates[-1],
        }
        if source != baseline:
            diffs = [(by_date[d] - obs[d]["value"]) ** 2 - (base[d] - obs[d]["value"]) ** 2 for d in dates]
            lo, hi = _block_ci(diffs) if len(diffs) >= 2 else (float("nan"), float("nan"))
            row["sq_err_minus_baseline_ci95"] = [lo, hi]
            row["claim"] = (
                "better than baseline"
                if len(dates) >= min_n and hi < 0
                else "worse than baseline"
                if len(dates) >= min_n and lo > 0
                else f"no claim (n={len(dates)}, need {min_n} and an interval excluding 0)"
            )
        out["rows"].append(row)
    return out
