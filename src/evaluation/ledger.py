"""Append-only, hash-chained ledger of forecasts and observations.

Why: a forecast comparison is only fair if every forecast was written down
before its target day, never edited afterwards, and scored against the same
quantity it predicted. The SQLite baseline table allows UPDATE and DELETE and
overwrites actuals in place; this ledger does not.

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


def _canonical(rec: Dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(prev: str, rec: Dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canonical(rec)).encode("utf-8")).hexdigest()


def read(path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LedgerError(f"line {n}: not JSON ({exc})") from exc
    return out


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


def _check(rec: Dict[str, Any], prior: List[Dict[str, Any]]) -> None:
    kind = rec.get("kind")
    if kind not in ("forecast", "observation"):
        raise LedgerError(f"unknown kind {kind!r}")
    for f in ("source", "target", "value", "issued_at"):
        if rec.get(f) in (None, ""):
            raise LedgerError(f"{kind} needs {f}")
    v = rec["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise LedgerError(f"value must be a finite number, got {v!r}")
    issued = _utc(rec["issued_at"])
    if prior and issued < _utc(prior[-1]["issued_at"]):
        raise LedgerError("issued_at goes backwards; records are appended in time order")
    if kind == "forecast":
        h = rec.get("horizon_days")
        if isinstance(h, bool) or not isinstance(h, int) or h < 1:
            raise LedgerError("forecast needs an integer horizon_days >= 1")
        if issued >= target_day_start(rec["target_date"]):
            raise LedgerError("a forecast must be issued before its target day starts (IST)")
        issue_day = issued.astimezone(IST).date()
        if (date.fromisoformat(rec["target_date"]) - issue_day).days != h:
            raise LedgerError("horizon_days does not match issue day and target date (IST)")
        key = _forecast_key(rec)
        if any(p["kind"] == "forecast" and _forecast_key(p) == key for p in prior):
            raise LedgerError(f"duplicate forecast {key}; first write wins")
    else:
        date.fromisoformat(rec["date"])
        sup = rec.get("supersedes")
        live = {p["hash"]: p for p in prior if p["kind"] == "observation"}
        if sup is not None:
            old = live.get(sup)
            if not old or (old["source"], old["target"], old["date"]) != (rec["source"], rec["target"], rec["date"]):
                raise LedgerError("supersedes must name an earlier observation of the same source, target and date")
        elif any(
            (p["source"], p["target"], p["date"]) == (rec["source"], rec["target"], rec["date"])
            for p in live.values()
        ):
            raise LedgerError("observation already recorded; correct it with a superseding record")


def append(rec: Dict[str, Any], path: Path = LEDGER_PATH) -> Dict[str, Any]:
    """Validate, chain and append one record. Returns the stored record."""
    prior = read(path)
    problems = verify_records(prior)
    if problems:
        raise LedgerError("ledger is already inconsistent: " + problems[0])
    rec = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
    _check(rec, prior)
    rec["prev"] = prior[-1]["hash"] if prior else GENESIS
    rec["hash"] = _hash(rec["prev"], rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return rec


def verify_records(records: List[Dict[str, Any]]) -> List[str]:
    problems: List[str] = []
    prev = GENESIS
    for i, rec in enumerate(records, 1):
        if rec.get("prev") != prev:
            problems.append(f"record {i}: chain broken (prev does not match record {i - 1})")
        elif rec.get("hash") != _hash(prev, rec):
            problems.append(f"record {i}: content changed after it was written (hash mismatch)")
        try:
            _check(rec, records[: i - 1])
        except LedgerError as exc:
            problems.append(f"record {i}: {exc}")
        prev = rec.get("hash", "")
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
