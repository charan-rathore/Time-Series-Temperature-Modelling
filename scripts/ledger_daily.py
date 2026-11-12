"""Daily forecast-ledger run (see src/evaluation/ledger.py).

Appends, in this order:
  1. observations: 21:00 IST airport readings (VOBG METAR, already polled
     hourly into data/baselines/forecasts.db) for past dates not yet in the
     ledger;
  2. persistence: "the 21:00 temperature D+h days ahead equals the last
     observed 21:00 temperature", for h = 1..3;
  3. open_meteo: Open-Meteo's FORECAST API (not its archive), hourly
     temperature_2m at 21:00 IST, for D+1..D+3.

Nothing is backfilled for forecasts: each is stamped with the time it was
fetched, and the ledger refuses one issued on or after its target day.
ThermoSense's own models predict the daily mean, a different quantity, so
they do not enter this ledger until they are trained on the 21:00 target.

Usage:
    python scripts/ledger_daily.py            # append today's records
    python scripts/ledger_daily.py --dry-run  # show what would be appended
    python scripts/ledger_daily.py --score    # print the scoreboard
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.ledger import (  # noqa: E402
    IST,
    LEDGER_PATH,
    OBS_SOURCE,
    TARGET,
    LedgerError,
    append,
    current_observations,
    read,
    score,
    verify,
)

DB_PATH = ROOT / "data" / "baselines" / "forecasts.db"
LAT, LON = 12.9716, 77.5946  # config/config.yaml location
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def airport_observations(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT date, sensor_temp_c, recorded_at FROM daily_actuals WHERE source = ? ORDER BY date",
            (OBS_SOURCE,),
        ).fetchall()
    finally:
        conn.close()
    return [{"date": d, "value": float(v), "observed_at": r} for d, v, r in rows]


def fetch_open_meteo(get: Callable[..., Any]) -> Dict[str, Any]:
    r = get(
        OPEN_METEO,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m",
            "timezone": "Asia/Kolkata",
            "forecast_days": 4,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def plan(now: datetime, records: List[Dict[str, Any]], observations: List[Dict[str, Any]], om: Optional[Dict[str, Any]]):
    today = now.astimezone(IST).date()
    stamp = iso(now)
    out: List[Dict[str, Any]] = []
    live_obs = current_observations(records, TARGET, OBS_SOURCE)
    for o in observations:
        # only days whose 21:00 IST window has passed
        if o["date"] >= today.isoformat():
            continue
        existing = live_obs.get(o["date"])
        if existing is not None:
            # Upstream corrections (a re-issued METAR for an already-recorded
            # date) must not be silently skipped: append a superseding record
            # when the value changed, naming the live record it replaces.
            if o["value"] != existing["value"] and existing.get("hash"):
                out.append({"kind": "observation", "source": OBS_SOURCE, "target": TARGET, "date": o["date"],
                            "value": o["value"], "observed_at": o["observed_at"], "issued_at": stamp,
                            "supersedes": existing["hash"]})
            continue
        out.append({"kind": "observation", "source": OBS_SOURCE, "target": TARGET, "date": o["date"],
                    "value": o["value"], "observed_at": o["observed_at"], "issued_at": stamp})
    live = current_observations(records + out, TARGET, OBS_SOURCE)
    if live:
        last_date = max(live)
        age = (today - datetime.fromisoformat(last_date).date()).days
        if age <= 2:
            for h in (1, 2, 3):
                out.append({"kind": "forecast", "source": "persistence", "target": TARGET,
                            "target_date": (today + timedelta(days=h)).isoformat(), "horizon_days": h,
                            "value": live[last_date]["value"], "issued_at": stamp,
                            "inputs": {"observation_date": last_date}})
    if om:
        times = om.get("hourly", {}).get("time", [])
        temps = om.get("hourly", {}).get("temperature_2m", [])
        by_time = dict(zip(times, temps))
        for h in (1, 2, 3):
            d = (today + timedelta(days=h)).isoformat()
            v = by_time.get(f"{d}T21:00")
            if isinstance(v, (int, float)):
                out.append({"kind": "forecast", "source": "open_meteo", "target": TARGET, "target_date": d,
                            "horizon_days": h, "value": float(v), "issued_at": stamp,
                            "inputs": {"api": "forecast", "hourly": "temperature_2m", "local_time": "21:00",
                                       "generationtime_ms": om.get("generationtime_ms")}})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args(argv)

    problems = verify(args.ledger)
    if problems:
        print("LEDGER INCONSISTENT:", *problems, sep="\n  ")
        return 1
    records = read(args.ledger)
    if args.score:
        print(json.dumps(score(records), indent=2))
        return 0

    om = None
    try:
        import requests

        om = fetch_open_meteo(requests.get)
    except Exception as exc:  # network or HTTP: record nothing for this source
        print(f"open_meteo: not fetched ({type(exc).__name__}); nothing recorded for it today")
    new = plan(datetime.now(timezone.utc), records, airport_observations(args.db), om)
    for rec in new:
        label = rec.get("target_date") or rec.get("date")
        if args.dry_run:
            print("would append", rec["kind"], rec["source"], label, rec["value"])
            continue
        try:
            append(rec, args.ledger)
            print("appended", rec["kind"], rec["source"], label, rec["value"])
        except LedgerError as exc:
            print("refused", rec["kind"], rec["source"], label, "-", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
