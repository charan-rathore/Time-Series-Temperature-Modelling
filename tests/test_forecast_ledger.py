"""The forecast ledger refuses late, duplicate and edited records."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.evaluation import ledger as L
from scripts import ledger_daily as daily

IST = L.IST


def fc(source, target_date, h, value, issued):
    return {"kind": "forecast", "source": source, "target": L.TARGET, "target_date": target_date,
            "horizon_days": h, "value": value, "issued_at": issued}


def obs(d, value, issued, supersedes=None):
    r = {"kind": "observation", "source": L.OBS_SOURCE, "target": L.TARGET, "date": d,
         "value": value, "observed_at": issued, "issued_at": issued}
    if supersedes:
        r["supersedes"] = supersedes
    return r


def test_append_chains_and_verifies(tmp_path):
    p = tmp_path / "l.jsonl"
    a = L.append(fc("open_meteo", "2026-11-04", 1, 22.0, "2026-11-03T13:40:00Z"), p)
    b = L.append(fc("persistence", "2026-11-04", 1, 21.0, "2026-11-03T13:40:01Z"), p)
    assert a["prev"] == L.GENESIS and b["prev"] == a["hash"]
    assert L.verify(p) == []


def test_forecast_issued_on_its_target_day_is_refused(tmp_path):
    p = tmp_path / "l.jsonl"
    # 2026-11-04 00:30 IST is already the target day
    with pytest.raises(L.LedgerError, match="before its target day"):
        L.append(fc("open_meteo", "2026-11-04", 1, 22.0, "2026-11-03T19:00:00Z"), p)


def test_horizon_must_match_issue_day(tmp_path):
    with pytest.raises(L.LedgerError, match="horizon"):
        L.append(fc("open_meteo", "2026-11-05", 1, 22.0, "2026-11-03T13:40:00Z"), tmp_path / "l.jsonl")


def test_first_write_wins(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(fc("open_meteo", "2026-11-04", 1, 22.0, "2026-11-03T13:40:00Z"), p)
    with pytest.raises(L.LedgerError, match="duplicate"):
        L.append(fc("open_meteo", "2026-11-04", 1, 25.0, "2026-11-03T14:00:00Z"), p)


def test_editing_a_past_line_is_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(fc("open_meteo", "2026-11-04", 1, 22.0, "2026-11-03T13:40:00Z"), p)
    L.append(fc("persistence", "2026-11-04", 1, 21.0, "2026-11-03T13:40:01Z"), p)
    lines = p.read_text().splitlines()
    first = json.loads(lines[0])
    first["value"] = 21.5  # quietly improve a forecast
    lines[0] = json.dumps(first)
    p.write_text("\n".join(lines) + "\n")
    problems = L.verify(p)
    assert any("hash mismatch" in x for x in problems)
    with pytest.raises(L.LedgerError, match="inconsistent"):
        L.append(fc("open_meteo", "2026-11-05", 2, 22.0, "2026-11-03T13:41:00Z"), p)


def test_deleting_a_line_is_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    for i, s in enumerate(["a", "b", "c"]):
        L.append(fc(s, "2026-11-04", 1, 20.0 + i, f"2026-11-03T13:4{i}:00Z"), p)
    lines = p.read_text().splitlines()
    p.write_text(lines[0] + "\n" + lines[2] + "\n")
    assert any("chain broken" in x for x in L.verify(p))


def test_observation_is_corrected_only_by_superseding(tmp_path):
    p = tmp_path / "l.jsonl"
    o = L.append(obs("2026-11-02", 21.0, "2026-11-03T13:40:00Z"), p)
    with pytest.raises(L.LedgerError, match="already recorded"):
        L.append(obs("2026-11-02", 23.0, "2026-11-03T13:41:00Z"), p)
    L.append(obs("2026-11-02", 22.0, "2026-11-03T13:42:00Z", supersedes=o["hash"]), p)
    live = L.current_observations(L.read(p))
    assert live["2026-11-02"]["value"] == 22.0


def test_score_makes_no_claim_on_small_samples(tmp_path):
    p = tmp_path / "l.jsonl"
    t0 = datetime(2026, 9, 1, 13, 40, tzinfo=timezone.utc)
    for k in range(10):
        issued = t0 + timedelta(days=k)
        target = (issued.astimezone(IST).date() + timedelta(days=1)).isoformat()
        stamp = issued.isoformat().replace("+00:00", "Z")
        L.append(fc("open_meteo", target, 1, 22.0, stamp), p)
        L.append(fc("persistence", target, 1, 24.0, stamp), p)
    for k in range(10):
        d = (datetime(2026, 9, 2) + timedelta(days=k)).date().isoformat()
        L.append(obs(d, 22.0, "2026-10-01T00:00:00Z"), p)
    rows = {r["source"]: r for r in L.score(L.read(p))["rows"]}
    assert rows["open_meteo"]["n"] == 10 and rows["open_meteo"]["mae"] == 0
    assert rows["open_meteo"]["claim"].startswith("no claim")
    rows = {r["source"]: r for r in L.score(L.read(p), min_n=10)["rows"]}
    assert rows["open_meteo"]["claim"] == "better than baseline"


def test_daily_plan_records_only_what_was_known(tmp_path):
    db = tmp_path / "f.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE daily_actuals (date TEXT PRIMARY KEY, sensor_temp_c REAL, recorded_at TEXT, source TEXT)")
    conn.executemany("INSERT INTO daily_actuals VALUES (?,?,?,?)", [
        ("2026-11-01", 21.0, "2026-11-01T15:17:00Z", "airport_vobg_metar"),
        ("2026-11-02", 22.0, "2026-11-02T15:17:00Z", "airport_vobg_metar"),
        ("2026-11-03", 23.0, "2026-11-03T15:17:00Z", "airport_vobg_metar"),  # today: not yet final
        ("2026-10-31", 30.0, "2026-10-31T15:17:00Z", "sensor"),
    ])
    conn.commit(); conn.close()
    now = datetime(2026, 11, 3, 13, 40, tzinfo=timezone.utc)  # 19:10 IST
    om = {"hourly": {"time": ["2026-11-04T21:00", "2026-11-05T21:00", "2026-11-06T21:00"],
                     "temperature_2m": [21.4, 21.9, None]}}
    new = daily.plan(now, [], daily.airport_observations(db), om)
    kinds = [(r["kind"], r["source"], r.get("date") or r["target_date"]) for r in new]
    assert ("observation", "airport_vobg_metar", "2026-11-03") not in kinds
    assert ("observation", "airport_vobg_metar", "2026-11-02") in kinds
    assert [r["value"] for r in new if r["source"] == "persistence"] == [22.0, 22.0, 22.0]
    assert [r["target_date"] for r in new if r["source"] == "open_meteo"] == ["2026-11-04", "2026-11-05"]
    p = tmp_path / "l.jsonl"
    for r in new:
        L.append(r, p)
    assert L.verify(p) == []
    # a second run the same day adds nothing new
    again = daily.plan(now + timedelta(minutes=5), L.read(p), daily.airport_observations(db), om)
    for r in again:
        with pytest.raises(L.LedgerError):
            L.append(r, p)


def test_daily_plan_supersedes_corrected_upstream_observation(tmp_path):
    db = tmp_path / "f.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE daily_actuals (date TEXT PRIMARY KEY, sensor_temp_c REAL, recorded_at TEXT, source TEXT)")
    conn.executemany("INSERT INTO daily_actuals VALUES (?,?,?,?)", [
        ("2026-11-02", 22.6, "2026-11-02T16:05:00Z", "airport_vobg_metar"),  # corrected re-issue
    ])
    conn.commit(); conn.close()
    p = tmp_path / "l.jsonl"
    original = L.append(obs("2026-11-02", 22.0, "2026-11-02T15:42:00Z"), p)
    now = datetime(2026, 11, 3, 13, 40, tzinfo=timezone.utc)
    new = daily.plan(now, L.read(p), daily.airport_observations(db), None)
    corr = [r for r in new if r["kind"] == "observation"]
    assert len(corr) == 1
    assert corr[0]["value"] == 22.6 and corr[0]["supersedes"] == original["hash"]
    L.append(corr[0], p)
    assert L.verify(p) == []
    live = L.current_observations(L.read(p))
    assert live["2026-11-02"]["value"] == 22.6
    # an unchanged re-issue on the next run adds nothing
    again = daily.plan(now + timedelta(minutes=5), L.read(p), daily.airport_observations(db), None)
    assert [r for r in again if r["kind"] == "observation"] == []


def test_daily_plan_supersession_chains_corrections_of_corrections(tmp_path):
    db = tmp_path / "f.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE daily_actuals (date TEXT PRIMARY KEY, sensor_temp_c REAL, recorded_at TEXT, source TEXT)")
    p = tmp_path / "l.jsonl"
    first = L.append(obs("2026-11-02", 22.0, "2026-11-02T15:42:00Z"), p)

    def run(value, stamp):
        conn.execute("UPDATE daily_actuals SET sensor_temp_c=?, recorded_at=?", (value, stamp))
        conn.commit()
        now = datetime(2026, 11, 3, 13, 40, tzinfo=timezone.utc)
        new = daily.plan(now, L.read(p), daily.airport_observations(db), None)
        corr = [r for r in new if r["kind"] == "observation"]
        return [L.append(r, p) for r in corr]

    conn.execute("INSERT INTO daily_actuals VALUES (?,?,?,?)", ("2026-11-02", 22.6, "2026-11-02T16:05:00Z", "airport_vobg_metar"))
    conn.commit()
    corr1 = run(22.6, "2026-11-02T16:05:00Z")
    assert len(corr1) == 1 and corr1[0]["supersedes"] == first["hash"]
    corr2 = run(23.1, "2026-11-02T17:40:00Z")  # correction of the correction
    assert len(corr2) == 1 and corr2[0]["value"] == 23.1
    assert corr2[0]["supersedes"] == corr1[0]["hash"]
    assert L.verify(p) == []
    live = L.current_observations(L.read(p))
    assert live["2026-11-02"]["value"] == 23.1
    conn.close()
