"""Serving must read data from paths Vercel actually bundles.

Vercel's Python function bundler leaves out committed files that live under
.gitignore'd directories (data/processed/, data/features/) even when
vercel.json includeFiles lists them. The API therefore serves zeros on the
live site. These tests pin the serving paths to a directory no ignore rule
touches (data/bundle/).
"""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gitignore_rules():
    rules = []
    for line in (ROOT / ".gitignore").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            rules.append(line)
    return rules


def _is_ignored(rel: str, rules) -> bool:
    for r in rules:
        if r.endswith("/") and (rel + "/").startswith(r):
            return True
        if fnmatch.fnmatch(rel, r) or fnmatch.fnmatch(Path(rel).name, r):
            return True
    return False


def _serving_paths():
    from src.api.routes import forecast, history, pipeline

    paths = {pipeline.PROCESSED_PATH, pipeline.FEATURES_PATH,
             forecast.PROCESSED_PATH, history.PROCESSED_PATH}
    return {str(p.resolve().relative_to(ROOT)) for p in paths}


def test_serving_data_paths_are_not_gitignored():
    rules = _gitignore_rules()
    for rel in _serving_paths():
        assert not _is_ignored(rel, rules), f"{rel} is covered by .gitignore"


def test_serving_data_files_are_committed():
    for rel in _serving_paths():
        assert (ROOT / rel).exists(), f"{rel} missing from the repo"


def test_refresh_bundle_writes_served_csvs(tmp_path, monkeypatch):
    """The daily/backfill bundle refresh must write the served CSVs
    (regression: cmd_backfill referenced an undefined df and crashed)."""
    import pandas as pd
    import scripts.run_pipeline as rp

    monkeypatch.setattr(rp, "_PROJECT_ROOT", tmp_path)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-18", "2026-10-19"]),
        "temp_c": [24.0, 25.0], "temp_c_api": [24.1, 25.1],
        "is_sensor_reading": [False, False], "gap_filled": [False, False],
    })
    rp._refresh_bundle(df)
    out = tmp_path / "data" / "bundle" / "daily_merged.csv"
    assert out.exists()
    assert len(pd.read_csv(out)) == 2


def test_feedback_mirrors_into_parquet_training_store(tmp_path, monkeypatch):
    """Accepted feedback must also land in the parquet store that
    training and ModelManager read."""
    import pandas as pd
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.api.routes import forecast

    csv_path = tmp_path / "bundle" / "daily_merged.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"date": [pd.Timestamp("2024-07-11")], "temp_c": [26.0],
                  "temp_c_api": [25.0], "is_sensor_reading": [False]}
                 ).to_csv(csv_path, index=False)
    monkeypatch.setattr(forecast, "PROCESSED_PATH", csv_path)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/forecast/feedback", json={"date": "2024-07-11", "actual_temp_c": 27.0})
    assert response.status_code == 200
    mirror = tmp_path / "processed" / "daily_merged.parquet"
    assert mirror.exists()
    df = pd.read_parquet(mirror)
    assert df.loc[0, "temp_c"] == 27.0
    assert bool(df.loc[0, "is_sensor_reading"])
