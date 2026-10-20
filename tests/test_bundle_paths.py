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
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


def _is_ignored(rel: str, rules) -> bool:
    """Last matching gitignore rule wins, including ! negations."""
    ignored = False
    for r in rules:
        neg = r.startswith("!")
        pat = r[1:] if neg else r
        if pat.endswith("/") and (rel + "/").startswith(pat):
            ignored = not neg
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
            ignored = not neg
    return ignored


def _serving_paths():
    from src.api.routes import forecast, history, pipeline

    paths = {pipeline.PROCESSED_PATH, pipeline.FEATURES_PATH,
             forecast.PROCESSED_PATH, history.PROCESSED_PATH}
    return {str(p.resolve().relative_to(ROOT)) for p in paths}


def _deployed_artifacts():
    """Everything the live function serves that must survive bundling."""
    artifacts = set(_serving_paths())
    artifacts.add("data/baselines/forecasts.db")
    for pattern in ("*.pkl", "*.ckpt"):
        for p in (ROOT / "models").glob(pattern):
            artifacts.add(str(p.relative_to(ROOT)))
    artifacts.add("models/results.json")
    return artifacts


def test_deployed_artifacts_are_not_gitignored():
    rules = _gitignore_rules()
    for rel in _deployed_artifacts():
        assert not _is_ignored(rel, rules), f"{rel} is covered by .gitignore"


def test_deployed_artifacts_are_committed():
    for rel in _deployed_artifacts():
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


def test_load_processed_falls_back_to_bundle_csv(tmp_path, monkeypatch):
    """Inference must work where the parquet is absent (Vercel bundle):
    load_processed() falls back to data/bundle/daily_merged.csv."""
    import pandas as pd
    from src.data import preprocess

    bundle = tmp_path / "data" / "bundle"
    bundle.mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-10-18", "2026-10-19"]),
                  "temp_c": [24.0, 25.0]}).to_csv(bundle / "daily_merged.csv", index=False)
    monkeypatch.setattr(preprocess, "_PROJECT_ROOT", tmp_path)
    df = preprocess.load_processed(str(tmp_path / "data" / "processed" / "daily_merged.parquet"))
    assert len(df) == 2
    assert list(df["temp_c"]) == [24.0, 25.0]


def test_daily_mode_reads_the_bundle_when_parquet_is_absent(tmp_path, monkeypatch, capsys):
    """/api/pipeline/daily must not die just because the gitignored
    parquet directory is missing while the served CSV bundle exists."""
    import pandas as pd
    import scripts.run_pipeline as rp
    from src.data import preprocess

    bundle = tmp_path / "data" / "bundle"
    bundle.mkdir(parents=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-18", "2026-10-19"]),
        "temp_c": [24.0, 25.0], "is_sensor_reading": [False, False],
        "gap_filled": [False, False],
    }).to_csv(bundle / "daily_merged.csv", index=False)
    monkeypatch.setattr(rp, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "_PROJECT_ROOT", tmp_path)

    import pandas as pd
    from src.data import fetcher
    monkeypatch.setattr(
        fetcher, "fetch_historical_open_meteo",
        lambda start_date, end_date: pd.DataFrame(
            {"date": pd.to_datetime([end_date]), "temp_c": [25.0]}))
    monkeypatch.setattr(
        rp, "fetch_forecast_open_meteo",
        lambda forecast_days=7: pd.DataFrame({"date": ["2026-10-20"], "temp_c": [25.0]}))
    monkeypatch.setattr(rp, "fetch_owm_current", lambda: None)

    class Args:
        mode = "daily"
    try:
        rp.cmd_daily(Args())
    except SystemExit as exc:
        raise AssertionError(f"cmd_daily exited despite the bundle: {exc}")


def test_committed_bundle_keeps_sensor_rows_with_covariates():
    """Sensor observations must survive into the served feature matrix:
    the model is supposed to learn the local sensor bias (api_bias), which
    is impossible if sensor rows lose their API covariates and get dropped.
    """
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "bundle" / "daily_merged.csv", parse_dates=["date"])
    sensor = df[df["is_sensor_reading"] == True]
    assert len(sensor) > 0, "no sensor rows in the committed data at all"
    assert sensor["temp_c_api"].notna().all(), "sensor rows missing API covariates"

    f = pd.read_csv(ROOT / "data" / "bundle" / "feature_matrix.csv")
    kept = f[f["is_sensor_reading"] == True]
    assert len(kept) > 0, "feature matrix dropped every sensor row"
    assert (kept["api_bias"].abs() > 0).any(), "api_bias is flat zero on sensor rows"
