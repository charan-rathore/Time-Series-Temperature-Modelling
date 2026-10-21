"""A small, file-based model registry.

models/registry.json is an append-only list of versions. A version records
what produced a model (artifact hashes, training-data hash and date range,
git commit) and how it scored in the rolling backtest, so any served
forecast can be traced to the exact artifacts and data behind it.

Promotion rule (the part that is enforced, not just documented): a
challenger replaces the champion only if its pooled day-1 backtest RMSE is
lower by at least ``min_gain`` (relative) AND it is evaluated on at least as
many observations. A model that merely matches the champion does not churn
production.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def dataset_provenance(path) -> dict:
    """Hash, date range and row count of the dataset file that was actually
    evaluated (parquet or CSV), so registrations point at the real input."""
    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=["date"])
    else:
        df = pd.read_csv(p, usecols=["date"])
    dates = pd.to_datetime(df["date"]).dt.date
    try:
        shown = str(p.resolve().relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        shown = str(p)
    return {
        "path": shown,
        "sha256": file_sha256(p)[:16],
        "range": f"{dates.min()} to {dates.max()}",
        "rows": int(len(df)),
    }

def artifact_hashes(models_dir: Path, names: Iterable[str]) -> Dict[str, str]:
    return {n: file_sha256(Path(models_dir) / n)[:16] for n in sorted(names)}


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def make_version(
    model: str,
    artifacts: Dict[str, str],
    data_hash: str,
    data_range: str,
    backtest_day1: dict,
    git_sha: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict:
    """Build one registry entry. The id is a hash of artifacts + data, so the
    same inputs always produce the same version id (re-registering is a no-op)."""
    ident = hashlib.sha256(
        json.dumps([model, artifacts, data_hash], sort_keys=True).encode()
    ).hexdigest()[:12]
    return {
        "version": f"{model}-{ident}",
        "model": model,
        "artifacts": artifacts,
        "data_hash": data_hash,
        "data_range": data_range,
        "backtest_day1": {
            "rmse": backtest_day1["rmse"],
            "n_observations": backtest_day1["n_observations"],
        },
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def load(path: Path = REGISTRY_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {"versions": [], "champions": {}}
    return json.loads(p.read_text())


def save(registry: dict, path: Path = REGISTRY_PATH) -> None:
    Path(path).write_text(json.dumps(registry, indent=2, sort_keys=True))


def should_promote(champion: Optional[dict], challenger: dict, min_gain: float = 0.02) -> bool:
    if champion is None:
        return True
    c, n = champion["backtest_day1"], challenger["backtest_day1"]
    if n["n_observations"] < c["n_observations"]:
        return False                      # less evidence than the incumbent
    return n["rmse"] <= c["rmse"] * (1 - min_gain)


def register(registry: dict, entry: dict, min_gain: float = 0.02) -> dict:
    """Append ``entry`` (idempotent on version id) and promote it if it wins.
    Returns {"registered": bool, "promoted": bool, "champion": version id}."""
    if any(v["version"] == entry["version"] for v in registry["versions"]):
        return {"registered": False, "promoted": False,
                "champion": registry["champions"].get(entry["model"])}
    registry["versions"].append(entry)
    current_id = registry["champions"].get(entry["model"])
    current = next((v for v in registry["versions"] if v["version"] == current_id), None)
    promoted = should_promote(current, entry, min_gain)
    if promoted:
        registry["champions"][entry["model"]] = entry["version"]
    return {"registered": True, "promoted": promoted,
            "champion": registry["champions"][entry["model"]]}


def champion_version(model: str, path: Path = REGISTRY_PATH) -> Optional[str]:
    return load(path)["champions"].get(model)
