#!/usr/bin/env python3
"""Register the committed artifacts in models/registry.json using the latest
models/backtest.json (run scripts/run_backtest.py first)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import registry as reg

ARTIFACTS = {"lgbm": ["lgbm_h1.pkl", "lgbm_h2.pkl", "lgbm_h3.pkl"], "sarima": ["sarima.pkl"]}


def main() -> int:
    bt_full = json.loads((ROOT / "models" / "backtest.json").read_text())
    bt = bt_full["models"]
    # Use the dataset the backtest actually evaluated, never an independently
    # hashed file that may have drifted from it.
    ds = bt_full.get("dataset")
    if not ds:
        raise SystemExit("models/backtest.json has no dataset provenance; "
                         "re-run scripts/run_backtest.py first")
    data_hash, data_range = ds["sha256"], ds["range"]
    registry = reg.load()
    for model, files in ARTIFACTS.items():
        if model not in bt:
            print(f"[registry] {model}: no backtest result, skipped")
            continue
        entry = reg.make_version(model, reg.artifact_hashes(ROOT / "models", files),
                                 data_hash, data_range, bt[model]["day1"])
        print(model, reg.register(registry, entry))
    reg.save(registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
