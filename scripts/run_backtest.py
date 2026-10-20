#!/usr/bin/env python3
"""Run the rolling-origin backtest on the committed feature matrix and write
models/backtest.json. Baselines always run; LightGBM runs when installed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.data.preprocess import load_processed
from src.evaluation.backtest import BASELINES, lgbm_forecaster, run_backtest
from src.features.engineer import build_feature_matrix


def main() -> int:
    config = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    df = build_feature_matrix(load_processed(), drop_na=True)
    forecasters = dict(BASELINES)
    try:
        import lightgbm  # noqa: F401
        forecasters["lgbm"] = lgbm_forecaster(config["models"]["lgbm"])
    except Exception as exc:
        print(f"[backtest] lgbm skipped: {exc}")
    result = run_backtest(df, forecasters)
    out = ROOT / "models" / "backtest.json"
    out.write_text(json.dumps(result, indent=2))
    for name, m in result["models"].items():
        d = m["day1"]
        print(f"{name:18s} day1 rmse {d['rmse']:.3f} n={d['n_observations']} "
              f"fold std {d['fold_rmse_std']:.3f} skill_vs_best_baseline "
              f"{d['skill_vs_best_baseline']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
