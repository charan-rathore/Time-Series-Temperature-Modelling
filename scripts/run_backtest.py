#!/usr/bin/env python3
"""Run the rolling-origin backtest on the committed feature matrix and write
models/backtest.json. Baselines always run; LightGBM runs when installed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.data.preprocess import load_processed, processed_source_path
from src.evaluation.backtest import (BASELINES, backtest_errors, lgbm_forecaster,
                                     rolling_origin_folds, run_backtest)
from src.evaluation.conformal import conformal_halfwidth, walk_forward_coverage
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
    # Provenance: record the dataset actually evaluated, so registration can
    # attach these metrics to the same bytes instead of re-hashing the bundle.
    from src.models import registry as reg

    result["dataset"] = reg.dataset_provenance(processed_source_path())
    # Conformal 90% intervals: half-width from pooled backtest residuals,
    # plus walk-forward coverage measured on folds the calibration never saw.
    folds = rolling_origin_folds(len(df), result["n_folds"], result["fold_len"], result["min_train"])
    errs = backtest_errors(df.reset_index(drop=True), forecasters, (1, 2, 3), folds)
    result["conformal_90"] = {
        name: {
            f"day{h}": {
                "halfwidth_c": conformal_halfwidth([e for f in errs[name][h] for e in f], 0.1),
                **walk_forward_coverage(errs[name][h], 0.1),
            }
            for h in (1, 2, 3)
        }
        for name in forecasters
    }
    out = ROOT / "models" / "backtest.json"
    out.write_text(json.dumps(result, indent=2))
    for name, m in result["models"].items():
        d = m["day1"]
        print(f"{name:18s} day1 rmse {d['rmse']:.3f} n={d['n_observations']} "
              f"fold std {d['fold_rmse_std']:.3f} skill_vs_best_baseline "
              f"{d['skill_vs_best_baseline']:+.3f}")
    for name, c in result['conformal_90'].items():
        d = c['day1']
        print(f"{name:18s} 90% halfwidth {d['halfwidth_c']:.2f} C, walk-forward coverage {d['pooled_coverage']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
