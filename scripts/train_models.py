#!/usr/bin/env python3
"""
ThermoSense - Model Training & Evaluation Pipeline

Trains all models (SARIMA, LightGBM, TFT, Ensemble) on the processed data,
evaluates them with time-series cross-validation, logs results to MLflow,
and saves the best models to models/ for API serving.

Usage:
    python scripts/train_models.py                    # train all models
    python scripts/train_models.py --models sarima lgbm  # train specific models
    python scripts/train_models.py --skip-tft         # skip TFT (needs pytorch)

Prerequisites:
    1. Run the data pipeline first:
       python scripts/run_pipeline.py --mode backfill
    2. Ensure requirements are installed:
       pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.preprocess import load_processed
from src.evaluation.metrics import evaluate_all, compare_models
from src.features.engineer import build_feature_matrix
from src.models.sarima_model import SARIMAXModel
from src.models.lgbm_model import LGBMForecastModel
from src.models.ensemble import EnsembleStacker

MODELS_DIR = _PROJECT_ROOT / "models"
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    from src.models.tft_model import TFTModel
    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False


def load_config() -> dict:
    import yaml
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data, engineer features, and split into train/val/test."""
    processed = load_processed()
    features = build_feature_matrix(processed, drop_na=True)

    features_path = FEATURES_DIR / "feature_matrix.parquet"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False)
    print(f"[train] Feature matrix: {features.shape} saved to {features_path.name}")

    test_days = config["evaluation"]["test_split_days"]
    val_days = test_days

    n = len(features)
    test_start = n - test_days
    val_start = test_start - val_days

    train_df = features.iloc[:val_start].copy()
    val_df = features.iloc[val_start:test_start].copy()
    test_df = features.iloc[test_start:].copy()

    print(f"[train] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def rolling_sarima_forecasts(
    model: SARIMAXModel,
    test_df: pd.DataFrame,
    horizons: list,
) -> Dict[int, tuple]:
    """h-day-ahead SARIMA forecasts for every held-out day that has a full lead.

    The forecast for target day d at horizon h is made from the history up to
    and including day d - h, which is the serving situation and the same
    pairing as LightGBM (row t predicts day t + h). After each origin the
    actual observation is fed to a copy of the fitted model, so the saved
    model is not touched. Returns {h: (pred, lower, upper)}, each an array
    over the test days with NaN for the first h days, so n = len(test) - h
    observations are scored per horizon (the old score used n = 1).
    """
    fitted = copy.deepcopy(model.model)
    exog = model._get_exog(test_df)
    y = test_df["temp_c"].values
    n = len(y)
    out = {h: tuple(np.full(n, np.nan) for _ in range(3)) for h in horizons}
    for t in range(n):
        # History now ends with test day t, exactly like a LightGBM row t.
        fitted.update(y[t:t + 1], X=exog[t:t + 1] if exog is not None else None)
        for h in horizons:
            if t + h >= n:
                continue
            fc, ci = fitted.predict(
                n_periods=h,
                X=exog[t + 1:t + 1 + h] if exog is not None else None,
                return_conf_int=True,
                alpha=0.2,
            )
            pred, lower, upper = out[h]
            pred[t + h] = fc[-1]
            lower[t + h] = ci[-1, 0]
            upper[t + h] = ci[-1, 1]
    return out


def train_sarima(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Train and evaluate SARIMA model."""
    print("\n" + "=" * 60)
    print("  Training SARIMA(X)")
    print("=" * 60)

    sarima_cfg = config["models"]["sarima"]
    model = SARIMAXModel(sarima_cfg)

    full_train = pd.concat([train_df, val_df], ignore_index=True)
    model.fit(full_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODELS_DIR / "sarima.pkl"))

    horizons = [h for h in (1, 2, 3) if h <= len(test_df)]
    forecasts = rolling_sarima_forecasts(model, test_df, horizons)
    actuals_all = test_df["temp_c"].values

    results = {}
    for horizon in horizons:
        preds, lower, upper = forecasts[horizon]
        scored = ~np.isnan(preds)
        metrics = evaluate_all(actuals_all[scored], preds[scored], lower[scored], upper[scored])
        results[f"day{horizon}"] = metrics
        print(f"  Day-{horizon}: RMSE={metrics['rmse']:.3f}\u00b0C, "
              f"MAE={metrics['mae']:.3f}\u00b0C (n={int(scored.sum())} held-out days)")

    return results


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Train and evaluate LightGBM models (one per horizon)."""
    print("\n" + "=" * 60)
    print("  Training LightGBM")
    print("=" * 60)

    lgbm_cfg = config["models"]["lgbm"]
    results = {}

    for horizon in [1, 2, 3]:
        model = LGBMForecastModel(lgbm_cfg, horizon=horizon)
        model.fit(train_df, val_df=val_df)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODELS_DIR / f"lgbm_h{horizon}.pkl"))

        preds = model.predict(steps=len(test_df), future_df=test_df)
        # preds[t] is the forecast for day t + horizon, so it is paired with
        # the actual of that day: n = len(test_df) - horizon observations.
        n_pairs = min(len(test_df) - horizon, len(preds))
        preds = preds[:n_pairs]
        actuals = test_df["temp_c"].values[horizon:horizon + n_pairs]

        metrics = evaluate_all(actuals, preds)
        results[f"day{horizon}"] = metrics
        print(f"  Horizon-{horizon}: RMSE={metrics['rmse']:.3f}°C, "
              f"MAE={metrics['mae']:.3f}°C")

        imp = model.feature_importance()
        print(f"  Top 5 features: {list(imp.head(5).index)}")

    return results


def _prepare_tft_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time_idx and location columns exist and time_idx is contiguous."""
    out = df.copy().reset_index(drop=True)
    out["time_idx"] = range(len(out))
    if "location" not in out.columns:
        out["location"] = "sensor"
    return out


def train_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Train and evaluate TFT model."""
    if not _TFT_AVAILABLE:
        print("\n[train] TFT unavailable (pytorch-forecasting not installed). Skipping.")
        return {}

    print("\n" + "=" * 60)
    print("  Training Temporal Fusion Transformer")
    print("=" * 60)

    tft_cfg = config["models"]["tft"]
    model = TFTModel(tft_cfg)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df = _prepare_tft_data(all_df)

    max_encoder = tft_cfg.get("max_encoder_length", 30)
    max_pred = tft_cfg.get("max_prediction_length", 3)

    n_train = len(train_df)
    n_val = len(val_df)

    if n_train < max_encoder + max_pred:
        print(f"  [TFT] Not enough training data ({n_train} rows, need {max_encoder + max_pred}). Skipping.")
        return {}

    tft_train = all_df.iloc[:n_train].copy()
    tft_val = all_df.iloc[:n_train + n_val].copy()

    try:
        model.fit(tft_train, val_df=tft_val)
    except Exception as e:
        print(f"  [TFT] Training failed: {e}")
        return {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODELS_DIR / "tft.pkl"))

    try:
        tft_full = all_df.copy()
        intervals = model.predict_intervals(steps=3, future_df=tft_full)
    except Exception as e:
        print(f"  [TFT] Prediction failed: {e}")
        return {}

    results = {}
    for horizon in [1, 2, 3]:
        if horizon - 1 < len(intervals["median"]):
            pred = intervals["median"][horizon - 1]
            actual = test_df["temp_c"].values[horizon - 1] if horizon <= len(test_df) else None
            if actual is not None:
                metrics = evaluate_all(
                    np.array([actual]),
                    np.array([pred]),
                    np.array([intervals["lower"][horizon - 1]]),
                    np.array([intervals["upper"][horizon - 1]]),
                )
                results[f"day{horizon}"] = metrics
                print(f"  Day-{horizon}: RMSE={metrics['rmse']:.3f}°C")

    return results


def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    sarima_results: dict,
    lgbm_results: dict,
    tft_results: dict,
) -> dict:
    """Train ensemble stacker on validation-period predictions only.

    Honesty boundary: the stacker learns its weights from validation-period
    predictions made by base models trained on the training split alone.
    The held-out test window is touched only afterwards, to evaluate the
    frozen stacker over every test day (n = len(test_df) per horizon).
    Fitting the stacker on test-period predictions, as this function once
    did, makes the reported "test" metrics in-sample and meaningless.
    """
    print("\n" + "=" * 60)
    print("  Training Ensemble Stacker")
    print("=" * 60)

    ensemble_cfg = config["models"]["ensemble"]
    sarima_cfg = config["models"]["sarima"]
    lgbm_cfg = config["models"]["lgbm"]

    # TFT is not a stacker input here: base_predictions() below only produces
    # SARIMA and LightGBM predictions, so listing "tft" would raise KeyError.
    available_base_models = ["sarima", "lgbm"]

    max_h = min(3, len(val_df), len(test_df))
    if len(val_df) <= max_h or len(test_df) <= max_h:
        print("  Skipping ensemble: validation/test windows too short to align horizons.")
        return {}
    horizons = list(range(1, max_h + 1))

    def base_predictions(sarima_fit_df: pd.DataFrame, lgbm_fit_df: pd.DataFrame,
                         lgbm_es_df: Optional[pd.DataFrame],
                         target_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict with freshly trained base models, aligned by TARGET day.

        SARIMA is fitted on sarima_fit_df. LightGBM is fitted on lgbm_fit_df
        and early-stopped on lgbm_es_df, which must never contain the labels
        of the rows being predicted.

        Row d of every returned array describes target day d, so
        target_df["temp_c"][d] is its label. A horizon-h LightGBM model
        predicts day t + h from features at row t, so its output is shifted
        by h (NaN for the first h days) before it is stacked with SARIMA
        (whose n-step forecast is already indexed by target day).
        """
        n = len(target_df)
        sarima_model = SARIMAXModel(sarima_cfg)
        sarima_model.fit(sarima_fit_df)
        p = sarima_model.predict(steps=n, future_df=target_df)
        sarima_full = np.zeros(n)
        sarima_full[: len(p)] = p[:n]
        sarima_cols = np.column_stack([sarima_full] * len(horizons))

        lgbm_cols = np.full((n, len(horizons)), np.nan)
        for h_idx, h in enumerate(horizons):
            m = LGBMForecastModel(lgbm_cfg, horizon=h)
            m.fit(lgbm_fit_df, val_df=lgbm_es_df)
            p = m.predict(steps=n, future_df=target_df)
            # p[t] targets day t + h, so target day d is p[d - h] (d >= h).
            for d in range(h, n):
                if d - h < len(p):
                    lgbm_cols[d, h_idx] = p[d - h]
        return {"sarima": sarima_cols, "lgbm": lgbm_cols}

    # 1) Fit the stacker on validation-period predictions (test untouched).
    #    The base models see the training split only. LightGBM early-stops on
    #    the tail of that split: early-stopping on val_df would pick the
    #    boosting round using the very labels the stacker then learns from.
    es_rows = max(max_h + 1, len(train_df) // 5)
    val_preds = base_predictions(
        train_df, train_df.iloc[:-es_rows], train_df.iloc[-es_rows:], val_df,
    )
    val_preds = {name: arr[max_h:] for name, arr in val_preds.items()}
    val_actuals = val_df["temp_c"].values[max_h:]
    val_actuals_2d = np.column_stack([val_actuals] * len(horizons))

    ensemble_cfg_with_models = {**ensemble_cfg, "base_models": available_base_models}
    stacker = EnsembleStacker(ensemble_cfg_with_models)
    stacker.fit_from_oof(
        {name: val_preds[name] for name in available_base_models},
        val_actuals_2d,
        horizons=horizons,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stacker.save(str(MODELS_DIR / "ensemble.pkl"))

    # 2) Evaluate with the base models that are served: SARIMA fitted on
    #    train+val (train_sarima) and LightGBM fitted on train with val for
    #    early stopping (train_lgbm). The frozen stacker is scored on every
    #    held-out test day that has all horizons available.
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    test_preds = base_predictions(full_train, train_df, val_df, test_df)
    results = {}
    for h_idx, h in enumerate(horizons):
        # Horizon h pairs the prediction for target day d with the actual of
        # day d, for d = h .. n - 1, so n = len(test_df) - h observations.
        X_meta = np.column_stack([
            test_preds[name][h:, h_idx] for name in available_base_models
        ])
        actuals = test_df["temp_c"].values[h:]
        ens_preds = stacker.meta_models[h].predict(X_meta)
        metrics = evaluate_all(actuals, ens_preds)
        results[f"day{h}"] = metrics
        print(f"  Day-{h}: RMSE={metrics['rmse']:.3f}\u00b0C, MAE={metrics['mae']:.3f}\u00b0C "
              f"(n={len(actuals)} held-out days)")

    return results


def log_to_mlflow(
    all_results: dict,
    config: dict,
) -> None:
    """Log all results to MLflow."""
    if not _MLFLOW_AVAILABLE:
        print("\n[train] MLflow not available. Skipping experiment tracking.")
        return

    print("\n" + "=" * 60)
    print("  Logging to MLflow")
    print("=" * 60)

    mlflow.set_tracking_uri(str(_PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("thermosense")

    for model_name, horizons in all_results.items():
        if not horizons:
            continue
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param("model", model_name)
            model_cfg = config["models"].get(model_name, {})
            for k, v in model_cfg.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

            for horizon_key, metrics in horizons.items():
                for metric_name, value in metrics.items():
                    key = f"{horizon_key}_{metric_name}"
                    if isinstance(value, str) and metric_name.endswith("_status"):
                        mlflow.set_tag(key, value)
                    elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value):
                        mlflow.log_metric(key, float(value))
                    # Undefined metrics are None, not zero. Do not log them as
                    # numeric measurements or allow nonfinite values into MLflow.

            model_path = MODELS_DIR / f"{model_name}.pkl"
            if model_path.exists():
                mlflow.log_artifact(str(model_path))

    print("  Results logged to MLflow. Run `mlflow ui --port 5000` to view.")


def print_comparison_table(all_results: dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Day-1 RMSE':>12} {'Day-2 RMSE':>12} {'Day-3 RMSE':>12} "
          f"{'Day-1 MAE':>12} {'Skill':>8}")
    print("-" * 75)

    for model_name, horizons in all_results.items():
        if not horizons:
            continue
        d1 = horizons.get("day1", {})
        d2 = horizons.get("day2", {})
        d3 = horizons.get("day3", {})
        print(f"{model_name:<15} "
              f"{d1.get('rmse', float('nan')):>12.3f} "
              f"{d2.get('rmse', float('nan')):>12.3f} "
              f"{d3.get('rmse', float('nan')):>12.3f} "
              f"{d1.get('mae', float('nan')):>12.3f} "
              f"{(d1.get('skill_score') if d1.get('skill_score') is not None else float('nan')):>8.3f}")

    results_path = MODELS_DIR / "results.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="ThermoSense Model Training")
    parser.add_argument(
        "--models", nargs="+", default=["sarima", "lgbm", "ensemble"],
        choices=["sarima", "lgbm", "tft", "ensemble", "all"],
        help="Models to train (default: sarima lgbm ensemble)",
    )
    parser.add_argument("--skip-tft", action="store_true", help="Skip TFT training")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    if "all" in args.models:
        args.models = ["sarima", "lgbm", "tft", "ensemble"]

    if args.skip_tft and "tft" in args.models:
        args.models.remove("tft")

    print("=" * 60)
    print("  ThermoSense - Model Training Pipeline")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    config = load_config()
    train_df, val_df, test_df = prepare_data(config)

    all_results = {}

    sarima_results = {}
    if "sarima" in args.models:
        sarima_results = train_sarima(train_df, val_df, test_df, config)
        all_results["sarima"] = sarima_results

    lgbm_results = {}
    if "lgbm" in args.models:
        lgbm_results = train_lgbm(train_df, val_df, test_df, config)
        all_results["lgbm"] = lgbm_results

    tft_results = {}
    if "tft" in args.models:
        tft_results = train_tft(train_df, val_df, test_df, config)
        all_results["tft"] = tft_results

    if "ensemble" in args.models:
        ens_results = train_ensemble(
            train_df, val_df, test_df, config,
            sarima_results, lgbm_results, tft_results,
        )
        all_results["ensemble"] = ens_results

    print_comparison_table(all_results)

    if not args.no_mlflow:
        log_to_mlflow(all_results, config)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Models saved to: {MODELS_DIR}")
    print("  Next: start the API with `uvicorn src.api.main:app --reload`")
    print("=" * 60)


if __name__ == "__main__":
    main()
