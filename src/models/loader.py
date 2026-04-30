"""
Model loader for ThermoSense API.

Loads trained models from the models/ directory at startup and provides
a unified interface for the API routes to generate forecasts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_forecast_open_meteo
from src.data.preprocess import load_processed
from src.features.engineer import build_feature_matrix
from src.models.sarima_model import SARIMAXModel
from src.models.lgbm_model import LGBMForecastModel
from src.models.ensemble import EnsembleStacker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _PROJECT_ROOT / "models"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


class ModelManager:
    """
    Manages loaded models and generates forecasts for the API layer.
    Thread-safe for use in a FastAPI application.
    """

    def __init__(self):
        self.sarima: Optional[SARIMAXModel] = None
        self.lgbm_models: Dict[int, LGBMForecastModel] = {}
        self.ensemble: Optional[EnsembleStacker] = None
        self.is_loaded = False
        self._results: Dict = {}

    def load_models(self, config: dict) -> None:
        """Load all available trained models from disk."""
        loaded = []

        sarima_path = MODELS_DIR / "sarima.pkl"
        if sarima_path.exists():
            self.sarima = SARIMAXModel(config.get("models", {}).get("sarima", {}))
            self.sarima.load(str(sarima_path))
            loaded.append("sarima")

        lgbm_cfg = config.get("models", {}).get("lgbm", {})
        for h in [1, 2, 3]:
            lgbm_path = MODELS_DIR / f"lgbm_h{h}.pkl"
            if lgbm_path.exists():
                m = LGBMForecastModel(lgbm_cfg, horizon=h)
                m.load(str(lgbm_path))
                self.lgbm_models[h] = m
                if "lgbm" not in loaded:
                    loaded.append("lgbm")

        ens_path = MODELS_DIR / "ensemble.pkl"
        if ens_path.exists():
            ens_cfg = config.get("models", {}).get("ensemble", {})
            self.ensemble = EnsembleStacker(ens_cfg)
            self.ensemble.load(str(ens_path))
            loaded.append("ensemble")

        results_path = MODELS_DIR / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                self._results = json.load(f)

        self.is_loaded = len(loaded) > 0
        print(f"[ModelManager] Loaded models: {loaded if loaded else 'none'}")

    def get_best_model_name(self) -> str:
        """Return the name of the best available model."""
        if self.ensemble and self.ensemble.is_fitted:
            return "ensemble"
        if self.lgbm_models:
            return "lgbm"
        if self.sarima and self.sarima.is_fitted:
            return "sarima"
        return "placeholder"

    def forecast(self, days: int = 3) -> List[Dict[str, Any]]:
        """
        Generate temperature forecasts for the next `days` days.
        Returns a list of dicts with predictions and intervals.
        """
        if not self.is_loaded:
            return self._placeholder_forecast(days)

        try:
            processed = load_processed()
            features = build_feature_matrix(processed, drop_na=True)
        except Exception as e:
            print(f"[ModelManager] Error loading data for forecast: {e}")
            return self._placeholder_forecast(days)

        forecasts = []
        model_name = self.get_best_model_name()

        for horizon in range(1, days + 1):
            pred, lower, upper = None, None, None

            if model_name == "ensemble" and self.ensemble:
                base_preds = {}
                if self.sarima and self.sarima.is_fitted:
                    try:
                        s_preds = self.sarima.predict(steps=horizon, future_df=features.tail(horizon))
                        base_preds["sarima"] = s_preds[-1:]
                    except Exception:
                        base_preds["sarima"] = np.array([features["temp_c"].mean()])

                if horizon in self.lgbm_models:
                    try:
                        l_preds = self.lgbm_models[horizon].predict(steps=1, future_df=features.tail(1))
                        base_preds["lgbm"] = l_preds
                    except Exception:
                        base_preds["lgbm"] = np.array([features["temp_c"].mean()])

                if base_preds:
                    for name in self.ensemble.base_model_names:
                        if name not in base_preds:
                            base_preds[name] = np.array([features["temp_c"].mean()])
                    try:
                        ens_pred = self.ensemble.predict(steps=1, base_predictions=base_preds)
                        pred = float(ens_pred[0])
                    except Exception:
                        pass

            if pred is None and horizon in self.lgbm_models:
                try:
                    l_preds = self.lgbm_models[horizon].predict(
                        steps=1, future_df=features.tail(1)
                    )
                    pred = float(l_preds[0])
                    model_name = "lgbm"
                except Exception:
                    pass

            if pred is None and self.sarima and self.sarima.is_fitted:
                try:
                    s_result = self.sarima.predict_intervals(
                        steps=horizon, future_df=features.tail(horizon)
                    )
                    pred = float(s_result["median"][-1])
                    lower = float(s_result["lower"][-1])
                    upper = float(s_result["upper"][-1])
                    model_name = "sarima"
                except Exception:
                    pass

            if pred is None:
                pred = float(features["temp_c"].iloc[-1])
                model_name = "fallback"

            if lower is None:
                lower = pred - 1.5
                upper = pred + 1.5

            forecasts.append({
                "horizon": horizon,
                "predicted_temp_c": round(pred, 2),
                "lower_bound_c": round(lower, 2),
                "upper_bound_c": round(upper, 2),
                "model_used": model_name,
            })

        return forecasts

    def get_results(self) -> Dict:
        """Return training results for the metrics endpoint."""
        return self._results

    def _placeholder_forecast(self, days: int) -> List[Dict[str, Any]]:
        """Fallback when no models are trained."""
        try:
            processed = load_processed()
            recent_temp = float(processed["temp_c"].iloc[-1])
        except Exception:
            recent_temp = 26.0

        forecasts = []
        for h in range(1, days + 1):
            forecasts.append({
                "horizon": h,
                "predicted_temp_c": round(recent_temp, 2),
                "lower_bound_c": round(recent_temp - 2.0, 2),
                "upper_bound_c": round(recent_temp + 2.0, 2),
                "model_used": "climatology",
            })
        return forecasts
