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

try:
    from src.models.sarima_model import SARIMAXModel
    _SARIMA_AVAILABLE = True
except Exception as exc:  # ImportError / missing native libs
    SARIMAXModel = None  # type: ignore[misc, assignment]
    _SARIMA_AVAILABLE = False
    print(f"[ModelManager] SARIMA unavailable: {exc}")

try:
    from src.models.lgbm_model import LGBMForecastModel
    _LGBM_AVAILABLE = True
except Exception as exc:  # ImportError / libgomp missing on some hosts
    LGBMForecastModel = None  # type: ignore[misc, assignment]
    _LGBM_AVAILABLE = False
    print(f"[ModelManager] LightGBM unavailable: {exc}")

try:
    from src.models.ensemble import EnsembleStacker
    _ENSEMBLE_AVAILABLE = True
except Exception as exc:
    EnsembleStacker = None  # type: ignore[misc, assignment]
    _ENSEMBLE_AVAILABLE = False
    print(f"[ModelManager] Ensemble unavailable: {exc}")

try:
    from src.models.tft_model import TFTModel
    _TFT_AVAILABLE = True
except Exception:
    _TFT_AVAILABLE = False

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
        self.tft = None
        self.ensemble: Optional[EnsembleStacker] = None
        self.is_loaded = False
        self._results: Dict = {}

    def load_models(self, config: dict) -> None:
        """Load all available trained models from disk."""
        loaded = []

        # Legacy SARIMA/LightGBM artifacts have no validated issue-time
        # provenance or walk-forward evaluation. Do not load them for serving.
        # TFT artifacts are also withheld until prospective sensor-backed
        # evaluation. Loading a checkpoint is not evidence of skill.

        # Old ensemble artifacts were trained with held-out test labels.
        # Do not serve them until a clean out-of-fold stacker is implemented.

        results_path = MODELS_DIR / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                self._results = json.load(f)
                self._results.pop("ensemble", None)
                self._results.pop("lgbm", None)
                self._results.pop("sarima", None)

        self.is_loaded = len(loaded) > 0
        print(f"[ModelManager] Loaded models: {loaded if loaded else 'none'}")

    def get_best_model_name(self) -> str:
        """Return the name of the best available model."""
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

                if self.tft and self.tft.is_fitted:
                    try:
                        tft_result = self.tft.predict_intervals(steps=horizon, future_df=features)
                        base_preds["tft"] = np.array([float(tft_result["median"][horizon - 1])])
                    except Exception:
                        base_preds["tft"] = np.array([features["temp_c"].mean()])

                if base_preds:
                    for name in self.ensemble.base_model_names:
                        if name not in base_preds:
                            base_preds[name] = np.array([features["temp_c"].mean()])
                    try:
                        ens_pred = self.ensemble.predict(steps=1, base_predictions=base_preds)
                        pred = float(ens_pred[0])
                    except Exception:
                        pass

            if pred is None and self.tft and self.tft.is_fitted:
                try:
                    tft_result = self.tft.predict_intervals(steps=horizon, future_df=features)
                    pred = float(tft_result["median"][horizon - 1])
                    lower = float(tft_result["lower"][horizon - 1])
                    upper = float(tft_result["upper"][horizon - 1])
                    model_name = "tft"
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
        """No invented temperatures: API owns explicit regional fallback."""
        return []
