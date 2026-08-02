"""
Ensemble stacker for ThermoSense.

A linear meta-learner (Ridge Regression) trained on out-of-fold predictions
from SARIMA(X), LightGBM, and TFT. Learns optimal per-horizon weighting
without data leakage.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .base_model import BaseModel


class EnsembleStacker(BaseModel):
    """
    Level-1 meta-learner over SARIMA, LightGBM, and TFT.

    Trained on out-of-fold (OOF) predictions so the meta-learner never
    sees predictions made on data it was also trained on - preventing
    overfitting at the ensemble level.

    Separate meta-models per forecast horizon (Day 1, Day 2, Day 3).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.meta_models: Dict[int, Ridge] = {}
        self.base_model_names: List[str] = config.get("base_models", ["sarima", "lgbm", "tft"])

    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """Not used - ensemble trains via fit_from_oof(). This satisfies the ABC."""
        raise NotImplementedError("Use fit_from_oof() instead for ensemble training.")

    def fit_from_oof(
        self,
        oof_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        horizons: List[int] = [1, 2, 3],
    ) -> None:
        """
        Train meta-learner from out-of-fold predictions.

        Args:
            oof_predictions: Dict mapping model_name → OOF predictions array.
                             Shape: (n_samples, n_horizons)
            actuals: Ground truth temperatures, shape (n_samples,) or (n_samples, n_horizons).
            horizons: List of forecast horizons to train separate meta-models for.
        """
        for h_idx, h in enumerate(horizons):
            X_meta = np.column_stack([
                oof_predictions[name][:, h_idx]
                for name in self.base_model_names
                if name in oof_predictions
            ])
            y_meta = actuals[:, h_idx] if actuals.ndim > 1 else actuals

            meta = Ridge(alpha=self.config.get("alpha", 1.0))
            meta.fit(X_meta, y_meta)
            self.meta_models[h] = meta
            print(f"[Ensemble] Horizon={h} - meta-model weights: {meta.coef_}")

        self.is_fitted = True

    def predict(
        self,
        steps: int = 3,
        base_predictions: Optional[Dict[str, np.ndarray]] = None,
        future_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Generate ensemble predictions by stacking base model outputs.

        Args:
            steps: Number of forecast horizons to predict.
            base_predictions: Dict mapping model_name → scalar prediction per model.
                Each value should be a scalar or 1-element array.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before calling predict().")
        if base_predictions is None:
            raise ValueError("base_predictions dict is required for ensemble inference.")

        results = []
        for h in range(1, steps + 1):
            if h not in self.meta_models:
                raise ValueError(f"No meta-model fitted for horizon {h}.")
            preds_list = []
            for name in self.base_model_names:
                if name in base_predictions:
                    val = base_predictions[name]
                    preds_list.append(float(np.atleast_1d(val)[0]))
            X_meta = np.array(preds_list).reshape(1, -1)
            results.append(self.meta_models[h].predict(X_meta)[0])

        return np.array(results)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "meta_models": self.meta_models,
                "base_model_names": self.base_model_names,
                "config": self.config,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.meta_models = state["meta_models"]
        self.base_model_names = state["base_model_names"]
        self.config = state["config"]
        self.is_fitted = True
