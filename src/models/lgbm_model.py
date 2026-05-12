"""
LightGBM forecasting model for ThermoSense.

Treats multi-step forecasting as supervised regression using the full
engineered feature matrix. SHAP values provide interpretability.

Expected improvement over SARIMA: RMSE ~0.7°C on Day-1 horizon.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base_model import BaseModel

TARGET_COL = "temp_c"
EXCLUDE_COLS = {TARGET_COL, "date", "location", "time_idx",
                "app_pred_day1", "app_pred_day2", "app_pred_day3",
                "api_bias"}


class LGBMForecastModel(BaseModel):
    """
    Gradient-boosted tree model for temperature forecasting.

    One model per forecast horizon (Day 1, Day 2, Day 3).
    Uses all engineered features from src/features/engineer.py.
    """

    def __init__(self, config: Dict[str, Any], horizon: int = 1):
        super().__init__(config)
        self.horizon = horizon
        self.feature_cols: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """
        Train LightGBM on the feature matrix.

        Args:
            train_df: Feature DataFrame (output of build_feature_matrix()).
            val_df: Optional validation DataFrame for early stopping.
        """
        self.feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

        # Shift the target backward to create a forward-looking label for the horizon
        shifted = train_df[TARGET_COL].shift(-self.horizon)
        valid_mask = shifted.notna()
        y_train = shifted[valid_mask].values
        X_train = train_df.loc[valid_mask, self.feature_cols]

        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": self.config.get("n_estimators", 500),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "num_leaves": self.config.get("num_leaves", 31),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "verbose": -1,
        }

        callbacks = [
            lgb.early_stopping(self.config.get("early_stopping_rounds", 50), verbose=False),
            lgb.log_evaluation(self.config.get("verbose_eval", 50)),
        ]

        train_set = lgb.Dataset(X_train, label=y_train)

        if val_df is not None:
            shifted_val = val_df[TARGET_COL].shift(-self.horizon)
            val_mask = shifted_val.notna()
            y_val = shifted_val[val_mask].values
            X_val = val_df.loc[val_mask, self.feature_cols]
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            self.model = lgb.train(
                params,
                train_set,
                valid_sets=[train_set, val_set],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(params, train_set)

        self.is_fitted = True
        print(f"[LightGBM] Horizon={self.horizon} — best iteration: {self.model.best_iteration}")

    def predict(
        self,
        steps: int = 1,
        future_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Predict temperature for the next `steps` days.

        Args:
            steps: Number of steps (should match self.horizon for single-model use).
            future_df: Feature DataFrame for the prediction rows.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        if future_df is None:
            raise ValueError("future_df with feature columns is required for LightGBM prediction.")

        available_cols = [c for c in self.feature_cols if c in future_df.columns]
        X = future_df[available_cols].iloc[:steps]
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted Series (SHAP-style gains)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        importance = self.model.feature_importance(importance_type="gain")
        return pd.Series(importance, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols,
                         "horizon": self.horizon, "config": self.config}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.feature_cols = state["feature_cols"]
        self.horizon = state["horizon"]
        self.config = state["config"]
        self.is_fitted = True
