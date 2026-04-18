"""
SARIMA(X) model for ThermoSense.

Upgrades the original ARIMA(1,0,0) to a seasonal ARIMA with exogenous variables
(humidity, pressure). Auto-selects order via AIC using pmdarima.

Reference baseline for comparison: ARIMA(1,0,0) → RMSE ~0.87°C (37-day dataset).
Target for SARIMAX with exogenous features: RMSE ~0.6–0.7°C.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pmdarima as pm

from .base_model import BaseModel


class SARIMAXModel(BaseModel):
    """
    Seasonal ARIMA with optional exogenous regressors (SARIMAX).

    Uses pmdarima.auto_arima to select optimal (p, d, q)(P, D, Q, m) orders
    via AIC minimisation. Seasonal period m=7 captures weekly temperature cycles.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exog_cols: List[str] = config.get("exog_cols", ["humidity_pct", "pressure_hpa"])

    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Fit auto-ARIMA with seasonal and exogenous components.

        Args:
            train_df: Training DataFrame with 'temp_c' and exog columns.
        """
        y = train_df["temp_c"].values
        exog = self._get_exog(train_df)

        self.model = pm.auto_arima(
            y,
            exogenous=exog,
            seasonal=self.config.get("seasonal", True),
            m=self.config.get("m", 7),
            information_criterion=self.config.get("information_criterion", "aic"),
            stepwise=self.config.get("stepwise", True),
            suppress_warnings=True,
            error_action="ignore",
        )
        self.is_fitted = True
        print(f"[SARIMA] Best model: {self.model.order} x {self.model.seasonal_order}")

    def predict(
        self,
        steps: int,
        future_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Generate point predictions for the next `steps` days.

        Args:
            steps: Number of days to forecast.
            future_df: DataFrame with future exogenous features (required if exog used in fit).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        future_exog = self._get_exog(future_df) if future_df is not None else None
        forecast = self.model.predict(n_periods=steps, exogenous=future_exog)
        return np.array(forecast)

    def predict_intervals(
        self,
        steps: int,
        future_df: Optional[pd.DataFrame] = None,
        quantiles: tuple = (0.1, 0.9),
    ) -> dict:
        """Return point prediction and 80% confidence interval from ARIMA."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_intervals().")

        future_exog = self._get_exog(future_df) if future_df is not None else None
        forecast, conf_int = self.model.predict(
            n_periods=steps,
            exogenous=future_exog,
            return_conf_int=True,
            alpha=0.2,
        )
        return {
            "median": np.array(forecast),
            "lower": conf_int[:, 0],
            "upper": conf_int[:, 1],
        }

    def _get_exog(self, df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        """Extract exogenous variables from DataFrame, returning None if unavailable."""
        if df is None or not self.exog_cols:
            return None
        available = [c for c in self.exog_cols if c in df.columns]
        if not available:
            return None
        return df[available].values
