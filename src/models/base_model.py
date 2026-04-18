"""Abstract base class for all ThermoSense forecasting models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    All forecasting models inherit from this class.
    Enforces a consistent fit/predict/evaluate interface.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """Train the model on the provided DataFrame."""

    @abstractmethod
    def predict(self, steps: int, future_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate point predictions.

        Args:
            steps: Number of future time steps (days) to predict.
            future_df: DataFrame with future exogenous features (if required).

        Returns:
            1D numpy array of predicted temperatures.
        """

    def predict_intervals(
        self,
        steps: int,
        future_df: Optional[pd.DataFrame] = None,
        quantiles: tuple = (0.1, 0.9),
    ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals. Override in models that support quantiles.
        Default falls back to point prediction with no intervals.
        """
        point = self.predict(steps, future_df)
        return {"median": point, "lower": point, "upper": point}

    def save(self, path: str) -> None:
        """Persist model to disk. Override per model type."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load model from disk. Override per model type."""
        raise NotImplementedError
