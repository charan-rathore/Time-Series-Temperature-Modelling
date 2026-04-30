"""
Temporal Fusion Transformer (TFT) model for ThermoSense.

TFT is an attention-based architecture purpose-built for multi-horizon
time series forecasting. It produces calibrated quantile predictions,
capturing both point forecasts and uncertainty intervals.

Reference: Lim et al. (2021), "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", International Journal of Forecasting.
arXiv: 1912.09363

Expected performance: Day-1 RMSE ~0.5°C — best among all single models.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pytorch_lightning as pl
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False

from .base_model import BaseModel


class TFTModel(BaseModel):
    """
    Temporal Fusion Transformer for 3-day temperature forecasting.

    Architecture advantages over LSTM:
    - Gated residual networks filter irrelevant features
    - Variable selection networks learn which inputs matter per time step
    - Multi-head attention captures long-range seasonal dependencies
    - Quantile outputs produce calibrated uncertainty intervals

    Requires pytorch-forecasting, torch, and pytorch-lightning.
    Install via: pip install pytorch-forecasting torch pytorch-lightning
    """

    def __init__(self, config: Dict[str, Any]):
        if not _TFT_AVAILABLE:
            raise ImportError(
                "pytorch-forecasting is required for TFTModel. "
                "Install with: pip install pytorch-forecasting torch pytorch-lightning"
            )
        super().__init__(config)
        self.trainer: Optional[pl.Trainer] = None
        self.train_dataset: Optional[TimeSeriesDataSet] = None

    @staticmethod
    def _clean_for_tft(df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-numeric columns that TimeSeriesDataSet cannot handle."""
        drop_cols = []
        for c in df.columns:
            if c in ("time_idx", "location", "temp_c"):
                continue
            if df[c].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[c]):
                drop_cols.append(c)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].astype(int)
        df = df.fillna(0)
        return df

    def prepare_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """
        Construct a pytorch-forecasting TimeSeriesDataSet from the feature DataFrame.

        Known reals: variables whose future values are available at prediction time
                     (calendar features, API forecast covariates).
        Unknown reals: variables only available up to the prediction point
                       (actual temperature, local bias rolling feature).
        """
        df = self._clean_for_tft(df)

        known_reals = [c for c in self.config.get("time_varying_known_reals", [])
                       if c in df.columns]
        unknown_reals = [c for c in self.config.get("time_varying_unknown_reals", [])
                         if c in df.columns]

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="temp_c",
            group_ids=["location"],
            max_encoder_length=self.config.get("max_encoder_length", 30),
            max_prediction_length=self.config.get("max_prediction_length", 3),
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["location"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return dataset

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """
        Train the TFT model.

        Args:
            train_df: Feature DataFrame from build_feature_matrix().
            val_df: Validation DataFrame. If None, uses last 20% of train_df.
        """
        self.train_dataset = self.prepare_dataset(train_df)

        if val_df is None:
            val_split = int(len(train_df) * 0.8)
            val_df = train_df.iloc[val_split:]

        val_df = self._clean_for_tft(val_df)
        val_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset, val_df, predict=True, stop_randomization=True
        )

        train_loader = self.train_dataset.to_dataloader(
            train=True, batch_size=self.config.get("batch_size", 64), num_workers=0
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=self.config.get("batch_size", 64) * 2, num_workers=0
        )

        quantiles = self.config.get("quantiles", [0.1, 0.5, 0.9])
        self.model = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=self.config.get("learning_rate", 1e-3),
            hidden_size=self.config.get("hidden_size", 64),
            attention_head_size=self.config.get("attention_head_size", 4),
            dropout=self.config.get("dropout", 0.1),
            hidden_continuous_size=self.config.get("hidden_continuous_size", 32),
            loss=QuantileLoss(quantiles=quantiles),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )

        self.trainer = pl.Trainer(
            max_epochs=self.config.get("epochs", 50),
            gradient_clip_val=self.config.get("gradient_clip_val", 0.1),
            enable_checkpointing=True,
            enable_progress_bar=True,
        )
        self.trainer.fit(self.model, train_loader, val_loader)
        self.is_fitted = True
        print("[TFT] Training complete.")

    def predict(
        self,
        steps: int = 3,
        future_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Return median (0.5 quantile) point predictions."""
        result = self.predict_intervals(steps, future_df)
        return result["median"]

    def predict_intervals(
        self,
        steps: int = 3,
        future_df: Optional[pd.DataFrame] = None,
        quantiles: tuple = (0.1, 0.9),
    ) -> Dict[str, np.ndarray]:
        """
        Return quantile predictions: lower (10%), median (50%), upper (90%).
        This is TFT's key advantage — calibrated uncertainty without post-hoc fitting.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_intervals().")

        df = future_df if future_df is not None else pd.DataFrame()
        df = self._clean_for_tft(df)
        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset, df, predict=True, stop_randomization=True
        )
        loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        raw_predictions, _ = self.model.predict(loader, mode="raw", return_x=True)
        quantile_preds = raw_predictions["prediction"]

        return {
            "lower": quantile_preds[:, :, 0].mean(axis=0).numpy(),
            "median": quantile_preds[:, :, 1].mean(axis=0).numpy(),
            "upper": quantile_preds[:, :, 2].mean(axis=0).numpy(),
        }

    def save(self, path: str) -> None:
        """Save TFT model checkpoint and dataset metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_path = path.with_suffix(".ckpt")
        if self.trainer is not None and self.model is not None:
            self.trainer.save_checkpoint(str(checkpoint_path))
        elif self.model is not None:
            torch.save(self.model.state_dict(), str(checkpoint_path))

        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "train_dataset_params": self.train_dataset.get_parameters() if self.train_dataset else None,
            }, f)

        print(f"[TFT] Saved to {path}")

    def load(self, path: str) -> None:
        """Load TFT model from checkpoint."""
        path = Path(path)
        checkpoint_path = path.with_suffix(".ckpt")
        meta_path = path.with_suffix(".meta.pkl")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TFT checkpoint not found: {checkpoint_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.config = meta["config"]

        if meta.get("train_dataset_params"):
            self.train_dataset = TimeSeriesDataSet.from_parameters(meta["train_dataset_params"])

        self.model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
        self.model.eval()
        self.is_fitted = True
        print(f"[TFT] Loaded from {path}")
