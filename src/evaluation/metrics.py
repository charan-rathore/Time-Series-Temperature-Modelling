"""
Evaluation metrics for ThermoSense model comparison.

Includes MAE, RMSE, MAPE, Skill Score (vs. climatology), and
quantile coverage for TFT confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Absolute Error in the same units as the target (°C)."""
    return float(np.mean(np.abs(actuals - predictions)))


def rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Root Mean Squared Error - penalises large errors more than MAE."""
    return float(np.sqrt(np.mean((actuals - predictions) ** 2)))


def mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Avoids division by zero via masking."""
    mask = actuals != 0
    return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)


def skill_score(
    actuals: np.ndarray,
    predictions: np.ndarray,
    climatology: Optional[float] = None,
) -> float:
    """
    Skill Score relative to climatology (mean of actuals if not provided).

    skill_score = 1 - (RMSE_model / RMSE_climatology)

    A score of 1.0 is perfect; 0.0 means no improvement over just predicting
    the mean; negative means worse than climatology.
    """
    if climatology is None:
        climatology = float(np.mean(actuals))
    rmse_climatology = rmse(actuals, np.full_like(actuals, climatology, dtype=float))
    if rmse_climatology == 0:
        return 1.0
    return float(1 - rmse(actuals, predictions) / rmse_climatology)


def quantile_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Fraction of actual values falling within [lower, upper] prediction interval.
    For a well-calibrated 90% interval, this should be close to 0.90.
    """
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def evaluate_all(
    actuals: np.ndarray,
    predictions: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    climatology: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all metrics and return as a dictionary.

    Args:
        actuals: Ground truth temperature values.
        predictions: Model point predictions (median / 0.5 quantile).
        lower: Lower bound of prediction interval (e.g., 0.1 quantile).
        upper: Upper bound of prediction interval (e.g., 0.9 quantile).
        climatology: Reference mean for skill score; uses mean(actuals) if None.

    Returns:
        Dict with keys: mae, rmse, mape, skill_score, coverage (if bounds given).
    """
    result = {
        "mae": mae(actuals, predictions),
        "rmse": rmse(actuals, predictions),
        "mape": mape(actuals, predictions),
        "skill_score": skill_score(actuals, predictions, climatology),
    }
    if lower is not None and upper is not None:
        result["coverage_90pct"] = quantile_coverage(actuals, lower, upper)
    return result


def compare_models(
    actuals: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        actuals: Ground truth array.
        model_predictions: Dict mapping model name → predictions array.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    clim = float(np.mean(actuals))
    rows = []
    for name, preds in model_predictions.items():
        metrics = evaluate_all(actuals, preds, climatology=clim)
        metrics["model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("rmse")
