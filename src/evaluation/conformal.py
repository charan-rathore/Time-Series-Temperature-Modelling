"""Split-conformal prediction intervals, calibrated on backtest residuals.

Instead of a fixed +/- offset, the interval half-width for horizon h is the
finite-sample-corrected (1 - alpha) quantile of absolute out-of-sample
errors that the same forecaster made earlier. Under exchangeability this
covers the truth about (1 - alpha) of the time; ``walk_forward_coverage``
checks that claim on data the calibration never saw instead of asserting it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def conformal_halfwidth(residuals: Sequence[float], alpha: float = 0.1) -> float:
    """Half-width q such that |error| <= q for about (1 - alpha) of new cases.

    Uses the split-conformal rank ceil((n + 1)(1 - alpha)). If that rank
    exceeds n the data cannot support the requested coverage (the largest
    residual would give only n/(n+1) marginal coverage), so an infinite
    half-width is returned: the honest "no finite interval can promise this",
    never a number that masquerades as the requested coverage.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    if n == 0:
        raise ValueError("need at least one residual to calibrate an interval")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(r[rank - 1])


def interval(point: float, halfwidth: float) -> Dict[str, float]:
    return {"lower": point - halfwidth, "upper": point + halfwidth}


def empirical_coverage(errors: Sequence[float], halfwidth: float) -> float:
    e = np.abs(np.asarray(errors, dtype=float))
    return float(np.mean(e <= halfwidth))


def walk_forward_coverage(
    fold_errors: List[Sequence[float]], alpha: float = 0.1, min_calibration_folds: int = 2
) -> dict:
    """Calibrate on folds 0..k-1, score coverage on fold k, for every k.

    Returns per-fold coverage and the pooled coverage plus mean half-width.
    This is the honest test: calibration data always precedes test data.
    """
    if len(fold_errors) <= min_calibration_folds:
        raise ValueError("need more folds than min_calibration_folds")
    covs, widths, hits, total = [], [], 0, 0
    for k in range(min_calibration_folds, len(fold_errors)):
        calib = np.concatenate([np.asarray(f, dtype=float) for f in fold_errors[:k]])
        q = conformal_halfwidth(calib, alpha)
        test = np.abs(np.asarray(fold_errors[k], dtype=float))
        covs.append(float(np.mean(test <= q)))
        widths.append(q)
        hits += int(np.sum(test <= q))
        total += len(test)
    return {
        "target_coverage": 1 - alpha,
        "fold_coverage": covs,
        "pooled_coverage": hits / total,
        "mean_halfwidth": float(np.mean(widths)),
        "n_test": total,
    }
