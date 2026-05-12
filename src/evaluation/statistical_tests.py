"""
Statistical Testing for Forecast Comparison

Provides rigorous statistical tests to validate claims like
"ThermoSense beats Google Weather by X%".

Tests included:
- Paired t-test (or Wilcoxon signed-rank for non-normal data)
- Effect size (Cohen's d for paired samples)
- 95% confidence interval on improvement
- Diebold-Mariano test for forecast comparison

Usage:
    from src.evaluation.statistical_tests import compare_forecasters
    
    result = compare_forecasters(
        errors_thermosense=[0.5, 0.3, 0.8, ...],  # ThermoSense absolute errors
        errors_baseline=[1.2, 0.9, 1.5, ...],      # Baseline absolute errors
        alternative="less"                         # ThermoSense errors are lower
    )
    
    print(f"Improvement: {result['mean_improvement_c']:.2f}°C")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size_d']:.2f}")
"""

from typing import Dict, Any, List, Optional, Literal
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compare_forecasters(
    errors_thermosense: List[float],
    errors_baseline: List[float],
    alternative: Literal["two-sided", "less", "greater"] = "less",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Statistical comparison of two forecasters' errors.
    
    Args:
        errors_thermosense: Absolute errors from ThermoSense predictions
        errors_baseline: Absolute errors from baseline forecaster
        alternative: Test alternative hypothesis
            - "less": ThermoSense errors are lower (we're better)
            - "greater": ThermoSense errors are higher (we're worse)
            - "two-sided": Errors are different (no direction)
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict with statistical test results including:
        - mean_improvement_c: Mean reduction in error (baseline - ours)
        - t_statistic: t-test statistic
        - p_value: p-value for the test
        - effect_size_d: Cohen's d for paired samples
        - ci_95_low/high: 95% confidence interval on improvement
        - significant: Whether p < alpha
        - n_samples: Number of paired observations
        - normality_p: p-value from Shapiro-Wilk normality test
        - test_used: Which test was used (t-test or Wilcoxon)
    """
    if not SCIPY_AVAILABLE:
        return {
            "error": "scipy not installed. Install with: pip install scipy",
            "significant": None,
        }
    
    ts = np.array(errors_thermosense)
    bl = np.array(errors_baseline)
    
    if len(ts) != len(bl):
        raise ValueError("Error arrays must have the same length")
    
    if len(ts) < 3:
        return {
            "error": "Need at least 3 paired observations",
            "n_samples": len(ts),
            "significant": None,
        }
    
    diff = bl - ts
    n = len(diff)
    mean_improvement = diff.mean()
    std_improvement = diff.std(ddof=1)
    
    if n >= 20:
        _, normality_p = stats.shapiro(diff)
    else:
        normality_p = 1.0
    
    use_parametric = normality_p > 0.05 or n >= 30
    
    if use_parametric:
        t_stat, p_two_sided = stats.ttest_rel(bl, ts)
        test_used = "paired_t_test"
        
        if alternative == "less":
            p_value = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
        elif alternative == "greater":
            p_value = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
        else:
            p_value = p_two_sided
    else:
        stat, p_two_sided = stats.wilcoxon(bl, ts, alternative=alternative)
        t_stat = stat
        p_value = p_two_sided
        test_used = "wilcoxon_signed_rank"
    
    if std_improvement > 0:
        cohens_d = mean_improvement / std_improvement
    else:
        cohens_d = float('inf') if mean_improvement > 0 else 0.0
    
    se = std_improvement / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low = mean_improvement - t_crit * se
    ci_high = mean_improvement + t_crit * se
    
    ts_rmse = np.sqrt((ts ** 2).mean())
    bl_rmse = np.sqrt((bl ** 2).mean())
    
    if bl_rmse > 0:
        pct_improvement = (1 - ts_rmse / bl_rmse) * 100
    else:
        pct_improvement = None
    
    return {
        "mean_improvement_c": round(mean_improvement, 4),
        "std_improvement_c": round(std_improvement, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "effect_size_d": round(cohens_d, 3),
        "effect_interpretation": interpret_cohens_d(cohens_d),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "significant": p_value < alpha,
        "alpha": alpha,
        "n_samples": n,
        "normality_p": round(normality_p, 4) if normality_p else None,
        "test_used": test_used,
        "alternative": alternative,
        "thermosense_rmse": round(ts_rmse, 4),
        "baseline_rmse": round(bl_rmse, 4),
        "pct_improvement": round(pct_improvement, 2) if pct_improvement else None,
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def diebold_mariano_test(
    errors_1: List[float],
    errors_2: List[float],
    h: int = 1,
    power: int = 2,
) -> Dict[str, Any]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    This is the standard test in forecasting literature for comparing
    two sets of forecast errors.
    
    Args:
        errors_1: Errors from forecaster 1
        errors_2: Errors from forecaster 2
        h: Forecast horizon (for autocorrelation adjustment)
        power: Loss function power (1=MAE, 2=MSE)
    
    Returns:
        Dict with DM test statistic and p-value
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not installed"}
    
    e1 = np.array(errors_1)
    e2 = np.array(errors_2)
    
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    
    d_mean = d.mean()
    
    gamma_0 = np.sum((d - d_mean) ** 2) / n
    
    gamma_sum = gamma_0
    for k in range(1, h):
        if k < n:
            gamma_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
            gamma_sum += 2 * gamma_k
    
    var_d = gamma_sum / n
    
    if var_d > 0:
        dm_stat = d_mean / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        dm_stat = 0.0
        p_value = 1.0
    
    return {
        "dm_statistic": round(dm_stat, 4),
        "p_value": round(p_value, 6),
        "mean_loss_diff": round(d_mean, 4),
        "n_samples": n,
        "forecast_horizon": h,
        "loss_power": power,
        "forecaster_1_better": dm_stat < 0,
    }


def compute_skill_score(
    errors_model: List[float],
    errors_reference: List[float],
) -> float:
    """
    Compute forecast skill score.
    
    Skill = 1 - (RMSE_model / RMSE_reference)
    
    Positive skill means the model is better than the reference.
    Skill of 1.0 means perfect forecast.
    Skill of 0.0 means same as reference.
    Negative skill means worse than reference.
    """
    rmse_model = np.sqrt(np.mean(np.array(errors_model) ** 2))
    rmse_ref = np.sqrt(np.mean(np.array(errors_reference) ** 2))
    
    if rmse_ref == 0:
        return 0.0
    
    return round(1 - rmse_model / rmse_ref, 4)


def summary_statistics(errors: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a set of errors."""
    e = np.array(errors)
    return {
        "mae": round(np.mean(np.abs(e)), 4),
        "rmse": round(np.sqrt(np.mean(e ** 2)), 4),
        "me": round(np.mean(e), 4),
        "std": round(np.std(e, ddof=1), 4),
        "min": round(np.min(e), 4),
        "max": round(np.max(e), 4),
        "median": round(np.median(e), 4),
        "n": len(e),
    }


if __name__ == "__main__":
    print("Statistical Testing Module")
    print("=" * 50)
    
    np.random.seed(42)
    n = 30
    
    baseline_errors = np.abs(np.random.normal(1.5, 0.5, n))
    thermosense_errors = np.abs(np.random.normal(0.8, 0.4, n))
    
    print("\nExample comparison:")
    print(f"  Baseline RMSE: {np.sqrt((baseline_errors**2).mean()):.3f}°C")
    print(f"  ThermoSense RMSE: {np.sqrt((thermosense_errors**2).mean()):.3f}°C")
    
    result = compare_forecasters(
        list(thermosense_errors),
        list(baseline_errors),
        alternative="less"
    )
    
    print(f"\nStatistical Test Results:")
    print(f"  Mean improvement: {result['mean_improvement_c']:.3f}°C")
    print(f"  95% CI: [{result['ci_95_low']:.3f}, {result['ci_95_high']:.3f}]°C")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Effect size (Cohen's d): {result['effect_size_d']:.2f} ({result['effect_interpretation']})")
    print(f"  Significant at α=0.05: {result['significant']}")
    print(f"  Percentage improvement: {result['pct_improvement']:.1f}%")
