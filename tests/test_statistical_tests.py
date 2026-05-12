"""
Tests for the statistical testing module.

Tests the forecast comparison statistical tests.
"""

import numpy as np
import pytest

from src.evaluation.statistical_tests import (
    compare_forecasters,
    diebold_mariano_test,
    compute_skill_score,
    summary_statistics,
    interpret_cohens_d,
)


class TestCompareForecasters:
    def test_significant_difference(self):
        """Should detect significant difference when one is clearly better."""
        np.random.seed(42)
        baseline = list(np.abs(np.random.normal(2.0, 0.3, 30)))
        better = list(np.abs(np.random.normal(1.0, 0.3, 30)))
        
        result = compare_forecasters(better, baseline, alternative="less")
        
        assert result["significant"] == True
        assert result["p_value"] < 0.05
        assert result["mean_improvement_c"] > 0
        assert result["effect_size_d"] > 0.5
    
    def test_no_significant_difference(self):
        """Should not find significance when errors are similar."""
        np.random.seed(42)
        errors1 = list(np.abs(np.random.normal(1.5, 0.3, 30)))
        errors2 = list(np.abs(np.random.normal(1.5, 0.3, 30)))
        
        result = compare_forecasters(errors1, errors2, alternative="two-sided")
        
        assert abs(result["mean_improvement_c"]) < 0.5
    
    def test_requires_same_length(self):
        """Should raise error for mismatched lengths."""
        with pytest.raises(ValueError):
            compare_forecasters([1, 2, 3], [1, 2])
    
    def test_requires_minimum_samples(self):
        """Should handle small samples gracefully."""
        result = compare_forecasters([1, 2], [2, 3])
        assert result.get("error") is not None
    
    def test_computes_confidence_interval(self):
        """Should compute valid confidence interval."""
        np.random.seed(42)
        baseline = list(np.abs(np.random.normal(2.0, 0.5, 50)))
        better = list(np.abs(np.random.normal(1.0, 0.5, 50)))
        
        result = compare_forecasters(better, baseline)
        
        assert result["ci_95_low"] < result["mean_improvement_c"]
        assert result["ci_95_high"] > result["mean_improvement_c"]
    
    def test_computes_percentage_improvement(self):
        """Should compute percentage improvement correctly."""
        result = compare_forecasters(
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )
        
        assert result["pct_improvement"] == 50.0


class TestInterpretCohensD:
    def test_negligible(self):
        assert interpret_cohens_d(0.1) == "negligible"
    
    def test_small(self):
        assert interpret_cohens_d(0.3) == "small"
    
    def test_medium(self):
        assert interpret_cohens_d(0.6) == "medium"
    
    def test_large(self):
        assert interpret_cohens_d(1.0) == "large"
    
    def test_handles_negative(self):
        assert interpret_cohens_d(-0.8) == "large"


class TestDieboldMarianoTest:
    def test_detects_better_forecaster(self):
        """Should correctly identify the better forecaster."""
        np.random.seed(42)
        errors1 = list(np.random.normal(0, 1.0, 50))
        errors2 = list(np.random.normal(0, 2.0, 50))
        
        result = diebold_mariano_test(errors1, errors2)
        
        assert result["forecaster_1_better"] == True
    
    def test_handles_equal_forecasters(self):
        """Should show no difference for identical errors."""
        errors = list(np.random.normal(0, 1.0, 30))
        
        result = diebold_mariano_test(errors, errors)
        
        assert result["dm_statistic"] == 0.0


class TestSkillScore:
    def test_perfect_forecast(self):
        """Perfect forecast should have skill close to 1."""
        model_errors = [0.0, 0.0, 0.0]
        ref_errors = [1.0, 1.0, 1.0]
        
        skill = compute_skill_score(model_errors, ref_errors)
        assert skill == 1.0
    
    def test_same_as_reference(self):
        """Same as reference should have skill of 0."""
        errors = [1.0, 1.0, 1.0]
        
        skill = compute_skill_score(errors, errors)
        assert skill == 0.0
    
    def test_worse_than_reference(self):
        """Worse than reference should have negative skill."""
        model_errors = [2.0, 2.0, 2.0]
        ref_errors = [1.0, 1.0, 1.0]
        
        skill = compute_skill_score(model_errors, ref_errors)
        assert skill < 0


class TestSummaryStatistics:
    def test_computes_all_stats(self):
        """Should compute all summary statistics."""
        errors = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        stats = summary_statistics(errors)
        
        assert stats["mae"] == 3.0
        assert stats["me"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["n"] == 5
    
    def test_rmse_calculation(self):
        """Should compute RMSE correctly."""
        errors = [1.0, 1.0, 1.0]
        
        stats = summary_statistics(errors)
        
        assert stats["rmse"] == 1.0
