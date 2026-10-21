import numpy as np
import pytest

from src.evaluation.conformal import (
    conformal_halfwidth, empirical_coverage, interval, walk_forward_coverage)


def test_halfwidth_uses_the_finite_sample_rank():
    r = np.arange(1, 10)                      # |errors| 1..9, n = 9
    # rank = ceil(10 * 0.9) = 9 -> the 9th smallest
    assert conformal_halfwidth(r, 0.1) == 9
    # rank = ceil(10 * 0.5) = 5
    assert conformal_halfwidth(r, 0.5) == 5


def test_too_few_residuals_cannot_promise_the_requested_coverage():
    # rank = ceil(3 * 0.95) = 3 > n = 2: the max residual would give only
    # 2/3 marginal coverage, so the honest answer is an infinite half-width
    assert conformal_halfwidth([1.0, 2.0], 0.05) == float("inf")
    # an infinite interval covers everything, truthfully
    assert empirical_coverage([100.0, -40.0], float("inf")) == 1.0
    # alpha reachable with the same n still returns a finite order statistic
    assert conformal_halfwidth([1.0, 2.0], 0.5) == 2.0


def test_signs_do_not_matter():
    assert conformal_halfwidth([-3, 1, -2, 2], 0.25) == conformal_halfwidth([3, 1, 2, 2], 0.25)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        conformal_halfwidth([], 0.1)
    with pytest.raises(ValueError):
        conformal_halfwidth([1.0], 1.5)


def test_interval_is_symmetric_around_the_point():
    assert interval(20.0, 1.5) == {"lower": 18.5, "upper": 21.5}


def test_walk_forward_coverage_is_near_target_on_exchangeable_errors():
    rng = np.random.default_rng(1)
    folds = [rng.normal(0, 1.0, 400) for _ in range(6)]
    out = walk_forward_coverage(folds, 0.1)
    assert out["pooled_coverage"] == pytest.approx(0.9, abs=0.03)
    assert out["n_test"] == 4 * 400


def test_walk_forward_exposes_a_regime_shift_instead_of_hiding_it():
    rng = np.random.default_rng(2)
    calm = [rng.normal(0, 0.5, 300) for _ in range(4)]
    stormy = [rng.normal(0, 2.0, 300)]
    out = walk_forward_coverage(calm + stormy, 0.1)
    assert out["fold_coverage"][-1] < 0.7        # the miss is reported, not smoothed over


def test_calibration_never_uses_the_fold_it_scores():
    folds = [np.full(50, 1.0), np.full(50, 1.0), np.full(50, 10.0)]
    out = walk_forward_coverage(folds, 0.1)
    assert out["fold_coverage"] == [0.0]         # q learned from 1.0s cannot cover 10s
    assert empirical_coverage([10.0], 1.0) == 0.0


def test_too_few_folds_rejected():
    with pytest.raises(ValueError):
        walk_forward_coverage([np.ones(10), np.ones(10)], 0.1)
