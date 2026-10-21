import numpy as np
import pandas as pd
import pytest

from src.quality.checks import (
    error_drift, find_gaps, find_outliers, find_stuck_runs, freshness,
    population_stability_index, summarize)


def test_gap_is_located_and_sized():
    dates = list(pd.date_range("2026-10-01", periods=5)) + list(pd.date_range("2026-10-09", periods=3))
    assert find_gaps(dates) == [{"after": "2026-10-05", "missing_days": 3}]


def test_no_gaps_in_a_clean_series_and_order_does_not_matter():
    d = list(pd.date_range("2026-10-01", periods=6))
    assert find_gaps(d) == [] and find_gaps(d[::-1]) == []


def test_single_spike_is_flagged_and_normal_swings_are_not():
    rng = np.random.default_rng(0)
    v = 25 + rng.normal(0, 0.6, 120)
    assert find_outliers(v) == []
    v[60] += 12
    assert find_outliers(v) == [60]


def test_a_level_shift_is_not_mistaken_for_a_spike():
    v = np.concatenate([np.full(50, 22.0), np.full(50, 30.0)]) + np.random.default_rng(1).normal(0, .1, 100)
    assert find_outliers(v) == []          # one jump up, none back: a regime change, not an error


def test_outliers_need_enough_history():
    assert find_outliers([20, 21, 90, 21]) == []


def test_stuck_sensor_run_is_detected():
    v = [20.1, 21.0, 22.4, 22.4, 22.4, 22.4, 22.4, 21.7]
    assert find_stuck_runs(v, min_run=4) == [{"start_index": 2, "length": 5, "value": 22.4}]
    assert find_stuck_runs([1, 2, 3, 4, 5]) == []


def test_psi_is_near_zero_for_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(3)
    a, b = rng.normal(25, 2, 2000), rng.normal(25, 2, 2000)
    assert population_stability_index(a, b) < 0.05
    assert population_stability_index(a, rng.normal(30, 2, 2000)) > 0.25


def test_psi_on_a_frozen_reference_still_sees_a_shift():
    # constant reference window: quantile bins collapse to one edge, but a
    # window that moved must register as drift, not "0.0 = stable"
    assert population_stability_index([20.0] * 50, [35.0] * 50) > 0.25
    # and a matching window still reads stable
    assert population_stability_index([20.0] * 50, [20.0] * 50) < 0.05


def test_summarize_reports_pipeline_filled_gaps():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "temp_c": [20.0, 21.0, 22.0, 23.0],
        "gap_filled": [False, True, True, False],
    })
    out = summarize(df, today=pd.Timestamp("2026-01-05"))
    assert out["gaps"] == []                      # date sequence is complete
    assert out["gap_filled_count"] == 2
    assert out["gap_filled_dates"] == ["2026-01-02", "2026-01-03"]
    # frames without the marker column simply report zero
    plain = df.drop(columns=["gap_filled"])
    assert summarize(plain, today=pd.Timestamp("2026-01-05"))["gap_filled_count"] == 0


def test_error_drift_flags_only_materially_worse_error():
    rng = np.random.default_rng(4)
    assert error_drift(rng.normal(0, 1.5, 30), 1.5)["status"] == "ok"
    assert error_drift(rng.normal(0, 4.0, 30), 1.5)["status"] == "drifting"
    assert error_drift([1, 2, 3], 1.5)["status"] == "insufficient_data"


def test_freshness_respects_the_slo():
    today = pd.Timestamp("2026-10-20")
    assert freshness("2026-10-19", today)["stale"] is False
    out = freshness("2026-10-14", today)
    assert out["stale"] is True and out["age_days"] == 6


def test_summarize_on_the_committed_dataset_runs_and_reports_shape():
    df = pd.read_csv("data/bundle/daily_merged.csv", parse_dates=["date"])
    s = summarize(df, today=pd.Timestamp("2026-10-20"))
    assert s["rows"] == len(df) and s["freshness"]["stale"] is False
    assert set(s) >= {"gaps", "outlier_count", "stuck_runs", "temp_psi_first_vs_second_half"}
