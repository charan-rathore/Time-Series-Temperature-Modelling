# METAR optional data must not crash the poller

Based on main `27cb203`. No changes to held comparison-date or daily-fetch candidates.

Actual `poll_once` to SQLite regression tests:

- A valid 25 C reading with singular dewpoint -243.04 previously raised ZeroDivisionError. It now stores temperature with unknown humidity.
- Five malformed timestamps, including nonfinite and out-of-range values, previously raised. They now return `no_time` before database creation.
- Initial tests used an incorrect SQL column/name. Those were corrected before the final source-only RED/GREEN check: baseline 6 FAIL / 12 PASS to candidate 18 PASS.

The full older pandas 2.3.3 environment had 340 PASS. The actual locked pandas 3.1 environment has 339 PASS / 1 FAIL: the preexisting `tests/test_results_freshness.py::test_saved_ensemble_scores_match_training_evaluation` check. No dependency or evaluation changes.

Merge hold remains. This is not full-green or merge-ready.
