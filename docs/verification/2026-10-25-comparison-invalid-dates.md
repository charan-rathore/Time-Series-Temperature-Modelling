# Comparison endpoint date validation

Base: `27cb203`. Scope: malformed `start_date`/`end_date` on `GET /api/leaderboard/comparison` returned 500 instead of the same 400 used by `/history`. Catch parsing `ValueError`; no date-range, prediction, evaluation, dependency, or model changes.

## Real API proof

FastAPI `TestClient`, `raise_server_exceptions=False`, actual route and local model setup:

- Before: two malformed-date regressions fail, 15 existing API tests pass.
- After: 19 API tests pass. Includes malformed start/end, valid explicit date range returning 200 with unchanged date values, and omitted dates returning 200.

## Full-suite gate remains blocked

Python 3.12, locked environment with pandas 3.1.0:

- Pristine baseline: 333 passed, 1 failed.
- Baseline source plus the initial two new tests: 333 passed, 3 failed (two intended REDs plus existing failure).
- Patched source plus initial two tests: 335 passed, 1 failed.
- Final patch including all four new API tests: 337 passed, 1 failed.

The unchanged failure is `test_saved_ensemble_scores_match_training_evaluation`: day-1 MAE 0.8986850520382664 versus saved 0.9296490805744587 exceeds the existing 0.02 tolerance. No tolerance, stored score, artifact, or package was changed to make it pass.

A separate existing environment with pandas 2.3.3 passes all 338 tests including both extra API controls. This is evidence about that environment, not approval to change the lock or a substitute for the failed locked-environment gate.

This PR must remain draft and unmerged until the full required environment passes. Generated feature-matrix test side effects were restored and are not in the patch.
