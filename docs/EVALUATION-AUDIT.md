# Evaluation audit (2 November 2026)

What the ThermoSense forecasts can and cannot claim today, re-derived from the
data in this repository by `scripts/walk_forward_audit.py`, which shares no
code with the training pipeline.

## Findings

1. **The published Ensemble score was in-sample.** `docs/demo-results.md`
   reported Ensemble day-1 RMSE 0.107 °C (MAE also 0.107, skill 1.000) over a
   14-day window, with day-2 and day-3 errors lower than day-1. Those numbers
   came from a version of `train_ensemble` that fitted the stacker on
   test-period predictions; `scripts/train_models.py` documents and fixes
   that, but the page and the README link were never updated. They are
   withdrawn.
2. **The "ground truth" is mostly Open-Meteo's own archive.** 831 of 871
   rows have `temp_c == temp_c_api`; 40 rows (2 June to 11 July 2024) differ
   from it and carry a sensor flag in the data (their physical provenance is
   unverified, see the qualification section). Every score below is
   forecasting the reanalysis, not a verified balcony sensor. Claims about beating commercial apps at *your* location
   need sensor data that does not exist yet.
3. **No commercial forecast is archived.** `app_pred_day1..3` hold 5 to 6
   distinct values, and 26/24/24 °C on about 97% of rows: a placeholder
   forward-filled from the 40 legacy rows, not a forecast. The demo table's
   "Regional API lag-1" row was persistence under another name.
   `data/baselines/forecasts.db` holds 6 forecasts. There is no fair
   external-forecast baseline yet.
4. **Machine learning barely beats persistence.** The repo's own
   rolling-origin backtest (`models/backtest.json`, 6 folds) gives LightGBM
   +2.4% day-1 skill over persistence and 0% at day 3. The walk-forward audit below
   (separate code, same author) agrees.

## Walk-forward audit (code separate from training)

365 daily origins (20 Oct 2025 to 16 Oct 2026), expanding window, models refit
weekly on data up to the origin only. "✓" means the 95% moving-block bootstrap
interval of RMSE(model) − RMSE(persistence) lies entirely below zero. Hot day:
temperature at or above the 90th percentile of the pre-test period (26.8 °C).

| Day 1 (n = 365) | RMSE °C | MAE °C | Skill vs persistence | 95% CI of RMSE − persistence | Hot days POD / FAR |
| --- | ---: | ---: | ---: | --- | --- |
| Persistence | 1.243 | 0.895 | 0 | - | 0.77 / 0.23 |
| Seasonal naive (7 d) | 1.870 | 1.448 | −0.504 | [+0.41, +0.84] | 0.58 / 0.42 |
| Day-of-year climatology | 1.693 | 1.350 | −0.362 | [+0.23, +0.69] | 0.29 / 0.17 |
| Anomaly AR(1) | 1.180 | 0.910 | +0.051 | [−0.12, +0.00] | 0.62 / 0.11 |
| **Ridge on lags + weather** | **1.147** | **0.864** | **+0.078 ✓** | [−0.16, −0.04] | 0.69 / 0.14 |
| Gradient boosting (LightGBM stand-in) | 1.256 | 0.943 | −0.010 | [−0.09, +0.13] | 0.69 / 0.20 |

Day 2 and day 3 follow the same pattern: ridge +7.6% and +8.3% skill (both
✓), gradient boosting not distinguishable from persistence. Full numbers:
`docs/walk-forward-audit.json`.

What this supports: a small, regularised linear model gives a consistent ~8%
RMSE improvement over "tomorrow equals today" on this series, and cuts hot-day
false alarms from 23% to 14% at day 1 while missing slightly more events
(POD 0.69 vs 0.77). What it does not support: any claim about sensor-level
accuracy or about beating commercial apps.

## Qualification of the headline (read before quoting the ~8%)

- One test window of 365 days of one series, refit weekly, one random seed for the bootstrap. Nothing here is a multi-station or out-of-region result.
- Scores are against the Open-Meteo archive column. 40 of the 871 rows differ from the archive and carry a sensor flag; their physical provenance (which device, placement, calibration) has not been verified, so "sensor" is a label in the data, not a verified measurement.
- A bug in the moving-block bootstrap (the highest valid block start was never drawn, so the last test day could not be sampled) was fixed in `block_starts`, with a regression test, and the table was regenerated. The change moved interval ends by at most 0.015 °C and no ✓ changed; the effect of any remaining protocol choice (block length 14, 1000 resamples) is not measured.
- The audit is not independently reviewed. The old 0.107 °C ensemble score was in-sample and is withdrawn; that is a separate matter from this table.

## What would make the claim real

- Archive real day-ahead forecasts from Open-Meteo's forecast API (not the
  archive) every day at a fixed time, so a fair external baseline exists.
- Score against sensor readings once at least 60 consecutive sensor days are
  logged; until then, label every metric "against the Open-Meteo archive".
- Keep `walk_forward_audit.py` in CI and publish its table instead of a
  single 14-day split.

## Leakage audit of the training pipeline (3 November 2026)

Checked by reading `src/features/engineer.py` and `src/models/lgbm_model.py`
against `data/bundle/feature_matrix.csv`.

- **No look-ahead in lags or rolling windows.** Lags use `shift(k)`, rolling
  statistics `shift(1).rolling(w)`, the API-bias rolling mean `shift(1)`, and
  the label is `temp_c.shift(-h)`. Gap filling is forward-fill only, and no
  row in the current data is gap-filled.
- **Issue-time assumption (train/serve skew, not test leakage).** 9 of the 34
  LightGBM features are the issue day's own weather: humidity, dew point,
  precipitation, pressure, cloud cover, wind, solar radiation, `temp_c_api`
  and `gap_filled`. That is valid only for a forecast issued after the issue
  day has ended. A forecast issued during the day would see partial or
  forecast weather that training never saw. The issue time has to be fixed
  and the features restricted to what is known then.
- **`gap_filled` is used as a feature.** `EXCLUDE_COLS` names
  `is_gap_filled`, but the column is `gap_filled`. It is constant (False)
  today, so it changes nothing yet; the name should be corrected before
  gap-filled rows appear.
- **The existing baseline table compares unlike quantities.**
  `daily_forecasts` stores Open-Meteo's daily *mean*, while the only measured
  actual (`airport_vobg_metar`) is the reading nearest 21:00 IST. Scoring one
  against the other would measure the diurnal cycle, not forecast skill.
  `fetch_open_meteo_forecast` also takes horizons from `date.today()`, which
  is UTC on a CI runner.

## Forecast ledger

`src/evaluation/ledger.py` and `scripts/ledger_daily.py`: an append-only,
hash-chained JSONL file (`data/ledger/ledger.jsonl`) of forecasts and
observations for one target, the temperature at 21:00 IST, matching what the
airport poller measures. Enforced on every append and re-checked on every
run: a forecast is issued before its target day starts (IST), its horizon
matches issue and target dates, the first forecast per source, date and
horizon wins, observations are corrected only by a superseding record, and
an edited, reordered or deleted line breaks the chain. The daily workflow
(`.github/workflows/forecast-ledger.yml`, 19:07 IST) records Open-Meteo's
forecast-API value for 21:00 on D+1..D+3 and persistence, and copies new
21:00 airport readings in. The scoreboard (`--score`) makes no claim below
60 matched days or when the 95% block-bootstrap interval of the
squared-error difference includes zero.

ThermoSense's own models predict the daily mean, so they are not in the
ledger until they are trained on the 21:00 target. Until the ledger has 60
matched days, there is no evidence either way about beating Open-Meteo.

## Reproduce

    python3 scripts/walk_forward_audit.py --test-days 365 --refit 7
    python3 scripts/ledger_daily.py --score   # ledger scoreboard
