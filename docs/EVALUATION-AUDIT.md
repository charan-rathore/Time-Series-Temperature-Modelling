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
   rows have `temp_c == temp_c_api`; 40 rows (2 June to 11 July 2024) are
   sensor readings. Every score below is forecasting the reanalysis, not a
   balcony sensor. Claims about beating commercial apps at *your* location
   need sensor data that does not exist yet.
3. **No commercial forecast is archived.** `app_pred_day1..3` hold 5 to 6
   distinct values, and 26/24/24 °C on about 97% of rows: a placeholder
   forward-filled from the 40 legacy rows, not a forecast. The demo table's
   "Regional API lag-1" row was persistence under another name.
   `data/baselines/forecasts.db` holds 6 forecasts. There is no fair
   external-forecast baseline yet.
4. **Machine learning barely beats persistence.** The repo's own
   rolling-origin backtest (`models/backtest.json`, 6 folds) gives LightGBM
   +2.4% day-1 skill over persistence and 0% at day 3. The independent audit
   agrees.

## Independent walk-forward audit

365 daily origins (20 Oct 2025 to 16 Oct 2026), expanding window, models refit
weekly on data up to the origin only. "✓" means the 95% moving-block bootstrap
interval of RMSE(model) − RMSE(persistence) lies entirely below zero. Hot day:
temperature at or above the 90th percentile of the pre-test period (26.8 °C).

| Day 1 (n = 365) | RMSE °C | MAE °C | Skill vs persistence | 95% CI of RMSE − persistence | Hot days POD / FAR |
| --- | ---: | ---: | ---: | --- | --- |
| Persistence | 1.243 | 0.895 | 0 | - | 0.77 / 0.23 |
| Seasonal naive (7 d) | 1.870 | 1.448 | −0.504 | [+0.41, +0.83] | 0.58 / 0.42 |
| Day-of-year climatology | 1.693 | 1.350 | −0.362 | [+0.22, +0.69] | 0.29 / 0.17 |
| Anomaly AR(1) | 1.180 | 0.910 | +0.051 | [−0.12, +0.00] | 0.62 / 0.11 |
| **Ridge on lags + weather** | **1.147** | **0.864** | **+0.078 ✓** | [−0.16, −0.03] | 0.69 / 0.14 |
| Gradient boosting (LightGBM stand-in) | 1.256 | 0.943 | −0.010 | [−0.09, +0.13] | 0.69 / 0.20 |

Day 2 and day 3 follow the same pattern: ridge +7.6% and +8.3% skill (both
✓), gradient boosting not distinguishable from persistence. Full numbers:
`docs/walk-forward-audit.json`.

What this supports: a small, regularised linear model gives a consistent ~8%
RMSE improvement over "tomorrow equals today" on this series, and cuts hot-day
false alarms from 23% to 14% at day 1 while missing slightly more events
(POD 0.69 vs 0.77). What it does not support: any claim about sensor-level
accuracy or about beating commercial apps.

## What would make the claim real

- Archive real day-ahead forecasts from Open-Meteo's forecast API (not the
  archive) every day at a fixed time, so a fair external baseline exists.
- Score against sensor readings once at least 60 consecutive sensor days are
  logged; until then, label every metric "against the Open-Meteo archive".
- Keep `walk_forward_audit.py` in CI and publish its table instead of a
  single 14-day split.

## Reproduce

    python3 scripts/walk_forward_audit.py --test-days 365 --refit 7
