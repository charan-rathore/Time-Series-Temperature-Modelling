# Historical Demo Evaluation Results - Not Validated

Real temperature dataset from the **Open-Meteo Historical Archive** (public, no API key):

- **Location:** Bangalore (12.9716, 77.5946), Asia/Kolkata
- **Series:** daily 21:00 local snapshot
- **Coverage:** 365 days (2025-08-02 to 2026-08-01)
- **Split:** train 323 / val 14 / test 14 (time-ordered)
- **Source URL:** https://archive-api.open-meteo.com/v1/archive

Machine-readable copy: [demo-results.json](demo-results.json)

**Do not use these numbers to claim forecast performance.** The ensemble stacker was fitted on the held-out test labels and then scored on a row from that same set. The other model and baseline results have not yet passed independent rolling evaluation. The rows below are retained for audit, not endorsement.

## Original reported Day-1 holdout results (unverified)

| Model | N | MAE (°C) | RMSE (°C) | MAPE (%) | Skill vs climatology |
|---|---:|---:|---:|---:|---:|
| Persistence (t-1) | 13 | 0.754 | 0.827 | 3.321 | -0.277 |
| Climatology (train mean) | 14 | 0.509 | 0.629 | 2.267 | 0.000 |
| Seasonal climatology (DOY) | 14 | 0.509 | 0.629 | 2.267 | 0.000 |
| Regional API lag-1 (Open-Meteo style) | 13 | 0.754 | 0.827 | 3.321 | -0.277 |
| ThermoSense SARIMA | 14 | 0.541 | 0.649 | 2.373 | -0.031 |
| ThermoSense LightGBM (pipeline eval) | 14 | 0.405 | 0.510 | 1.805 | 0.104 |
| ThermoSense Ensemble (pipeline eval) | 14 | 0.107 | 0.107 | 0.472 | 1.000 |

## Multi-horizon metrics from `scripts/train_models.py`

| Model | Day-1 RMSE | Day-1 MAE | Day-2 RMSE | Day-3 RMSE | Day-1 MAPE |
|---|---:|---:|---:|---:|---:|
| SARIMA | 1.148 | 1.148 | 0.890 | 0.842 | 5.06% |
| LightGBM | 0.510 | 0.405 | 0.629 | 0.626 | 1.81% |
| Ensemble | 0.107 | 0.107 | 0.068 | 0.068 | 0.47% |

## How to read the benchmarks

- **Persistence / API lag-1:** naive operational baselines used widely in weather verification.
- **Climatology / seasonal climatology:** standard reference forecasts; skill score is relative to train-mean climatology.
- **ThermoSense historical model outputs:** calculations were made on an Open-Meteo Bangalore series but do not establish an operational forecast advantage.

The ensemble score is invalid, not a production guarantee. The leaderboard is uninitialized and no live sensor data is present on the public demo. Run leakage-free rolling evaluations against observations from an actual site before making any accuracy comparison.

## Reproduce

```bash
python3 scripts/run_pipeline.py --mode backfill
# rebuild clean continuous archive window if needed, then:
python3 scripts/audit_observations.py  # same-time archive audit, not a forecast score
python3 scripts/backtest_legacy.py     # simple walk-forward legacy baselines
```

Live UI: https://thermosense-black.vercel.app
