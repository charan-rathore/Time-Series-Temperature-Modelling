# Demo Evaluation Results

Real temperature dataset from the **Open-Meteo Historical Archive** (public, no API key):

- **Location:** Bangalore (12.9716, 77.5946), Asia/Kolkata
- **Series:** daily 21:00 local snapshot
- **Coverage:** 365 days (2025-08-02 to 2026-08-01)
- **Split:** train 323 / val 14 / test 14 (time-ordered)
- **Source URL:** https://archive-api.open-meteo.com/v1/archive

Machine-readable copy: [demo-results.json](demo-results.json)

## Day-1 holdout vs industry-style benchmarks

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
- **ThermoSense models:** trained on the same Open-Meteo Bangalore series with the project feature pipeline.

On this holdout window, LightGBM and the Ensemble beat persistence and climatology on Day-1 MAE/RMSE. Ensemble numbers are very strong on a small 14-day test window and should be treated as a demo score, not a production guarantee; longer rolling evaluation is recommended once live sensor ground truth accumulates.

## Reproduce

```bash
python3 scripts/run_pipeline.py --mode backfill
# rebuild clean continuous archive window if needed, then:
python3 scripts/train_models.py --models sarima lgbm ensemble --no-mlflow
```

Live UI: https://thermosense-black.vercel.app
