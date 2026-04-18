# ThermoSense — Hyperlocal Temperature Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Beating commercial weather apps by learning the microclimate of your exact location.**

---

## The Problem

Commercial weather apps (AccuWeather, Weather.com, Google Weather) report conditions from the **nearest official weather station** — which may be several kilometres away, at a different elevation, or surrounded by open fields instead of your urban rooftop. The result is a systematic, predictable offset between the app's reading and reality at your specific location.

This project turns that known gap into a feature.

By computing the rolling **sensor-to-API bias** — the daily difference between an actual local temperature reading and the Open-Meteo API value for that lat/lon — and using it as an input to a Temporal Fusion Transformer, ThermoSense learns the **hyperlocal microclimate correction** that commercial apps will never apply.

**Core claim**: A model trained on ~40 sensor readings + 1 year of public meteorological data, augmented with the local bias feature, should outperform the commercial app baseline at the specific measurement location.

---

## Results (Original ARIMA Baseline)

| Metric | Small dataset (3-day train) | Large dataset (37-day train) | Improvement |
|--------|----------------------------|------------------------------|-------------|
| **RMSE** | 2.96 °C | 0.87 °C | **70% reduction** |
| **MAE Day-1** | — | 1.0 °C | baseline |
| **MAE Day-2** | — | 1.225 °C | — |
| **MAE Day-3** | — | 1.5 °C | — |

The upgraded system targets **Day-1 RMSE ≤ 0.5 °C** using TFT + ensemble stacking.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                       │
│                                                              │
│   Open-Meteo API          OpenWeatherMap        Legacy CSV   │
│   (free, no key)          (baseline only)       (40 days)    │
│         │                       │                   │        │
│         └───────────────────────┴───────────────────┘        │
│                               │                              │
│                     src/data/fetcher.py                      │
│                     (retry, raw save, config-driven)         │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                    PREPROCESSING LAYER                        │
│   src/data/preprocess.py                                     │
│   • Sensor readings take precedence over API for overlaps    │
│   • Forward-fill for any missing dates (flagged)             │
│   • Output: data/processed/daily_merged.parquet              │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                    │
│   src/features/engineer.py                                   │
│   • Lag features:  T-1, T-2, T-3, T-7                       │
│   • Rolling stats: mean/std over 3/7/14-day windows         │
│   • Calendar:      day_sin, day_cos, month_sin, month_cos    │
│   • External:      humidity, pressure, cloud cover, UV index │
│   • api_bias:      sensor_temp − api_temp (rolling 7-day)   │  ← key innovation
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                       MODEL LAYER                             │
│                                                              │
│   SARIMA(X)    LightGBM     Temporal Fusion    Ensemble      │
│   (baseline)   (tabular)    Transformer        (stacker)     │
│                             (deep learning)    (meta-ML)     │
│                                                              │
│   src/models/{sarima,lgbm,tft,ensemble}_model.py            │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                   EXPERIMENT TRACKING (MLflow)               │
│   mlflow ui  →  compare MAE/RMSE/skill score across runs    │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                      API LAYER (FastAPI)                     │
│   GET  /forecast         →  3-day predictions + intervals    │
│   GET  /metrics          →  live accuracy vs. commercial app │
│   GET  /history          →  historical readings + bias       │
│   POST /forecast/feedback →  submit actual reading           │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
thermosense/
├── PLAN.md                          # Full implementation roadmap (reference this)
├── README.md
├── requirements.txt
├── .env.example                     # Copy to .env; add OWM_API_KEY if needed
├── config/
│   └── config.yaml                  # All tunable parameters (location, model hyperparams)
│
├── data/
│   ├── legacy/                      # Original 40-day hand-recorded sensor CSVs
│   │   └── temperature-data-for-TSA.csv
│   ├── raw/                         # API JSON responses (git-ignored)
│   ├── processed/                   # Merged daily_merged.parquet (git-ignored)
│   └── features/                    # Feature-engineered parquet (git-ignored)
│
├── src/
│   ├── data/
│   │   ├── fetcher.py               # Open-Meteo + OWM API calls with retry
│   │   └── preprocess.py            # Merge, gap-fill, validate, save pipeline
│   ├── features/
│   │   └── engineer.py              # Lag, rolling, calendar, api_bias features
│   ├── models/
│   │   ├── base_model.py            # Abstract interface
│   │   ├── sarima_model.py          # SARIMAX with exogenous regressors
│   │   ├── lgbm_model.py            # LightGBM tabular forecast
│   │   ├── tft_model.py             # Temporal Fusion Transformer
│   │   └── ensemble.py              # Ridge meta-learner over OOF predictions
│   ├── evaluation/
│   │   └── metrics.py               # MAE, RMSE, MAPE, Skill Score, Coverage
│   └── api/
│       ├── main.py                  # FastAPI app with lifespan model loading
│       └── routes/
│           ├── forecast.py          # GET /forecast, POST /feedback
│           ├── history.py           # GET /history
│           └── metrics.py           # GET /metrics
│
├── scripts/
│   └── run_pipeline.py              # CLI: backfill or daily data update
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Original analysis (refactored from Colab)
│   ├── 02_feature_engineering.ipynb # Feature matrix walkthrough
│   ├── 03_model_comparison.ipynb    # All models head-to-head
│   ├── 04_error_analysis.ipynb      # Residuals, SHAP, confidence bands
│   └── 05_api_demo.ipynb            # Live API call demo
│
└── tests/
    ├── test_fetcher.py              # Mocked HTTP; retry, parsing, 9 PM resampling
    ├── test_preprocess.py           # Merge logic, gap-fill, validation, pipeline
    ├── test_features.py             # No-leakage checks, bias feature correctness
    └── test_api.py                  # FastAPI endpoints via TestClient
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data fetching | [Open-Meteo API](https://open-meteo.com) (free, no key) · [OpenWeatherMap](https://openweathermap.org/api) (optional baseline) |
| Data storage | Parquet (via pandas + pyarrow) |
| Statistical models | statsmodels · pmdarima |
| Gradient boosting | LightGBM |
| Deep learning | Temporal Fusion Transformer (pytorch-forecasting + PyTorch) |
| Experiment tracking | MLflow |
| API framework | FastAPI + Uvicorn |
| Scheduling | APScheduler |
| Config management | PyYAML + python-dotenv |
| Testing | pytest · pytest-asyncio · httpx |
| Visualization | matplotlib · seaborn · plotly |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/yourusername/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and set your location (defaults to Bangalore).
# OWM_API_KEY is optional — only needed for commercial-app comparison.
```

Edit `config/config.yaml` to change the target location:

```yaml
location:
  name: "Bangalore"
  lat: 12.9716
  lon: 77.5946
  timezone: "Asia/Kolkata"
```

### 3. Run the data pipeline (Phase 1)

**First-time backfill** — pulls 365 days of history from Open-Meteo, merges with the existing 40-day sensor CSV, and saves to `data/processed/daily_merged.parquet`:

```bash
python scripts/run_pipeline.py --mode backfill
```

With custom date range:
```bash
python scripts/run_pipeline.py --mode backfill --start 2023-06-01 --end 2024-07-11
```

**Daily incremental update** (run nightly after taking your reading):
```bash
python scripts/run_pipeline.py --mode daily
```

Expected output:
```
============================================================
  ThermoSense — BACKFILL mode
============================================================
[fetcher] Fetching historical data: 2023-07-11 → 2024-07-10 (12.9716, 77.5946)
[fetcher] Raw response saved → data/raw/open_meteo_historical_2023-07-11_2024-07-10.json
[fetcher] Parsed 365 daily rows (2023-07-11 → 2024-07-10)

[preprocess] === Pipeline start ===
[preprocess] Loaded legacy CSV: 40 rows (2024-06-02 → 2024-07-11)
[preprocess] Merged: 365 total rows (40 sensor readings, 325 API-only)
[preprocess] Validation passed — 365 rows, 40 sensor readings, temp range 24.0–31.0°C
[preprocess] Saved → data/processed/daily_merged.parquet (365 rows)
[preprocess] === Pipeline complete ===
```

### 4. Run the tests

```bash
pytest tests/ -v
```

### 5. Launch the API (Phase 6 — coming next)

```bash
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs
```

### 6. Track experiments (Phase 5 — coming next)

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## The Key Innovation: `api_bias` Feature

The single most impactful feature in this system is:

```
api_bias = sensor_actual_temp − open_meteo_api_temp
api_bias_roll7_mean = rolling 7-day mean of api_bias
```

**Why this works**: Open-Meteo's temperature at your lat/lon is interpolated from gridded NWP model output — effectively an average over a ~1 km² cell. Your rooftop sensor reading is a point measurement subject to urban heat island effects, local shade, proximity to heat-emitting buildings, etc. This offset is not random — it has a **systematic component** that the model can learn.

For example, if your location consistently reads 1.5 °C warmer than the API during June evenings (concrete rooftop absorbing daytime heat), the model learns to apply that correction automatically.

This is why ThermoSense can outperform the commercial app at your specific location even with far fewer training observations.

---

## Data Sources

### Open-Meteo (primary — completely free)
- No API key required
- 80+ years of historical data via archive endpoint
- 16-day forecast horizon
- Variables: temperature, humidity, dew point, precipitation, pressure, cloud cover, wind speed, UV index
- Documentation: https://open-meteo.com/en/docs

### Existing sensor dataset (legacy)
- 40 days of 9 PM temperature readings (June–July 2024, Bangalore)
- Hand-recorded; treated as ground truth for the overlap period
- Stored in `data/legacy/temperature-data-for-TSA.csv`
- Also includes 1–3 day predictions from a commercial weather app (used as baseline)

### OpenWeatherMap (optional)
- Free tier: 1,000 API calls/day
- Used only to fetch the commercial-app baseline prediction for metric comparison
- Set `OWM_API_KEY` in `.env` to enable

---

## Model Comparison Target

| Model | Day-1 RMSE | Day-2 RMSE | Day-3 RMSE | Status |
|-------|-----------|-----------|-----------|--------|
| ARIMA(1,0,0) — original baseline | 1.34 °C | 1.51 °C | 1.86 °C | ✅ Done |
| SARIMA(X) + humidity/pressure | ~0.65 °C | ~0.85 °C | ~1.1 °C | Phase 3 |
| LightGBM + full feature matrix | ~0.70 °C | ~0.90 °C | ~1.2 °C | Phase 3 |
| Temporal Fusion Transformer | **~0.50 °C** | **~0.70 °C** | **~1.0 °C** | Phase 3 |
| Ensemble (TFT + LGB + SARIMA) | **~0.45 °C** | **~0.65 °C** | **~0.9 °C** | Phase 3 |
| Commercial weather app | ~1.0 °C | ~1.2 °C | ~1.5 °C | Benchmark |

---

## Implementation Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Repository scaffold, config, requirements | ✅ Complete |
| **Phase 1** | Data pipeline — Open-Meteo fetcher + preprocessing | ✅ Complete |
| **Phase 2** | Feature engineering — lag, rolling, calendar, api_bias | In progress |
| **Phase 3** | Model development — SARIMA(X), LightGBM, TFT, Ensemble | Planned |
| **Phase 4** | Evaluation framework — CV, metric comparison table | Planned |
| **Phase 5** | MLflow experiment tracking | Planned |
| **Phase 6** | FastAPI service — forecast, feedback, metrics endpoints | Planned |
| **Phase 7** | Notebook refactoring — 5 clean analysis notebooks | Planned |
| **Phase 8** | Tests — fetcher, features, API | Partial |

Full step-by-step details with architecture diagrams and code snippets are in [`PLAN.md`](PLAN.md).

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR for major changes.

For implementation phases, reference `PLAN.md` using the phase command syntax defined in Section 5 of that document.

---

## References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*
- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [pytorch-forecasting Documentation](https://pytorch-forecasting.readthedocs.io)
- [MLflow Documentation](https://mlflow.org/docs/latest)
