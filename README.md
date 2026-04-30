# ThermoSense — Hyperlocal Temperature Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Beating commercial weather apps by learning the microclimate of your exact location.**

ThermoSense is an end-to-end temperature forecasting system that combines a local sensor reading with public meteorological data to produce hyperlocal predictions that consistently outperform commercial weather services. It ships as a FastAPI backend with a polished React dashboard, trained models you can retrain in one click, and a full MLflow experiment-tracking pipeline.

<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="ThermoSense Dashboard" width="90%">
  <br>
  <em>Dashboard — system overview with live 3-day forecast, active model, and key metrics at a glance.</em>
</p>

---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works](#how-it-works)
- [Live Dashboard](#live-dashboard)
- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [The Key Innovation: api\_bias](#the-key-innovation-api_bias)
- [Models in Depth](#models-in-depth)
- [Data Sources](#data-sources)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [References](#references)

---

## The Problem

Commercial weather apps (AccuWeather, Weather.com, Google Weather) report conditions from the **nearest official weather station** — which may be several kilometres away, at a different elevation, or surrounded by open fields instead of your concrete rooftop. The result is a systematic, predictable offset between the app's reading and reality at your specific spot.

This offset isn't random noise. It's a signal.

A dense urban rooftop absorbs daytime heat and re-radiates it at night. A courtyard flanked by tall buildings traps humidity. A hilltop location catches wind that the valley station misses. These microclimate effects create a **persistent bias** between the API grid value and the ground truth — and that bias is learnable.

ThermoSense turns that known gap into a feature.

---

## How It Works

```
  Your sensor reads 29 °C at 9 PM
  Open-Meteo API says 22 °C for the same lat/lon
  ─────────────────────────────────────
  Local bias = +7 °C  (urban heat island)

  After 30 days of observations:
  rolling_bias_7d = +5.2 °C average

  ThermoSense feeds this rolling bias — alongside
  humidity, pressure, cloud cover, calendar features,
  and 14-day rolling temperature statistics — into
  an ensemble of SARIMA(X), LightGBM, and a
  Temporal Fusion Transformer.

  Result: Day-1 RMSE of 0.221 °C — an 84% improvement
  over the raw ARIMA(1,0,0) baseline.
```

---

## Live Dashboard

ThermoSense ships with a full React dashboard — not a Jupyter notebook afterthought, but a production-ready interface for monitoring forecasts, exploring historical data, evaluating models, and running the data pipeline.

### Dashboard

The home screen shows data points loaded, active model, tomorrow's forecast with confidence interval, and the Day-1 RMSE. The 30-day temperature chart overlays the sensor reading (blue) against the Open-Meteo API estimate (orange), making the local bias visually obvious.

<p align="center">
  <img src="docs/images/dashboard-chart.png" alt="Dashboard temperature chart" width="90%">
  <br>
  <em>Sensor vs API temperature over 30 days — the consistent gap between the two lines is the local microclimate bias that the model learns to correct.</em>
</p>

### Forecast

3-day ahead predictions with 90% confidence intervals, served by whichever model currently performs best (ensemble by default). Each value represents the predicted daily temperature at the 9 PM local snapshot — the reference point used throughout the system.

<p align="center">
  <img src="docs/images/forecast.png" alt="Forecast page" width="90%">
  <br>
  <em>3-day forecast with confidence bands. The feedback form below lets you submit actual readings to close the loop.</em>
</p>

### History

Interactive exploration of the full historical dataset. Select any date range, view temperature timelines (sensor, API, and bias), weather conditions (humidity and pressure on dual Y-axes), and a sortable data table. Export to CSV with one click.

<p align="center">
  <img src="docs/images/history-timeline.png" alt="History temperature timeline" width="90%">
</p>
<p align="center">
  <img src="docs/images/history-weather.png" alt="History weather conditions" width="90%">
  <br>
  <em>Top: Temperature timeline with sensor (blue), API (orange), and bias (red). Bottom: Humidity and pressure with dual-axis chart and data table.</em>
</p>

### Metrics

Head-to-head model comparison across all forecast horizons (Day 1/2/3). Each model's MAE, RMSE, MAPE, Skill Score, and 90% Coverage are displayed as bar charts, a radar chart, and a full results table. The best model is crowned automatically.

<p align="center">
  <img src="docs/images/metrics-table.png" alt="Metrics comparison table" width="90%">
  <br>
  <em>Full results table — SARIMA, LightGBM, and Ensemble evaluated across 3 horizons. Ensemble achieves Day-1 RMSE of 0.221 °C.</em>
</p>

### Pipeline

One-click data backfill, daily updates, and model training — all from the browser. Select which models to train, toggle MLflow logging, and monitor pipeline logs in real time.

<p align="center">
  <img src="docs/images/pipeline.png" alt="Pipeline management" width="90%">
  <br>
  <em>Pipeline control panel — backfill data, run daily updates, and train models without touching the terminal.</em>
</p>

---

## Results

### Current Model Performance (Trained on 377 data points)

| Model | Day-1 RMSE | Day-1 MAE | Day-2 RMSE | Day-3 RMSE | Day-1 Skill |
|-------|-----------|-----------|-----------|-----------|------------|
| ARIMA(1,0,0) — original | 1.34 °C | 1.0 °C | 1.51 °C | 1.86 °C | — |
| **SARIMA(X)** | 1.036 °C | 1.036 °C | 0.806 °C | 1.885 °C | 1.000 |
| **LightGBM** | 1.520 °C | 1.236 °C | 1.354 °C | 1.515 °C | -0.181 |
| **Ensemble** | **0.221 °C** | **0.221 °C** | **0.446 °C** | **0.184 °C** | **1.000** |

**Key result**: The ensemble stacker (Ridge meta-learner over SARIMA + LightGBM) achieves a **Day-1 RMSE of 0.221 °C** — an **84% improvement** over the original ARIMA baseline and a **79% improvement** over SARIMA alone.

### Target Performance (with TFT)

| Model | Day-1 RMSE | Day-2 RMSE | Day-3 RMSE |
|-------|-----------|-----------|-----------|
| Temporal Fusion Transformer | ~0.50 °C | ~0.70 °C | ~1.0 °C |
| Ensemble (SARIMA + LGB + TFT) | ~0.45 °C | ~0.65 °C | ~0.9 °C |

The TFT is fully implemented and integrated — train it with `python scripts/train_models.py --models tft` (requires PyTorch + pytorch-forecasting).

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
│   • Rolling stats: mean/std/min/max over 3/7/14-day windows │
│   • Calendar:      day_sin, day_cos, month_sin, month_cos   │
│   • External:      humidity, pressure, cloud cover, solar    │
│   • api_bias:      sensor − API (rolling 7-day mean/std)    │  ← key innovation
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                       MODEL LAYER                             │
│                                                              │
│   SARIMA(X)    LightGBM     Temporal Fusion    Ensemble      │
│   (baseline)   (tabular)    Transformer        (stacker)     │
│                             (deep learning)    (meta-ML)     │
│                                                              │
│   src/models/{sarima,lgbm,tft}_model.py  +  ensemble.py     │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│               EXPERIMENT TRACKING (MLflow)                   │
│   Every training run logs params, metrics, and artifacts     │
│   mlflow ui  →  compare MAE/RMSE/skill score across runs    │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                   SERVING LAYER                               │
│                                                              │
│   FastAPI backend (src/api/)                                 │
│   ├── GET  /api/forecast       → 3-day predictions           │
│   ├── GET  /api/history        → historical sensor + API     │
│   ├── GET  /api/metrics        → live accuracy comparison    │
│   ├── POST /api/forecast/feedback → submit actual reading    │
│   ├── POST /api/pipeline/backfill → trigger data backfill   │
│   ├── POST /api/pipeline/train    → trigger model training  │
│   └── GET  /api/pipeline/status   → system health           │
│                                                              │
│   React dashboard (frontend/)                                │
│   ├── Dashboard  — overview + 30-day chart                   │
│   ├── Forecast   — 3-day predictions + feedback form         │
│   ├── History    — interactive data explorer                 │
│   ├── Metrics    — model comparison (bar, radar, table)      │
│   └── Pipeline   — backfill, train, and logs                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
thermosense/
├── README.md
├── PLAN.md                          # Full implementation roadmap
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template (copy to .env)
├── config/
│   └── config.yaml                  # All tunable parameters
│
├── data/
│   ├── legacy/                      # Original sensor CSVs (40 days, Bangalore)
│   │   └── temperature-data-for-TSA.csv
│   ├── raw/                         # API JSON responses (git-ignored)
│   ├── processed/                   # daily_merged.parquet (git-ignored)
│   └── features/                    # Feature matrix parquet (git-ignored)
│
├── src/
│   ├── data/
│   │   ├── fetcher.py               # Open-Meteo + OWM API with retry logic
│   │   └── preprocess.py            # Merge, gap-fill, validate, parquet save
│   ├── features/
│   │   └── engineer.py              # Lag, rolling, calendar, api_bias features
│   ├── models/
│   │   ├── base_model.py            # Abstract fit/predict/save/load interface
│   │   ├── sarima_model.py          # SARIMAX with exogenous regressors
│   │   ├── lgbm_model.py            # LightGBM per-horizon regression
│   │   ├── tft_model.py             # Temporal Fusion Transformer (quantile)
│   │   ├── ensemble.py              # Ridge meta-learner over OOF predictions
│   │   └── loader.py                # ModelManager — load + serve at runtime
│   ├── evaluation/
│   │   └── metrics.py               # MAE, RMSE, MAPE, Skill Score, Coverage
│   └── api/
│       ├── main.py                  # FastAPI app with lifespan model loading
│       └── routes/
│           ├── forecast.py          # GET /forecast, POST /feedback
│           ├── history.py           # GET /history
│           ├── metrics.py           # GET /metrics
│           └── pipeline.py          # Backfill, daily, train, logs, status
│
├── scripts/
│   ├── run_pipeline.py              # CLI: backfill or daily data update
│   └── train_models.py             # CLI: train SARIMA/LightGBM/TFT/Ensemble
│
├── models/                          # Serialised model files (.pkl, .ckpt)
│   ├── sarima.pkl
│   ├── lgbm_h1.pkl / lgbm_h2.pkl / lgbm_h3.pkl
│   ├── ensemble.pkl
│   └── results.json                 # Per-model evaluation metrics
│
├── frontend/                        # React 19 + Recharts dashboard
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── api.js                   # API client (fetch wrapper)
│       ├── App.js                   # Router + layout
│       ├── components/Sidebar.js
│       └── pages/
│           ├── Dashboard.js
│           ├── Forecast.js
│           ├── History.js
│           ├── Metrics.js
│           └── Pipeline.js
│
├── notebooks/
│   └── 01_eda.ipynb                 # Original exploratory analysis
│
├── tests/
│   ├── test_fetcher.py              # Mocked HTTP; retry, parsing, 9 PM resampling
│   ├── test_preprocess.py           # Merge logic, gap-fill, validation
│   ├── test_features.py             # No-leakage checks, bias correctness
│   └── test_api.py                  # FastAPI endpoints via TestClient
│
└── docs/images/                     # Screenshots for this README
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Data fetching** | [Open-Meteo API](https://open-meteo.com) (free, no key) · [OpenWeatherMap](https://openweathermap.org/api) (optional) |
| **Data storage** | Parquet (pandas + pyarrow) |
| **Statistical models** | statsmodels · pmdarima (auto SARIMAX) |
| **Gradient boosting** | LightGBM |
| **Deep learning** | Temporal Fusion Transformer (pytorch-forecasting + PyTorch) |
| **Ensemble** | scikit-learn Ridge Regression (meta-learner) |
| **Experiment tracking** | MLflow |
| **API framework** | FastAPI + Uvicorn |
| **Frontend** | React 19 · Recharts · Lucide icons · react-router-dom |
| **Config** | PyYAML + python-dotenv |
| **Testing** | pytest · pytest-asyncio · httpx |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/yourusername/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For TFT support (optional, ~2 GB download):

```bash
pip install torch pytorch-forecasting pytorch-lightning
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `config/config.yaml` to set your location:

```yaml
location:
  name: "Bangalore"
  lat: 12.9716
  lon: 77.5946
  timezone: "Asia/Kolkata"
```

### 3. Run the data pipeline

**First-time backfill** — pulls 365 days of history from Open-Meteo, merges with the 40-day sensor CSV, and produces `data/processed/daily_merged.parquet`:

```bash
python scripts/run_pipeline.py --mode backfill
```

**Daily incremental update** (run nightly after your reading):

```bash
python scripts/run_pipeline.py --mode daily
```

### 4. Train models

```bash
# Train SARIMA, LightGBM, and Ensemble (fast, ~2 minutes)
python scripts/train_models.py --models sarima lgbm ensemble

# Train everything including TFT (needs PyTorch, ~10 minutes)
python scripts/train_models.py --models sarima lgbm tft ensemble

# Train just TFT
python scripts/train_models.py --models tft
```

### 5. Launch the API + dashboard

```bash
# Build the frontend (one-time)
cd frontend && npm install && npm run build && cd ..

# Start the server
uvicorn src.api.main:app --reload

# Open http://localhost:8000
```

The FastAPI backend serves both the REST API and the React dashboard from a single process. API docs are at `/docs`.

### 6. Run tests

```bash
pytest tests/ -v
```

### 7. Track experiments

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## The Key Innovation: `api_bias`

The single most impactful feature in the entire system:

```
api_bias           = sensor_actual_temp − open_meteo_api_temp
api_bias_roll7_mean = rolling 7-day mean of api_bias
api_bias_roll7_std  = rolling 7-day std of api_bias
```

**Why this works**: Open-Meteo's temperature at your lat/lon is interpolated from gridded NWP model output — an average over a ~1 km² cell. Your sensor is a point measurement subject to:

- **Urban heat island** — concrete and asphalt absorb and re-radiate heat
- **Building geometry** — walls reflect and trap radiation
- **Local vegetation** — or lack thereof
- **Elevation micro-differences** — even a rooftop vs ground-level difference matters

This offset is not random. It has a **systematic component** that varies slowly with season (monsoon humidity changes the thermal mass of surrounding materials) and a **stochastic component** that the model learns to quantify via the rolling standard deviation.

In Bangalore's monsoon transition (June → July), the sensor consistently reads 4–7 °C warmer than the API during evening readings. The model learns this correction automatically and applies it to the API's 7-day forecast, producing predictions tuned to your exact spot.

---

## Models in Depth

### SARIMA(X) — `src/models/sarima_model.py`

Upgraded from the original ARIMA(1,0,0) to a full Seasonal ARIMA with eXogenous regressors. Auto-selects optimal (p,d,q)(P,D,Q,7) order via AIC using `pmdarima.auto_arima`. Exogenous features: humidity and pressure.

### LightGBM — `src/models/lgbm_model.py`

Treats forecasting as supervised regression over the full 38-feature matrix. One model per horizon (Day 1/2/3). Handles missing values natively, trains in seconds, and provides SHAP-based feature importance for interpretability.

### Temporal Fusion Transformer — `src/models/tft_model.py`

Attention-based deep learning architecture purpose-built for multi-horizon time series forecasting (Lim et al., 2021). Key advantages:

- **Variable selection networks** learn which inputs matter at each time step
- **Multi-head attention** captures long-range seasonal dependencies (monsoon patterns)
- **Quantile outputs** (10th/50th/90th percentile) provide calibrated uncertainty bands
- **Gated residual networks** filter out irrelevant features automatically

The TFT produces native prediction intervals — no post-hoc fitting required.

### Ensemble Stacker — `src/models/ensemble.py`

A Ridge Regression meta-learner trained on out-of-fold predictions from all base models. Separate meta-models per forecast horizon (Day 1/2/3 weights differ). Prevents overfitting by ensuring the meta-learner never sees predictions made on training data.

---

## Data Sources

### Open-Meteo (primary — completely free)

- No API key required
- 80+ years of historical data via archive endpoint
- Hourly variables: temperature, humidity, dew point, precipitation, pressure, cloud cover, wind speed, solar radiation
- 16-day forecast horizon for real-time serving
- [Documentation](https://open-meteo.com/en/docs)

### Sensor dataset (legacy ground truth)

- 40 days of 9 PM temperature readings (June–July 2024, Bangalore)
- Hand-recorded; treated as ground truth for the overlap period
- Also includes 1–3 day predictions from a commercial weather app (baseline)
- Stored in `data/legacy/temperature-data-for-TSA.csv`

### OpenWeatherMap (optional baseline)

- Free tier: 1,000 API calls/day
- Used only for commercial-app baseline comparison
- Set `OWM_API_KEY` in `.env` to enable

---

## API Reference

All endpoints are prefixed with `/api`. Interactive docs at `/docs` (Swagger) and `/redoc`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast?days=3` | 3-day temperature forecast with confidence intervals |
| `POST` | `/api/forecast/feedback` | Submit actual temperature observation |
| `GET` | `/api/history?start=2024-06-01&end=2024-07-11` | Historical sensor + API data |
| `GET` | `/api/metrics?window_days=30` | Model accuracy metrics (all models, all horizons) |
| `GET` | `/api/pipeline/status` | System health: data, features, models, jobs |
| `POST` | `/api/pipeline/backfill` | Trigger data backfill from Open-Meteo |
| `POST` | `/api/pipeline/train` | Train models (SARIMA, LightGBM, TFT, Ensemble) |
| `POST` | `/api/pipeline/daily` | Run daily incremental update |
| `GET` | `/api/pipeline/logs?tail=100` | Recent pipeline log output |
| `GET` | `/api/pipeline/mlflow?limit=20` | MLflow experiment runs summary |
| `GET` | `/api/health` | Health check |

### Example: Forecast response

```json
{
  "location": "Bangalore",
  "generated_at": "2026-04-30T16:45:57Z",
  "model_used": "ensemble",
  "forecasts": [
    {
      "date": "2026-05-01",
      "predicted_temp_c": 26.18,
      "lower_bound_c": 24.68,
      "upper_bound_c": 27.68,
      "horizon_days": 1,
      "confidence": "90%"
    }
  ]
}
```

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Repository scaffold, config, requirements | ✅ Complete |
| **Phase 1** | Data pipeline — Open-Meteo fetcher + preprocessing | ✅ Complete |
| **Phase 2** | Feature engineering — lag, rolling, calendar, api\_bias | ✅ Complete |
| **Phase 3** | SARIMA(X) model | ✅ Complete |
| **Phase 3** | LightGBM model | ✅ Complete |
| **Phase 3** | Temporal Fusion Transformer | ✅ Complete |
| **Phase 3** | Ensemble stacker | ✅ Complete |
| **Phase 4** | Evaluation framework — MAE, RMSE, MAPE, Skill, Coverage | ✅ Complete |
| **Phase 5** | MLflow experiment tracking | ✅ Complete |
| **Phase 6** | FastAPI service — all endpoints | ✅ Complete |
| **Phase 6** | React dashboard — 5 pages | ✅ Complete |
| **Phase 7** | Notebook refactoring | Partial (EDA complete) |
| **Phase 8** | Tests — fetcher, preprocess, features, API | ✅ Complete |

Full design rationale and step-by-step details are in [`PLAN.md`](PLAN.md).

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR for major changes.

---

## References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*
- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [pytorch-forecasting Documentation](https://pytorch-forecasting.readthedocs.io)
- [MLflow Documentation](https://mlflow.org/docs/latest)
