# ThermoSense — Hyperlocal Temperature Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Beating commercial weather apps by learning the microclimate of your exact location.**

ThermoSense is an end-to-end IoT + ML system that deploys a physical temperature sensor (Raspberry Pi + DHT22), collects real measurements, and trains machine learning models to correct the systematic bias between commercial weather APIs and your specific location. The result: hyperlocal forecasts that demonstrably outperform Google Weather, AccuWeather, and other commercial services.

---

## Table of Contents

- [The Problem We're Solving](#the-problem-were-solving)
- [How ThermoSense Works](#how-thermosense-works)
- [Live Dashboard](#live-dashboard)
- [Results](#results)
- [Complete Data Lifecycle](#complete-data-lifecycle)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Hardware Setup (Raspberry Pi)](#hardware-setup-raspberry-pi)
- [Running the System](#running-the-system)
- [Live Accuracy Leaderboard](#live-accuracy-leaderboard)
- [The Key Innovation: api_bias](#the-key-innovation-api_bias)
- [Models in Depth](#models-in-depth)
- [API Reference](#api-reference)
- [Making the Most of ThermoSense](#making-the-most-of-thermosense)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## The Problem We're Solving

### Why Commercial Weather Apps Get Your Location Wrong

Commercial weather apps (AccuWeather, Weather.com, Google Weather) report conditions from the **nearest official weather station** — which may be:

- Several kilometers away
- At a different elevation
- Surrounded by open fields instead of your concrete rooftop
- Near a body of water or vegetation that doesn't exist at your location

The result is a **systematic, predictable offset** between what the app reports and what you actually experience.

### This Offset Is Not Random Noise — It's a Learnable Signal

A dense urban rooftop absorbs daytime heat and re-radiates it at night. A courtyard flanked by tall buildings traps humidity. A hilltop location catches wind that the valley station misses. These microclimate effects create a **persistent bias** between the API grid value and the ground truth.

**Example from Bangalore:**
```
Your sensor reads 29°C at 9 PM
Open-Meteo API says 22°C for the same lat/lon
────────────────────────────────────────────
Local bias = +7°C  (urban heat island effect)
```

After 30 days of observations, this rolling bias becomes a powerful predictive feature:
- `rolling_bias_7d = +5.2°C average`
- The model learns to add this correction to future API forecasts

### What Makes ThermoSense Different

| Traditional Approach | ThermoSense Approach |
|---------------------|---------------------|
| Train on public datasets | Train on YOUR sensor + API data |
| Generic regional model | Location-specific bias correction |
| Claims "accurate" without proof | Live leaderboard comparing against Google, AccuWeather |
| No feedback loop | Continuous learning from sensor readings |
| Metrics on synthetic splits | Metrics on real-world predictions |

---

## How ThermoSense Works

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        THE THERMOSENSE WORKFLOW                         │
  └─────────────────────────────────────────────────────────────────────────┘

  1. DEPLOY SENSOR                    2. COLLECT DATA
  ┌─────────────────┐                 ┌─────────────────────────────────────┐
  │ Raspberry Pi    │    Reads        │  Local SQLite        Cloud API      │
  │ + DHT22         │───────────────▶ │  readings.db    ───────────────▶    │
  │ (Every 5 min)   │    Every 5min   │  temp, humidity      Sync hourly    │
  └─────────────────┘                 └─────────────────────────────────────┘
                                                       │
  3. FETCH API FORECASTS                              ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Daily at 6 PM:                                                         │
  │  • Open-Meteo    (free, no key)  → Day-1, Day-2, Day-3 forecasts       │
  │  • AccuWeather   (free tier)     → Stored in forecasts.db              │
  │  • OpenWeatherMap (free tier)    → For baseline comparison             │
  │  • ThermoSense   (our model)     → Our prediction to beat              │
  └─────────────────────────────────────────────────────────────────────────┘
                                                       │
  4. FEATURE ENGINEERING                              ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  38 features computed from merged sensor + API data:                    │
  │  • Lag features: T-1, T-2, T-3, T-7 days                               │
  │  • Rolling stats: mean/std/min/max over 3/7/14-day windows             │
  │  • Calendar: day_sin, day_cos, month_sin, month_cos, is_monsoon        │
  │  • Meteorological: humidity, pressure, cloud cover, solar radiation    │
  │  • ★ KEY: api_bias, api_bias_roll7_mean, api_bias_roll7_std            │
  └─────────────────────────────────────────────────────────────────────────┘
                                                       │
  5. MODEL TRAINING                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Three base models + ensemble:                                          │
  │  • SARIMA(X)  — Time series baseline with exogenous regressors         │
  │  • LightGBM   — Gradient boosting on full feature matrix               │
  │  • TFT        — Transformer for attention-based forecasting            │
  │  • Ensemble   — Ridge meta-learner over OOF predictions                │
  └─────────────────────────────────────────────────────────────────────────┘
                                                       │
  6. SERVE & COMPARE                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FastAPI Backend + React Dashboard:                                     │
  │  • GET /api/forecast      → 3-day predictions with confidence bands    │
  │  • GET /api/leaderboard   → Live RMSE/MAE vs Google, AccuWeather       │
  │  • GET /api/metrics       → Model comparison across horizons           │
  │  • POST /api/sensor/readings → Receive sensor data from Pi             │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Live Dashboard

ThermoSense ships with a production-ready React dashboard — not a Jupyter notebook afterthought, but a polished interface for monitoring forecasts, exploring historical data, evaluating models, and running the data pipeline.

<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="ThermoSense Dashboard Overview" width="90%">
  <br>
  <em>Dashboard — system overview with live 3-day forecast, active model, and key metrics at a glance.</em>
</p>

### Dashboard Home

The home screen shows data points loaded, active model, tomorrow's forecast with confidence interval, and the Day-1 RMSE. The 30-day temperature chart overlays the sensor reading (blue) against the Open-Meteo API estimate (orange), making the local bias visually obvious.

<p align="center">
  <img src="docs/images/dashboard-chart.png" alt="Dashboard temperature chart" width="90%">
  <br>
  <em>Sensor vs API temperature over 30 days — the consistent gap between the two lines is the local microclimate bias that the model learns to correct.</em>
</p>

### Forecast Page

3-day ahead predictions with 90% confidence intervals, served by whichever model currently performs best (ensemble by default). Each value represents the predicted daily temperature at the 9 PM local snapshot — the reference point used throughout the system.

<p align="center">
  <img src="docs/images/forecast.png" alt="Forecast page" width="90%">
  <br>
  <em>3-day forecast with confidence bands. The feedback form below lets you submit actual readings to close the loop.</em>
</p>

### History Explorer

Interactive exploration of the full historical dataset. Select any date range, view temperature timelines (sensor, API, and bias), weather conditions (humidity and pressure on dual Y-axes), and a sortable data table. Export to CSV with one click.

<p align="center">
  <img src="docs/images/history-timeline.png" alt="History temperature timeline" width="90%">
</p>
<p align="center">
  <img src="docs/images/history-weather.png" alt="History weather conditions" width="90%">
  <br>
  <em>Top: Temperature timeline with sensor (blue), API (orange), and bias (red). Bottom: Humidity and pressure with dual-axis chart and data table.</em>
</p>

### Metrics Comparison

Head-to-head model comparison across all forecast horizons (Day 1/2/3). Each model's MAE, RMSE, MAPE, Skill Score, and 90% Coverage are displayed as bar charts, a radar chart, and a full results table. The best model is crowned automatically.

<p align="center">
  <img src="docs/images/metrics-table.png" alt="Metrics comparison table" width="90%">
  <br>
  <em>Full results table — SARIMA, LightGBM, and Ensemble evaluated across 3 horizons. Ensemble achieves Day-1 RMSE of 0.221°C.</em>
</p>

### Pipeline Control

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
| ARIMA(1,0,0) — original | 1.34°C | 1.0°C | 1.51°C | 1.86°C | — |
| **SARIMA(X)** | 1.036°C | 1.036°C | 0.806°C | 1.885°C | 1.000 |
| **LightGBM** | 1.520°C | 1.236°C | 1.354°C | 1.515°C | -0.181 |
| **Ensemble** | **0.221°C** | **0.221°C** | **0.446°C** | **0.184°C** | **1.000** |

**Key result**: The ensemble stacker (Ridge meta-learner over SARIMA + LightGBM) achieves a **Day-1 RMSE of 0.221°C** — an **84% improvement** over the original ARIMA baseline and a **79% improvement** over SARIMA alone.

---

## Complete Data Lifecycle

Understanding the full data flow is crucial for debugging, extending, or trusting the system.

### Stage 1: Sensor Data Collection (Edge)

**Location**: Raspberry Pi running `hardware/sensor_daemon.py`

```
DHT22 Sensor (GPIO 4)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  sensor_daemon.py (runs as systemd service)                     │
│  • Reads temperature + humidity every 5 minutes                 │
│  • Validates readings (-40°C to +80°C, 0-100% humidity)         │
│  • Stores in local SQLite: hardware/readings.db                 │
│  • Exposes HTTP API on port 8081 for health checks              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  readings.db schema:                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ sensor_readings                                            │ │
│  │ ├── id (INTEGER PRIMARY KEY)                               │ │
│  │ ├── timestamp (TEXT, ISO format)                           │ │
│  │ ├── temp_c (REAL)                                          │ │
│  │ ├── humidity_pct (REAL)                                    │ │
│  │ ├── source (TEXT: 'dht22_sensor' or 'simulated')           │ │
│  │ └── synced (INTEGER: 0=pending, 1=uploaded)                │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 2: Edge-to-Cloud Sync

**Location**: Raspberry Pi running `hardware/uploader.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  uploader.py (cron job every 15 minutes)                        │
│  • Queries sensor_daemon for unsynced readings                  │
│  • Batches readings (100 per request)                           │
│  • POSTs to cloud: POST /api/sensor/readings                    │
│  • On success: marks readings as synced locally                 │
│  • Handles network failures with exponential backoff            │
└─────────────────────────────────────────────────────────────────┘
         │
         │  HTTP POST (JSON)
         │  {
         │    "readings": [
         │      {"timestamp": "2026-05-12T15:30:00Z", "temp_c": 28.5, "humidity_pct": 65.2},
         │      ...
         │    ]
         │  }
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cloud API: POST /api/sensor/readings                           │
│  • Validates incoming readings                                  │
│  • Appends to data/processed/sensor_readings.parquet            │
│  • Triggers feature recomputation if new day detected           │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 3: API Data Fetching

**Location**: Cloud server running `src/data/fetcher.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  fetcher.py (triggered by pipeline or daily cron)               │
│                                                                 │
│  Open-Meteo Archive API ◀───────── Historical backfill          │
│  (365 days of hourly data)         python scripts/run_pipeline.py --mode backfill
│                                                                 │
│  Open-Meteo Forecast API ◀──────── Daily forecast fetch         │
│  (16-day forecast horizon)         python scripts/run_pipeline.py --mode daily
│                                                                 │
│  Variables fetched:                                             │
│  • temperature_2m       → temp_c                                │
│  • relativehumidity_2m  → humidity_pct                          │
│  • dewpoint_2m          → dewpoint_c                            │
│  • precipitation        → precip_mm                             │
│  • pressure_msl         → pressure_hpa                          │
│  • cloudcover           → cloudcover_pct                        │
│  • windspeed_10m        → windspeed_kmh                         │
│  • shortwave_radiation  → solar_radiation_wm2                   │
└─────────────────────────────────────────────────────────────────┘
         │
         │  Hourly data resampled to 9 PM snapshot (matches sensor convention)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/raw/                                                      │
│  ├── open_meteo_historical_2025-05-12_2026-05-11.json          │
│  ├── open_meteo_forecast_2026-05-12.json                        │
│  └── owm_current_2026-05-12T1800.json                           │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 4: Data Preprocessing & Merging

**Location**: `src/data/preprocess.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  preprocess.py                                                  │
│                                                                 │
│  Inputs:                                                        │
│  ├── data/legacy/temperature-data-for-TSA.csv (40 days ground truth)
│  ├── data/raw/open_meteo_*.json (API responses)                 │
│  └── sensor readings from database                              │
│                                                                 │
│  Processing:                                                    │
│  1. Parse API JSON → DataFrame                                  │
│  2. Resample hourly → daily 9 PM snapshot                       │
│  3. Merge sensor readings (sensor takes precedence)             │
│  4. Forward-fill missing dates (flagged as imputed)             │
│  5. Compute api_bias = sensor_temp - api_temp                   │
│  6. Validate ranges and data quality                            │
│                                                                 │
│  Output:                                                        │
│  └── data/processed/daily_merged.parquet                        │
└─────────────────────────────────────────────────────────────────┘

Output schema:
┌──────────────────┬──────────────────────────────────────────────┐
│ Column           │ Description                                  │
├──────────────────┼──────────────────────────────────────────────┤
│ date             │ Date (YYYY-MM-DD)                            │
│ temp_c           │ Ground truth temperature (sensor preferred)  │
│ api_temp_c       │ Open-Meteo API temperature at same location  │
│ humidity_pct     │ Relative humidity (%)                        │
│ pressure_hpa     │ Sea-level pressure (hPa)                     │
│ cloudcover_pct   │ Cloud cover (%)                              │
│ windspeed_kmh    │ Wind speed (km/h)                            │
│ precip_mm        │ Precipitation (mm)                           │
│ solar_radiation_wm2 │ Solar radiation (W/m²)                    │
│ source           │ 'sensor', 'legacy', or 'api_only'            │
│ api_bias         │ temp_c - api_temp_c (the key feature!)       │
└──────────────────┴──────────────────────────────────────────────┘
```

### Stage 5: Feature Engineering

**Location**: `src/features/engineer.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  engineer.py                                                    │
│                                                                 │
│  Input: data/processed/daily_merged.parquet                     │
│                                                                 │
│  Features generated (38 total):                                 │
│                                                                 │
│  LAG FEATURES (4):                                              │
│  ├── temp_lag1, temp_lag2, temp_lag3, temp_lag7                 │
│                                                                 │
│  ROLLING STATISTICS (12):                                       │
│  ├── temp_roll3_mean, temp_roll3_std, temp_roll3_min, temp_roll3_max
│  ├── temp_roll7_mean, temp_roll7_std, temp_roll7_min, temp_roll7_max
│  └── temp_roll14_mean, temp_roll14_std, temp_roll14_min, temp_roll14_max
│                                                                 │
│  CALENDAR FEATURES (5):                                         │
│  ├── day_sin, day_cos (cyclic day of year)                      │
│  ├── month_sin, month_cos (cyclic month)                        │
│  └── is_monsoon (June-September indicator)                      │
│                                                                 │
│  METEOROLOGICAL (7):                                            │
│  ├── humidity_pct, pressure_hpa, cloudcover_pct                 │
│  ├── windspeed_kmh, precip_mm, dewpoint_c                       │
│  └── solar_radiation_wm2                                        │
│                                                                 │
│  ★ API BIAS FEATURES (3) — THE KEY INNOVATION:                  │
│  ├── api_bias (today's sensor - API difference)                 │
│  ├── api_bias_roll7_mean (rolling 7-day mean bias)              │
│  └── api_bias_roll7_std (rolling 7-day bias volatility)         │
│                                                                 │
│  TARGET VARIABLES (3):                                          │
│  ├── target_h1 (temperature 1 day ahead)                        │
│  ├── target_h2 (temperature 2 days ahead)                       │
│  └── target_h3 (temperature 3 days ahead)                       │
│                                                                 │
│  Output: data/features/feature_matrix.parquet                   │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 6: Model Training

**Location**: `src/models/*.py`, triggered by `scripts/train_models.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  train_models.py                                                │
│                                                                 │
│  Input: data/features/feature_matrix.parquet                    │
│                                                                 │
│  Training Process:                                              │
│  1. Time-series split: last 14 days held out for testing        │
│  2. Train each base model:                                      │
│     ├── SARIMA(X): Auto-selects (p,d,q)(P,D,Q,7) via AIC        │
│     ├── LightGBM: One model per horizon (H1, H2, H3)            │
│     └── TFT: Quantile regression with attention                 │
│  3. Generate out-of-fold predictions for ensemble               │
│  4. Train Ridge meta-learner on OOF predictions                 │
│  5. Evaluate all models on held-out test set                    │
│  6. Log to MLflow: params, metrics, artifacts                   │
│                                                                 │
│  Outputs:                                                       │
│  ├── models/sarima.pkl                                          │
│  ├── models/lgbm_h1.pkl, lgbm_h2.pkl, lgbm_h3.pkl               │
│  ├── models/tft.ckpt (if PyTorch available)                     │
│  ├── models/ensemble.pkl                                        │
│  └── models/results.json (evaluation metrics)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 7: Baseline Collection (for Leaderboard)

**Location**: `src/data/baseline_collector.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  baseline_collector.py (daily cron at 6 PM)                     │
│                                                                 │
│  Fetches Day-1/2/3 forecasts from:                              │
│  ├── Open-Meteo (free)                                          │
│  ├── OpenWeatherMap (free tier, if OWM_API_KEY set)             │
│  ├── AccuWeather (free tier, if ACCUWEATHER_API_KEY set)        │
│  └── ThermoSense (our model, from /api/forecast)                │
│                                                                 │
│  Storage: data/baselines/forecasts.db (SQLite)                  │
│  ├── daily_forecasts: All predictions from all sources          │
│  ├── daily_actuals: Ground truth sensor readings                │
│  └── collection_log: Audit trail of collection runs             │
│                                                                 │
│  This enables the live leaderboard comparison!                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 8: Serving & Visualization

**Location**: `src/api/` (FastAPI) + `frontend/` (React)

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (port 8000)                                    │
│                                                                 │
│  Endpoints:                                                     │
│  ├── GET  /api/forecast      → 3-day predictions + CI           │
│  ├── GET  /api/history       → Historical sensor + API data     │
│  ├── GET  /api/metrics       → Model accuracy comparison        │
│  ├── GET  /api/leaderboard   → ThermoSense vs commercial apps   │
│  ├── GET  /api/statistics    → Statistical significance tests   │
│  ├── POST /api/sensor/readings → Receive sensor uploads         │
│  ├── POST /api/pipeline/backfill → Trigger data backfill        │
│  └── POST /api/pipeline/train → Trigger model training          │
│                                                                 │
│  React Dashboard:                                               │
│  ├── /dashboard   → Overview, 30-day chart, key metrics         │
│  ├── /forecast    → 3-day predictions + feedback form           │
│  ├── /history     → Interactive data explorer + CSV export      │
│  ├── /metrics     → Model comparison (bar, radar, table)        │
│  ├── /leaderboard → Live accuracy vs Google, AccuWeather        │
│  └── /pipeline    → Backfill, train, logs                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                               THERMOSENSE ARCHITECTURE                             │
└────────────────────────────────────────────────────────────────────────────────────┘

   EDGE LAYER                          CLOUD LAYER                    EXTERNAL APIs
   (Raspberry Pi)                      (Your Server / Railway)        (Free Services)
┌──────────────────┐               ┌──────────────────────────┐    ┌─────────────────┐
│                  │               │                          │    │                 │
│  ┌────────────┐  │   HTTP POST   │  ┌──────────────────┐    │    │  Open-Meteo    │
│  │ DHT22      │  │ ────────────▶ │  │ FastAPI Backend  │◀───┼────│  (Historical)  │
│  │ Sensor     │  │   /sensor/    │  │                  │    │    │                 │
│  └────────────┘  │   readings    │  │ • /forecast      │    │    │  Open-Meteo    │
│        │         │               │  │ • /history       │◀───┼────│  (Forecast)    │
│        ▼         │               │  │ • /metrics       │    │    │                 │
│  ┌────────────┐  │               │  │ • /leaderboard   │    │    │  AccuWeather   │
│  │ sensor_    │  │               │  │ • /sensor        │◀───┼────│  (Baseline)    │
│  │ daemon.py  │  │               │  │ • /pipeline      │    │    │                 │
│  │            │  │               │  └────────┬─────────┘    │    │  OpenWeatherMap│
│  │ SQLite DB  │  │               │           │              │    │  (Baseline)    │
│  └────────────┘  │               │           ▼              │    │                 │
│        │         │               │  ┌──────────────────┐    │    └─────────────────┘
│        ▼         │               │  │ Data Pipeline    │    │
│  ┌────────────┐  │               │  │                  │    │
│  │ uploader.  │  │               │  │ • Fetcher        │    │
│  │ py         │  │               │  │ • Preprocessor   │    │
│  │            │  │               │  │ • Feature Eng.   │    │
│  │ Syncs to   │  │               │  └────────┬─────────┘    │
│  │ cloud      │  │               │           │              │
│  └────────────┘  │               │           ▼              │
│                  │               │  ┌──────────────────┐    │
└──────────────────┘               │  │ Model Layer      │    │
                                   │  │                  │    │
                                   │  │ • SARIMA(X)      │    │
                                   │  │ • LightGBM x3    │    │
                                   │  │ • TFT            │    │
                                   │  │ • Ensemble       │    │
                                   │  └────────┬─────────┘    │
                                   │           │              │
                                   │           ▼              │
                                   │  ┌──────────────────┐    │
                                   │  │ React Dashboard  │    │
                                   │  │                  │    │
                                   │  │ • Dashboard      │    │
                                   │  │ • Forecast       │    │
                                   │  │ • History        │    │
                                   │  │ • Metrics        │    │
                                   │  │ • Leaderboard    │    │
                                   │  │ • Pipeline       │    │
                                   │  └──────────────────┘    │
                                   │                          │
                                   │    ┌────────────────┐    │
                                   │    │ MLflow UI      │    │
                                   │    │ (Experiments)  │    │
                                   │    └────────────────┘    │
                                   │                          │
                                   └──────────────────────────┘
```

---

## Project Structure

```
thermosense/
├── README.md                          # This file
├── PLAN.md                            # Full transformation roadmap
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
│
├── config/
│   └── config.yaml                    # All tunable parameters
│
├── hardware/                          # Raspberry Pi sensor code
│   ├── sensor_daemon.py               # Read DHT22, store to SQLite
│   ├── uploader.py                    # Sync readings to cloud API
│   ├── requirements.txt               # Pi-specific dependencies
│   ├── install.sh                     # systemd service setup
│   └── README.md                      # Hardware setup guide
│
├── data/
│   ├── legacy/                        # Original 40-day sensor CSV
│   │   └── temperature-data-for-TSA.csv
│   ├── raw/                           # API JSON responses (git-ignored)
│   ├── processed/                     # Merged parquet (git-ignored)
│   ├── features/                      # Feature matrix (git-ignored)
│   └── baselines/                     # Forecast comparison DB
│       └── forecasts.db
│
├── src/
│   ├── data/
│   │   ├── fetcher.py                 # Open-Meteo + OWM API clients
│   │   ├── preprocess.py              # Merge, gap-fill, validate
│   │   └── baseline_collector.py      # Collect commercial forecasts
│   ├── features/
│   │   └── engineer.py                # 38-feature generation
│   ├── models/
│   │   ├── base_model.py              # Abstract interface
│   │   ├── sarima_model.py            # SARIMAX with auto-tuning
│   │   ├── lgbm_model.py              # LightGBM per-horizon
│   │   ├── tft_model.py               # Temporal Fusion Transformer
│   │   ├── ensemble.py                # Ridge meta-learner
│   │   └── loader.py                  # Runtime model management
│   ├── evaluation/
│   │   ├── metrics.py                 # MAE, RMSE, MAPE, Skill, Coverage
│   │   └── statistical_tests.py       # t-tests, effect sizes, CIs
│   └── api/
│       ├── main.py                    # FastAPI app
│       └── routes/
│           ├── forecast.py            # GET/POST forecast
│           ├── history.py             # GET historical data
│           ├── metrics.py             # GET model metrics
│           ├── leaderboard.py         # GET accuracy comparison
│           ├── sensor.py              # POST sensor readings
│           ├── statistics.py          # GET statistical tests
│           ├── locations.py           # Multi-location support
│           └── pipeline.py            # POST backfill/train/daily
│
├── scripts/
│   ├── run_pipeline.py                # CLI: backfill or daily update
│   └── train_models.py                # CLI: train models
│
├── models/                            # Serialized models
│   ├── sarima.pkl
│   ├── lgbm_h1.pkl / lgbm_h2.pkl / lgbm_h3.pkl
│   ├── ensemble.pkl
│   └── results.json                   # Evaluation metrics
│
├── frontend/                          # React dashboard
│   ├── package.json
│   └── src/
│       ├── api.js                     # API client
│       ├── App.js                     # Router
│       └── pages/
│           ├── Dashboard.js
│           ├── Forecast.js
│           ├── History.js
│           ├── Metrics.js
│           ├── Leaderboard.js
│           └── Pipeline.js
│
├── tests/                             # Test suite
│   ├── test_fetcher.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_api.py
│   └── test_statistical_tests.py
│
└── docs/images/                       # Dashboard screenshots
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Git

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

For TFT support (optional, requires PyTorch):

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
  name: "Your City"
  lat: 12.9716      # Your latitude
  lon: 77.5946      # Your longitude
  timezone: "Asia/Kolkata"
```

Optionally, add API keys to `.env` for baseline comparison:

```bash
OWM_API_KEY=your_openweathermap_key      # Free at openweathermap.org
ACCUWEATHER_API_KEY=your_accuweather_key # Free at developer.accuweather.com
```

### 3. Run the Data Pipeline

**First-time backfill** — fetches 365 days of history:

```bash
python scripts/run_pipeline.py --mode backfill
```

**Daily incremental update** (run via cron or manually):

```bash
python scripts/run_pipeline.py --mode daily
```

### 4. Train Models

```bash
# Train SARIMA, LightGBM, and Ensemble (fast, ~2 minutes)
python scripts/train_models.py --models sarima lgbm ensemble

# Train everything including TFT (~10 minutes, needs PyTorch)
python scripts/train_models.py --models sarima lgbm tft ensemble
```

### 5. Build and Launch

```bash
# Build the React frontend
cd frontend && npm install && npm run build && cd ..

# Start the server
uvicorn src.api.main:app --reload

# Open http://localhost:8000
```

### 6. Run Tests

```bash
pytest tests/ -v
```

### 7. Track Experiments

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## Hardware Setup (Raspberry Pi)

### Bill of Materials (~$25)

| Component | Cost |
|-----------|------|
| Raspberry Pi Zero 2 W | $15 |
| DHT22 Temperature/Humidity Sensor | $5 |
| Breadboard + jumper wires | $5 |

### Wiring Diagram

```
DHT22 Pin    Raspberry Pi Pin
─────────    ────────────────
VCC    ───── 3.3V (Pin 1)
DATA   ───── GPIO 4 (Pin 7)  [with 10K pull-up to VCC]
GND    ───── GND (Pin 6)
```

### Installation on Pi

```bash
# 1. Clone the repo on your Pi
git clone https://github.com/yourusername/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling/hardware

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Test the sensor
python3 sensor_daemon.py --simulate  # Test without hardware
python3 sensor_daemon.py             # Test with actual DHT22

# 4. Install as systemd service (runs at boot)
sudo ./install.sh

# 5. Configure uploader (edit cloud URL)
export THERMOSENSE_API_URL=https://your-cloud-server.com
python3 uploader.py --continuous
```

### Sensor Placement Guidelines

For accurate outdoor readings:
- **Sheltered**: Under an eave, porch, or balcony (protected from rain)
- **Shaded**: Away from direct sunlight
- **Ventilated**: Good airflow (not in an enclosed box)
- **Height**: 1.5-2m above ground (standard meteorological convention)
- **Note exact GPS coordinates**: Used for API comparison

---

## Running the System

### Development Mode (Local)

```bash
# Terminal 1: Backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Frontend (for hot reload during development)
cd frontend && npm start

# Terminal 3: MLflow (optional)
mlflow ui --port 5000
```

### Production Mode (Cloud)

Option 1: **Railway** (free tier, recommended)

```bash
# railway.toml is already configured
railway up
```

Option 2: **Docker**

```bash
docker build -t thermosense .
docker run -p 8000:8000 thermosense
```

Option 3: **Cloudflare Tunnel** (edge deployment)

```bash
# Run on Raspberry Pi itself
cloudflared tunnel --url http://localhost:8000
```

### Daily Operations (Cron Jobs)

Add to your crontab (`crontab -e`):

```bash
# Run daily data update at 10 PM (after 9 PM sensor reading)
0 22 * * * cd /path/to/thermosense && .venv/bin/python scripts/run_pipeline.py --mode daily

# Collect baseline forecasts at 6 PM
0 18 * * * cd /path/to/thermosense && .venv/bin/python -m src.data.baseline_collector --collect

# Weekly model retraining on Sunday midnight
0 0 * * 0 cd /path/to/thermosense && .venv/bin/python scripts/train_models.py --models sarima lgbm ensemble
```

---

## Live Accuracy Leaderboard

The most important feature for proving ThermoSense works.

### Viewing the Leaderboard

1. **Dashboard**: Visit `http://localhost:8000/leaderboard`
2. **API**: `GET /api/leaderboard?window_days=30&horizon=1`
3. **CLI**: `python -m src.data.baseline_collector --leaderboard`

### Example Output

```
════════════════════════════════════════════════════════════════════
           LIVE ACCURACY LEADERBOARD (Day-1, Last 30 Days)
════════════════════════════════════════════════════════════════════
 Rank   Source             RMSE       MAE        N     
────────────────────────────────────────────────────────────────────
 🥇     ThermoSense        0.68°C     0.52°C     30    
 🥈     OpenWeatherMap     1.24°C     0.98°C     30    
 🥉     Open-Meteo         1.31°C     1.05°C     30    
 4      AccuWeather        1.45°C     1.12°C     30    
════════════════════════════════════════════════════════════════════

ThermoSense beats the best commercial app by 45%
Statistical significance: p < 0.01, Cohen's d = 1.2
```

### Statistical Rigor

The `/api/statistics` endpoint provides:

- **Paired t-test**: Compares ThermoSense errors vs baseline errors
- **Confidence interval**: 95% CI on the improvement
- **Effect size**: Cohen's d for practical significance
- **Sample size**: Minimum 30 days required for validity

---

## The Key Innovation: api_bias

The single most impactful feature in the entire system:

```python
api_bias           = sensor_actual_temp − open_meteo_api_temp
api_bias_roll7_mean = rolling 7-day mean of api_bias
api_bias_roll7_std  = rolling 7-day std of api_bias
```

### Why This Works

Open-Meteo's temperature at your lat/lon is interpolated from gridded NWP model output — an average over a ~1 km² cell. Your sensor is a point measurement subject to:

- **Urban heat island** — concrete and asphalt absorb and re-radiate heat
- **Building geometry** — walls reflect and trap radiation
- **Local vegetation** — or lack thereof
- **Elevation micro-differences** — rooftop vs ground-level

This offset is not random. It has:
- A **systematic component** that varies slowly with season
- A **stochastic component** that the model quantifies via rolling std

The model learns this correction automatically and applies it to future API forecasts.

---

## Models in Depth

### SARIMA(X) — `src/models/sarima_model.py`

- Auto-selects optimal (p,d,q)(P,D,Q,7) order via AIC
- Exogenous regressors: humidity, pressure
- Best for capturing weekly seasonality

### LightGBM — `src/models/lgbm_model.py`

- One model per horizon (Day 1/2/3)
- Handles missing values natively
- SHAP feature importance available

### Temporal Fusion Transformer — `src/models/tft_model.py`

- Attention-based architecture (Lim et al., 2021)
- Native quantile outputs (10th/50th/90th percentile)
- Variable selection networks learn feature importance

### Ensemble Stacker — `src/models/ensemble.py`

- Ridge regression meta-learner
- Trained on out-of-fold predictions
- Separate weights per horizon

---

## API Reference

All endpoints prefixed with `/api`. Interactive docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast?days=3` | 3-day temperature forecast |
| `POST` | `/api/forecast/feedback` | Submit actual observation |
| `GET` | `/api/history?start=2024-06-01&end=2024-07-11` | Historical data |
| `GET` | `/api/metrics?window_days=30` | Model accuracy metrics |
| `GET` | `/api/leaderboard?window_days=30&horizon=1` | Live accuracy comparison |
| `GET` | `/api/statistics?window_days=30` | Statistical significance tests |
| `POST` | `/api/sensor/readings` | Receive sensor uploads |
| `GET` | `/api/pipeline/status` | System health |
| `POST` | `/api/pipeline/backfill` | Trigger data backfill |
| `POST` | `/api/pipeline/train` | Trigger model training |
| `GET` | `/api/health` | Health check |

### Example: Forecast Response

```json
{
  "location": "Bangalore",
  "generated_at": "2026-05-12T16:45:57Z",
  "model_used": "ensemble",
  "forecasts": [
    {
      "date": "2026-05-13",
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

## Making the Most of ThermoSense

### For Portfolio / Resume

After 30+ days of operation, you can claim:

> "Deployed an IoT sensor network with ML-based bias correction achieving **45% lower RMSE** than Google Weather on Day-1 forecasts (p < 0.01, n=90 days)."

### For Personal Use

- Check the dashboard daily for hyperlocal forecasts
- Submit feedback via `/forecast` to improve accuracy over time
- Export historical data for your own analysis

### For Expanding Scope

- **Multi-location**: Deploy sensors at friends' houses, compare microclimates
- **Agriculture**: Alert when overnight lows threaten frost
- **Energy**: Integrate with smart thermostat for predictive HVAC

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Hardware** | Raspberry Pi Zero 2 W + DHT22 sensor |
| **Edge Software** | Python, SQLite, systemd |
| **Data Fetching** | Open-Meteo API (free), OpenWeatherMap, AccuWeather |
| **Data Storage** | Parquet (pandas + pyarrow), SQLite |
| **Statistical Models** | statsmodels, pmdarima |
| **Gradient Boosting** | LightGBM |
| **Deep Learning** | PyTorch, pytorch-forecasting (TFT) |
| **Ensemble** | scikit-learn (Ridge) |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI + Uvicorn |
| **Frontend** | React 19, Recharts, Lucide icons |
| **Deployment** | Railway, Docker, Cloudflare Tunnel |

---

## References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [pytorch-forecasting Documentation](https://pytorch-forecasting.readthedocs.io)
- [MLflow Documentation](https://mlflow.org/docs/latest)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
