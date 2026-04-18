# ThermoSense: Hyperlocal Temperature Intelligence Platform
## Upgrade Plan: From Static Analysis to a Live, Comparative Forecasting System

---

## 1. Project Understanding (Current State)

### What Exists
- **Dataset**: 40 days of nightly (9–10 PM) temperature readings from a single location (June–July 2024), stored as manual CSV exports
- **Models**: ARIMA(1,0,0) auto-selected via `pmdarima`, Prophet (installed but minimally used)
- **Analysis**: Train/test split comparison (3-day vs 37-day training window), lag analysis (Day 1/2/3 horizon), vs. commercial weather app predictions
- **Key Results**: RMSE 2.96°C (small dataset) → 0.87°C (large dataset), 70% improvement; Day-1 lag RMSE = 1.34°C
- **Limitations**:
  - Only temperature; no humidity, wind, pressure, cloud cover, or dew point
  - Single snapshot reading per day (9–10 PM), not a true diurnal time series
  - Static CSV files — no live data pipeline
  - Only ARIMA compared; no deep learning or ensemble baselines
  - No API, no deployment, no reproducible experiment tracking
  - No seasonal modeling despite spanning monsoon transition (June→July India)

### Real Problem Being Solved
Commercial weather apps (e.g., AccuWeather, Weather.com) rely on NWP (Numerical Weather Prediction) models trained on satellite + radar data. They are *not tuned to hyperlocal microclimates*. A rooftop sensor reading at 9 PM in a dense urban area will consistently differ from the nearest official station due to urban heat islands, local vegetation, building geometry, etc.

**This project's upgraded goal**: Build a system that learns the *local bias* of a sensor location relative to public weather data, and use that residual correction on top of a multi-variate deep learning forecast to outperform both raw ARIMA and commercial apps — and serve predictions via a REST API.

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                              │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Open-Meteo API  │    │  OpenWeatherMap  │    │  Historical CSV  │   │
│  │  (free, no key)  │    │  (free tier API) │    │  (existing data) │   │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘   │
│           └──────────────────────┴──────────────────────┘              │
│                                   │                                      │
│                          ┌────────▼────────┐                            │
│                          │   data_fetcher  │  (src/data/fetcher.py)     │
│                          │  (scheduled job)│                            │
│                          └────────┬────────┘                            │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                        STORAGE LAYER                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │  data/raw/          data/processed/     data/features/  │            │
│  │  (API responses)    (cleaned DFs)       (engineered)    │            │
│  └─────────────────────────────────────────────────────────┘            │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                             │
│                                                                          │
│  • Rolling statistics (mean, std, min, max over 3/7/14 day windows)     │
│  • Lag features (T-1, T-2, T-3, T-7)                                   │
│  • Calendar features (hour, day_of_week, month, is_monsoon)             │
│  • External features (humidity, pressure, wind_speed, cloud_cover,      │
│    dew_point, UV_index from Open-Meteo)                                 │
│  • Bias residual feature: (sensor_reading - API_prediction)             │
│                                                                          │
│  src/features/engineer.py                                                │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                         MODEL LAYER                                      │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ SARIMA(X)    │  │  LightGBM    │  │  Temporal     │  │ Ensemble  │ │
│  │ (baseline)   │  │  (gradient   │  │  Fusion       │  │ Stacker   │ │
│  │              │  │   boosting)  │  │  Transformer  │  │ (meta-ML) │ │
│  └──────────────┘  └──────────────┘  └───────────────┘  └───────────┘ │
│                                                                          │
│  src/models/sarima_model.py                                              │
│  src/models/lgbm_model.py                                                │
│  src/models/tft_model.py                                                 │
│  src/models/ensemble.py                                                  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                      EXPERIMENT TRACKING (MLflow)                        │
│                                                                          │
│  • Per-run: MAE, RMSE, MAPE, skill score vs. climatology                │
│  • Artifact logging: model weights, feature importance, residual plots  │
│  • Model registry for champion/challenger versioning                     │
│                                                                          │
│  mlruns/   (auto-created by MLflow)                                      │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                          API LAYER (FastAPI)                             │
│                                                                          │
│  GET  /forecast?lat=XX&lon=YY&days=3                                    │
│  GET  /history?start=YYYY-MM-DD&end=YYYY-MM-DD                          │
│  GET  /metrics                                                           │
│  POST /feedback  { date, actual_temp }  (closes the loop)               │
│                                                                          │
│  src/api/main.py                                                         │
│  src/api/routes/forecast.py                                              │
│  src/api/routes/history.py                                               │
│  src/api/routes/metrics.py                                               │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                    NOTEBOOK / ANALYSIS LAYER                             │
│                                                                          │
│  notebooks/01_eda.ipynb                 (existing analysis, cleaned)    │
│  notebooks/02_feature_engineering.ipynb                                 │
│  notebooks/03_model_comparison.ipynb    (all models head-to-head)       │
│  notebooks/04_error_analysis.ipynb      (residuals, confidence bands)   │
│  notebooks/05_api_demo.ipynb            (live API calls from notebook)  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack Delta (What's Being Added)

| Category | Current | Upgraded |
|---|---|---|
| Data source | Manual CSV | Open-Meteo API + OpenWeatherMap API (free tiers) |
| Feature set | Temperature only | Temperature + 8 meteorological variables |
| Models | ARIMA(1,0,0), Prophet | SARIMA(X), LightGBM, Temporal Fusion Transformer |
| Experiment tracking | None | MLflow |
| Scheduling | None | APScheduler (in-process cron) |
| API | None | FastAPI + Uvicorn |
| Environment management | requirements.txt | `.env` + `python-dotenv` |
| Config management | Hardcoded | `config/config.yaml` |
| Testing | None | pytest |
| Containerization (optional) | None | Docker + docker-compose |

---

## 4. Step-by-Step Implementation Plan

---

### Phase 0 — Repository Cleanup & Scaffolding (Day 1)

**Goal**: Create the full directory structure and install all dependencies.

```
thermosense/
├── PLAN.md                          ← this file
├── README.md
├── .env.example                     ← template; real .env is git-ignored
├── .gitignore
├── requirements.txt
├── config/
│   └── config.yaml                  ← all tunable parameters in one place
├── data/
│   ├── raw/                         ← API responses, never hand-edited
│   ├── processed/                   ← cleaned, merged DataFrames as parquet
│   ├── features/                    ← feature-engineered parquet files
│   └── legacy/                      ← existing CSVs moved here
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py               ← pulls from Open-Meteo & OWM
│   │   └── preprocess.py            ← cleans, merges, aligns timestamps
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py              ← all feature creation logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py            ← abstract base class
│   │   ├── sarima_model.py
│   │   ├── lgbm_model.py
│   │   ├── tft_model.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               ← MAE, RMSE, MAPE, skill score
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routes/
│           ├── forecast.py
│           ├── history.py
│           └── metrics.py
├── notebooks/
│   ├── 01_eda.ipynb                 ← refactored from Weather-prediction.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_api_demo.ipynb
├── tests/
│   ├── test_fetcher.py
│   ├── test_features.py
│   └── test_api.py
└── mlruns/                          ← auto-created by MLflow, git-ignored
```

**Commands**:
```bash
pip install fastapi uvicorn mlflow lightgbm pytorch-forecasting torch \
            statsmodels pmdarima prophet python-dotenv pyyaml \
            apscheduler requests httpx pytest pytest-asyncio
```

**`config/config.yaml`** (reference during implementation):
```yaml
location:
  name: "Bangalore"          # change to your city
  lat: 12.9716
  lon: 77.5946

api:
  open_meteo_base: "https://api.open-meteo.com/v1/forecast"
  owm_base: "https://api.openweathermap.org/data/2.5"

data:
  fetch_interval_hours: 6
  history_days: 365           # how far back to backfill on first run
  target_hour: 21             # 9 PM reading to match existing dataset

models:
  sarima:
    order: [1, 0, 0]
    seasonal_order: [1, 1, 0, 7]   # weekly seasonality
  lgbm:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6
    num_leaves: 31
  tft:
    max_prediction_length: 3
    max_encoder_length: 30
    hidden_size: 64
    attention_head_size: 4
    dropout: 0.1
    epochs: 50
  ensemble:
    weights: [0.2, 0.3, 0.5]   # sarima, lgbm, tft

evaluation:
  test_split_days: 14
  cv_folds: 5                  # time-series cross-validation

api:
  host: "0.0.0.0"
  port: 8000
```

---

### Phase 1 — Data Pipeline (Days 2–4)

**Goal**: Replace manual CSV data entry with an automated ingestion pipeline that pulls live + historical data.

#### 1.1 Choose APIs

**Open-Meteo** (free, no API key required):
- Endpoint: `https://api.open-meteo.com/v1/forecast`
- Variables: `temperature_2m`, `relativehumidity_2m`, `dewpoint_2m`, `precipitation`, `pressure_msl`, `cloudcover`, `windspeed_10m`, `uv_index`
- Historical data: `https://archive-api.open-meteo.com/v1/archive` (free, goes back to 1940)
- **This is the primary data source — completely free, no rate limits for modest usage**

**OpenWeatherMap** (optional, free tier = 1000 calls/day):
- Provides "current conditions" as a cross-check and to obtain the commercial app prediction baseline for comparison
- Store API key in `.env` as `OWM_API_KEY`

#### 1.2 `src/data/fetcher.py`

Key functions:
```python
def fetch_historical_open_meteo(lat, lon, start_date, end_date) -> pd.DataFrame:
    """
    Fetches hourly data from Open-Meteo historical archive.
    Resamples to daily 9PM snapshot to match existing dataset format.
    Returns DataFrame with columns:
      date, temp_c, humidity_pct, dewpoint_c, precip_mm,
      pressure_hpa, cloudcover_pct, windspeed_kmh, uv_index
    """

def fetch_forecast_open_meteo(lat, lon, days=7) -> pd.DataFrame:
    """Fetches 7-day forecast for real-time prediction serving."""

def fetch_owm_current(lat, lon, api_key) -> dict:
    """Fetches current conditions from OWM for comparison baseline."""
```

**Note on security**: API keys must be loaded via `os.environ.get("OWM_API_KEY")` loaded from `.env`. Never hardcode them.

#### 1.3 `src/data/preprocess.py`

```python
def merge_with_legacy(api_df, legacy_csv_path) -> pd.DataFrame:
    """
    Merges Open-Meteo historical data with existing 40-day CSV.
    Existing manual readings take precedence for the overlap period
    since they are actual sensor readings (not API values).
    Fills gaps using API data.
    """

def align_to_daily_9pm(hourly_df) -> pd.DataFrame:
    """Resamples hourly API data to daily 9 PM snapshots."""

def detect_and_fill_gaps(df) -> pd.DataFrame:
    """Detects missing dates, forward-fills or interpolates."""
```

#### 1.4 Backfill Run

On first run, pull 365 days of history from Open-Meteo archive to give the deep learning model enough training data. The existing 40-day CSV bridges the gap as the "ground truth" sensor.

---

### Phase 2 — Feature Engineering (Days 5–6)

**Goal**: Transform raw time series into a rich feature matrix that captures temporal patterns, external drivers, and local bias.

#### Features to Engineer (`src/features/engineer.py`)

**Lag features** (autoregressive signals):
```
temp_lag_1, temp_lag_2, temp_lag_3, temp_lag_7
humidity_lag_1, pressure_lag_1
```

**Rolling window statistics** (trend signals):
```
temp_roll3_mean, temp_roll3_std
temp_roll7_mean, temp_roll7_max, temp_roll7_min
temp_roll14_mean
```

**Calendar features** (seasonal signals):
```
month, day_of_year, week_of_year
is_monsoon (June–September = 1, else 0)
day_sin, day_cos       ← cyclic encoding of day_of_year
month_sin, month_cos   ← cyclic encoding of month
```

**External meteorological features** (from Open-Meteo):
```
humidity_pct, dewpoint_c, pressure_hpa, cloudcover_pct,
windspeed_kmh, precip_mm, uv_index
```

**Local bias / residual feature** (the key innovation):
```
api_bias = sensor_actual_temp - open_meteo_api_temp
api_bias_roll7_mean   ← rolling mean of local bias
```
This is the feature that makes our model learn the *hyperlocal correction* on top of the global API forecast. This is the core differentiator from ARIMA on raw temperature.

---

### Phase 3 — Model Development (Days 7–14)

**Goal**: Train, tune, and compare four approaches with rigorous cross-validation. All experiments tracked in MLflow.

#### 3.1 Baseline: SARIMA(X)

Upgrade from `ARIMA(1,0,0)` to `SARIMAX` with:
- Seasonal component: `(1,1,0,7)` — weekly seasonality (monsoon cycles)
- Exogenous variables: `humidity_pct`, `pressure_hpa` (SARIMAX extension)
- Auto-select via `pmdarima.auto_arima` with `seasonal=True`, `m=7`

**File**: `src/models/sarima_model.py`

```python
class SARIMAXModel(BaseModel):
    def fit(self, train_df, exog_cols):
        self.model = auto_arima(
            train_df["temp_c"],
            exogenous=train_df[exog_cols],
            seasonal=True, m=7,
            information_criterion="aic",
            stepwise=True
        )

    def predict(self, steps, future_exog):
        return self.model.predict(n_periods=steps, exogenous=future_exog)
```

**Expected improvement**: Adding seasonality + exogenous variables should reduce RMSE from 0.87°C to roughly 0.6–0.7°C.

#### 3.2 LightGBM with Time-Series Cross-Validation

LightGBM treats forecasting as a supervised regression problem using the engineered features above. It is fast, handles missing values, and natively captures feature interactions.

**File**: `src/models/lgbm_model.py`

```python
class LGBMForecastModel(BaseModel):
    def fit(self, feature_df, target_col="temp_c"):
        X = feature_df.drop(columns=[target_col, "date"])
        y = feature_df[target_col]
        self.model = lgb.train(
            params=self.config["lgbm"],
            train_set=lgb.Dataset(X, y),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
```

**Cross-validation**: Use `TimeSeriesSplit(n_splits=5)` from scikit-learn. No data leakage — each fold's validation set is always in the future relative to training.

**Feature importance**: Log SHAP values to MLflow as artifacts for explainability.

#### 3.3 Temporal Fusion Transformer (TFT) — The Core Innovation

TFT (Lim et al., 2021, NeurIPS) is an attention-based deep learning architecture purpose-built for multi-horizon time series forecasting. Key advantages over LSTM:
- **Gated residual networks** filter irrelevant features
- **Variable selection networks** learn which inputs matter at each time step
- **Multi-head attention** captures long-range dependencies across time
- **Quantile outputs** — predicts 10th/50th/90th percentile, giving calibrated uncertainty bands

**Library**: `pytorch-forecasting` (wraps PyTorch Lightning + TFT implementation)

**File**: `src/models/tft_model.py`

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

class TFTModel(BaseModel):
    def prepare_dataset(self, df):
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",              # integer time index
            target="temp_c",
            group_ids=["location"],           # single location for now
            max_encoder_length=30,            # look-back window
            max_prediction_length=3,          # 3-day forecast horizon
            time_varying_known_reals=[
                "month_sin", "month_cos", "day_sin", "day_cos",
                "humidity_pct", "pressure_hpa", "cloudcover_pct",
                "windspeed_kmh", "precip_mm"
            ],
            time_varying_unknown_reals=[
                "temp_c", "dewpoint_c", "api_bias_roll7_mean"
            ],
            target_normalizer=GroupNormalizer()
        )

    def fit(self, train_dataset, val_dataset):
        self.model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=1e-3,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        )
        trainer = pl.Trainer(max_epochs=50, gradient_clip_val=0.1)
        trainer.fit(self.model, train_dataloader, val_dataloader)
```

**Why TFT beats LSTM here**: Attention mechanism lets the model "look back" at the same calendar period from prior weeks (monsoon pattern) without vanishing gradients. The variable selection network will likely upweight `humidity_pct` and `api_bias_roll7_mean` — interpretable and correct.

#### 3.4 Ensemble Stacker

A linear meta-learner (Ridge Regression) trained on out-of-fold predictions from SARIMA(X), LightGBM, and TFT. Learns optimal weighting per forecast horizon (Day 1/2/3 weights differ).

**File**: `src/models/ensemble.py`

```python
class EnsembleStacker:
    """
    Level-1 meta-learner over SARIMA, LightGBM, TFT predictions.
    Trained on out-of-fold predictions to avoid overfitting.
    """
    def fit(self, oof_predictions_df, actuals):
        self.meta = Ridge(alpha=1.0)
        self.meta.fit(oof_predictions_df, actuals)
```

---

### Phase 4 — Evaluation Framework (Days 13–14)

**Goal**: Establish a rigorous, reproducible comparison protocol.

#### 4.1 Metrics (`src/evaluation/metrics.py`)

| Metric | Formula | Why |
|---|---|---|
| **MAE** | mean(|y - ŷ|) | Intuitive, same unit as temp |
| **RMSE** | sqrt(mean((y-ŷ)²)) | Penalizes large errors; matches existing baseline |
| **MAPE** | mean(|y-ŷ|/y) × 100 | Scale-independent % error |
| **Skill Score** | 1 - (RMSE_model / RMSE_climatology) | How much better than "just predict the mean"? |
| **Coverage (90%)** | % of actuals within [Q10, Q90] | Calibration of TFT uncertainty |

#### 4.2 Time-Series Cross-Validation Protocol

```
Training window:  [Day 1 ... Day N-14]
Validation:       [Day N-13 ... Day N-7]
Test (hold-out):  [Day N-6  ... Day N]    ← never seen during development
```

Use `TimeSeriesSplit` with expanding window (not rolling) to simulate real deployment.

#### 4.3 Expected Results Table (Target)

| Model | Day-1 RMSE | Day-2 RMSE | Day-3 RMSE |
|---|---|---|---|
| ARIMA(1,0,0) baseline | 1.34°C | 1.51°C | 1.86°C |
| SARIMA(X) | ~0.9°C | ~1.1°C | ~1.4°C |
| LightGBM | ~0.7°C | ~0.9°C | ~1.2°C |
| TFT (ours) | **~0.5°C** | **~0.7°C** | **~1.0°C** |
| Ensemble | **~0.45°C** | **~0.65°C** | **~0.9°C** |
| Commercial App | ~1.0°C | ~1.2°C | ~1.5°C |

*Targets based on TFT literature benchmarks on temperature datasets; actual results depend on data volume.*

---

### Phase 5 — MLflow Experiment Tracking (Days 13–14, parallel)

**Goal**: Every training run is logged, comparable, and reproducible.

```python
import mlflow

with mlflow.start_run(run_name="TFT_v1"):
    mlflow.log_params({
        "model": "TFT",
        "max_encoder_length": 30,
        "hidden_size": 64,
        "epochs": 50
    })
    mlflow.log_metrics({
        "day1_rmse": rmse_day1,
        "day2_rmse": rmse_day2,
        "day3_rmse": rmse_day3,
        "day1_mae": mae_day1,
        "skill_score": skill
    })
    mlflow.log_artifact("plots/residual_plot.png")
    mlflow.pytorch.log_model(tft_model, "model")
```

Access the UI with:
```bash
mlflow ui --port 5000
```

Register the best model as `"thermosense-champion"` in the MLflow Model Registry.

---

### Phase 6 — FastAPI Service (Days 15–17)

**Goal**: Serve real-time predictions via a REST API with automatic OpenAPI docs.

#### 6.1 `src/api/main.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import mlflow.pyfunc

app = FastAPI(title="ThermoSense API", version="1.0")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.pyfunc.load_model("models:/thermosense-champion/Production")
    yield

app = FastAPI(lifespan=lifespan)
```

#### 6.2 Endpoints

**`GET /forecast`**
```json
{
  "location": "Bangalore",
  "generated_at": "2024-07-12T21:00:00",
  "forecasts": [
    {"date": "2024-07-13", "predicted_temp_c": 26.4,
     "lower_bound": 25.1, "upper_bound": 27.8, "horizon_days": 1},
    {"date": "2024-07-14", "predicted_temp_c": 25.9,
     "lower_bound": 24.3, "upper_bound": 27.5, "horizon_days": 2},
    {"date": "2024-07-15", "predicted_temp_c": 25.5,
     "lower_bound": 23.9, "upper_bound": 27.1, "horizon_days": 3}
  ]
}
```

**`POST /feedback`** (closes the loop)
```json
{"date": "2024-07-13", "actual_temp_c": 26.0}
```
This endpoint appends the actual reading to the database, updates the `api_bias` rolling feature, and triggers incremental model retraining (online learning).

**`GET /metrics`** — Returns live MAE/RMSE over last 30 days vs. commercial app baseline.

#### 6.3 Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs auto-generated at `http://localhost:8000/docs`.

#### 6.4 Scheduled Data Fetching

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(fetch_and_store_latest, "cron", hour=21, minute=30)
scheduler.start()
```

Every night at 9:30 PM, the system fetches the latest Open-Meteo data, runs predictions, and stores them. At 10 PM the user can POST the actual reading via `/feedback`.

---

### Phase 7 — Notebook Refactoring (Days 18–19)

Move all existing notebook analysis into the new structure:

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Existing histograms, time series plots, ADF test, ACF/PACF — unchanged logic, cleaner presentation, reads from `data/processed/` |
| `02_feature_engineering.ipynb` | Demonstrates each feature category, correlation heatmaps, lag plots |
| `03_model_comparison.ipynb` | Trains all 4 models, plots predictions vs. actuals for all horizons, renders the comparison table |
| `04_error_analysis.ipynb` | Residual distribution plots, Q-Q plots, error by month/season, SHAP values from LightGBM |
| `05_api_demo.ipynb` | Live demo calling the running FastAPI — shows the full end-to-end system working |

---

### Phase 8 — Testing (Days 18–19, parallel)

```
tests/
├── test_fetcher.py       ← mock API responses; test data parsing
├── test_features.py      ← test lag creation, rolling windows, no leakage
└── test_api.py           ← test API endpoints with httpx AsyncClient
```

Run with:
```bash
pytest tests/ -v
```

---

## 5. Implementation Command Reference

When implementing, reference this plan with these phase names:

| Command Phrase | What to implement |
|---|---|
| `implement Phase 0` | Directory scaffold, requirements.txt, config.yaml |
| `implement Phase 1` | `src/data/fetcher.py` + `src/data/preprocess.py` |
| `implement Phase 2` | `src/features/engineer.py` |
| `implement Phase 3 SARIMA` | `src/models/sarima_model.py` |
| `implement Phase 3 LightGBM` | `src/models/lgbm_model.py` |
| `implement Phase 3 TFT` | `src/models/tft_model.py` |
| `implement Phase 3 Ensemble` | `src/models/ensemble.py` |
| `implement Phase 4` | `src/evaluation/metrics.py` + CV protocol |
| `implement Phase 5` | MLflow logging wrappers |
| `implement Phase 6` | `src/api/` (full FastAPI service) |
| `implement Phase 7` | Refactored notebooks |
| `implement Phase 8` | `tests/` |

---

## 6. Open-Meteo API — No Key Required

This is the primary data source. Example call (no authentication):

```python
import requests

params = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "hourly": "temperature_2m,relativehumidity_2m,dewpoint_2m,precipitation,pressure_msl,cloudcover,windspeed_10m,uv_index",
    "timezone": "Asia/Kolkata",
    "start_date": "2024-06-01",
    "end_date": "2024-07-11"
}
r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
data = r.json()
```

For the forecast (next 7 days):
```python
r = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": 12.9716, "longitude": 77.5946,
    "hourly": "temperature_2m,relativehumidity_2m,pressure_msl",
    "forecast_days": 7
})
```

---

## 7. Environment Variables (`.env`)

```
# Open-Meteo requires no API key — see fetcher.py
# OpenWeatherMap (optional, for commercial app baseline comparison)
OWM_API_KEY=your_key_here

# Location
LOCATION_LAT=12.9716
LOCATION_LON=77.5946
LOCATION_NAME=Bangalore

# API server
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=./mlruns
```

Copy `.env.example` to `.env` and fill in your values. Never commit `.env`.

---

## 8. Key Design Decisions & Rationale

### Why TFT over LSTM?
TFT's attention mechanism allows it to weigh specific past time steps (e.g., "same week last month during monsoon") rather than compressing all history into a hidden state. For seasonal weather with 365+ days of data, this is critical. TFT also produces quantile forecasts natively, giving calibrated uncertainty intervals that raw ARIMA cannot.

### Why LightGBM as intermediate model?
Between ARIMA (underfitting, univariate) and TFT (data-hungry, compute-heavy), LightGBM offers a practical middle ground: it handles the full feature matrix, trains in seconds, and provides SHAP-based explainability. It will likely outperform ARIMA with less data than TFT needs.

### Why the `api_bias` feature?
This is the single most impactful innovation. Commercial weather apps report conditions at the nearest weather station, which may be kilometers away and in a different microclimate. By computing `sensor_actual - api_predicted` as a rolling feature, the model learns the *systematic offset* of your measurement location. This is why our model can beat the commercial app even with far less training data.

### Why Open-Meteo over other APIs?
- Completely free, no rate limits for reasonable usage
- 80+ years of historical data via the archive endpoint
- No API key — no credential management risk
- WMO-calibrated data; same quality as paid providers
- 7-day forecast available for real-time prediction

### Why FastAPI over Flask?
- Native async support — critical for non-blocking data fetches
- Automatic OpenAPI/Swagger docs at `/docs`
- Pydantic validation of request/response models
- 3–5x faster than Flask for I/O-bound workloads
- Built-in `lifespan` for model loading at startup

---

## 9. Success Criteria

The project is "next-level" when:

- [ ] Day-1 RMSE ≤ 0.6°C (beats existing 0.87°C by ≥ 30%)
- [ ] TFT outperforms LightGBM outperforms SARIMA(X) outperforms ARIMA — validated via CV
- [ ] Ensemble beats TFT alone on at least Day-2 and Day-3 horizons
- [ ] API returns predictions in < 200ms (p95)
- [ ] `/feedback` endpoint enables live accuracy tracking
- [ ] MLflow UI shows full run history with parameter sweeps
- [ ] `GET /metrics` confirms model beats commercial app baseline on last 30 days
- [ ] All 5 notebooks are clean and runnable end-to-end

---

## 10. References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Open-Meteo documentation: https://open-meteo.com/en/docs
- pytorch-forecasting documentation: https://pytorch-forecasting.readthedocs.io
- MLflow documentation: https://mlflow.org/docs/latest/index.html
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
