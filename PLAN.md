# ThermoSense: Hyperlocal Temperature Intelligence Platform
## Transformation Plan: From Portfolio Demo to Measurable Impact

---

## Executive Summary

**Current State**: A well-engineered ML pipeline with no real-world impact. Models trained on synthetic backfill data, no actual sensor, no deployed service, no users, and metrics measured against self-generated baselines.

**Target State**: A deployed, continuously-running system that:
1. Collects real sensor data (ESP32 + DHT22 or Raspberry Pi)
2. Makes daily predictions consumed by actual users
3. Publishes live accuracy metrics comparing ThermoSense vs Google Weather / AccuWeather
4. Demonstrates measurable improvement over commercial apps on a public leaderboard
5. Has a clear deployment story (Raspberry Pi edge device + cloud API)

**Why This Matters for Resume**: The difference between "I built an ML pipeline" and "I deployed a system that beats Google Weather by 40% at predicting temperature in my specific location, measured over 90 days of live data" is the difference between a forgettable project and a memorable one.

---

## Part 1: Honest Assessment of Current State

### What Exists (Good Foundation)

| Component | Status | Quality |
|-----------|--------|---------|
| Data pipeline (fetcher, preprocess) | Complete | Production-grade |
| Feature engineering | Complete | 38 features, well-designed |
| SARIMA(X) model | Complete | Working |
| LightGBM models (H1/H2/H3) | Complete | Working |
| TFT model | Complete | Needs PyTorch |
| Ensemble stacker | Complete | Working |
| FastAPI backend | Complete | Clean architecture |
| React dashboard | Complete | Polished UI |
| Tests | Complete | Good coverage |
| MLflow integration | Complete | Working |

### What's Missing (Critical Gaps)

| Gap | Impact | Severity |
|-----|--------|----------|
| **No real sensor** | The "hyperlocal" claim is unverifiable | Critical |
| **No live deployment** | No predictions being consumed | Critical |
| **No real baseline comparison** | "84% improvement" is against your own ARIMA, not Google Weather | Critical |
| **No user** | No one benefits from this system | Critical |
| **Tiny ground truth** | 40 days of manual readings is statistically weak | High |
| **No live accuracy tracking** | Cannot prove it works over time | High |
| **No public leaderboard** | Results aren't verifiable by others | Medium |

### Current Metrics Are Misleading

The results.json shows:
- Ensemble Day-1 RMSE: 0.221°C
- SARIMA Day-1 RMSE: 1.036°C

**Problem**: These metrics are computed on Open-Meteo API data predicting... Open-Meteo API data (with 40 days of sensor overlap). The "api_bias" feature leaks because you're training on data where you know both the sensor reading and the API value.

**Real question**: Can ThermoSense predict tomorrow's *actual sensor reading* better than AccuWeather's Day-1 forecast?

That question has never been answered.

---

## Part 2: Transformation Roadmap

### Phase A: Deploy a Real Sensor (Week 1-2)

**Goal**: Collect real, continuous temperature data from a physical device.

#### Option 1: Raspberry Pi Zero 2 W + DHT22 (Recommended)

**Hardware cost**: ~$25 total
- Raspberry Pi Zero 2 W: $15
- DHT22 sensor: $5
- Breadboard + wires: $5

**Why Raspberry Pi over ESP32**:
- Can run Python directly (no firmware flashing)
- WiFi built-in
- Can host a local SQLite database and sync to cloud
- Can run the ThermoSense API locally for edge inference

**Implementation**:

```
/hardware/
├── sensor_daemon.py      # Reads DHT22 every 5 minutes, writes to SQLite
├── uploader.py           # Syncs SQLite to cloud API every hour
├── requirements.txt      # Adafruit_DHT, requests
└── install.sh            # systemd service setup
```

**sensor_daemon.py** pseudocode:
```python
import Adafruit_DHT
import sqlite3
from datetime import datetime

DHT_SENSOR = Adafruit_DHT.DHT22
DHT_PIN = 4  # GPIO pin

def read_sensor():
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temp_c": round(temperature, 2),
        "humidity_pct": round(humidity, 2),
        "source": "dht22_sensor"
    }

def store_reading(reading, db_path="readings.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO readings (timestamp, temp_c, humidity_pct, source)
        VALUES (?, ?, ?, ?)
    """, (reading["timestamp"], reading["temp_c"], reading["humidity_pct"], reading["source"]))
    conn.commit()
    conn.close()
```

**Deployment location**: Place the sensor:
- Outdoors but sheltered (balcony, porch, under an eave)
- Away from direct sunlight and heat sources
- At a height of 1.5-2m (standard meteorological convention)
- Note exact GPS coordinates for Open-Meteo comparison

#### Option 2: Use a Weather Station with API (No Hardware)

If you cannot deploy hardware, use a personal weather station (PWS) from Weather Underground's network:

1. Go to https://www.wunderground.com/wundermap
2. Find a PWS within 1km of your location
3. Use the Weather Underground API to fetch that station's readings
4. This gives you "hyperlocal" data without deploying hardware

**Trade-off**: Less impressive for resume ("I used someone else's sensor") but still demonstrates the bias-correction concept.

---

### Phase B: Live Baseline Comparison (Week 2-3)

**Goal**: Collect daily predictions from commercial weather apps and compare against ThermoSense.

#### Daily Automated Baseline Collection

Every day at 6 PM, collect Day-1 forecasts from:

1. **Google Weather** (scrape or use unofficial API)
2. **AccuWeather** (free tier API: 50 calls/day)
3. **Open-Meteo** (already integrated)
4. **OpenWeatherMap** (already integrated)
5. **ThermoSense** (your model's prediction)

Store in a table:

```sql
CREATE TABLE daily_forecasts (
    forecast_date DATE NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'google', 'accuweather', 'openmeteo', 'owm', 'thermosense'
    predicted_temp_c REAL NOT NULL,
    horizon_days INT NOT NULL,
    PRIMARY KEY (forecast_date, source, horizon_days)
);

CREATE TABLE daily_actuals (
    date DATE PRIMARY KEY,
    sensor_temp_c REAL NOT NULL,  -- 9 PM reading from your DHT22
    recorded_at TIMESTAMP NOT NULL
);
```

**Comparison query**:
```sql
SELECT 
    f.source,
    AVG(ABS(f.predicted_temp_c - a.sensor_temp_c)) AS mae,
    SQRT(AVG(POWER(f.predicted_temp_c - a.sensor_temp_c, 2))) AS rmse,
    COUNT(*) AS n_days
FROM daily_forecasts f
JOIN daily_actuals a ON f.forecast_date = a.date
WHERE f.horizon_days = 1
GROUP BY f.source
ORDER BY rmse;
```

**This is the key deliverable**: A table showing ThermoSense vs Google vs AccuWeather over 30/60/90 days of real predictions.

---

### Phase C: Public Deployment (Week 3-4)

**Goal**: Make ThermoSense accessible online with a live accuracy dashboard.

#### Option 1: Railway / Render (Free Tier)

Deploy the FastAPI backend to Railway (free tier: 500 hours/month, which is enough for always-on):

```bash
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"
```

**Public URL**: `https://thermosense.up.railway.app`

#### Option 2: Raspberry Pi as Edge + Cloudflare Tunnel

Run ThermoSense directly on the Pi, expose via Cloudflare Tunnel (free):

```bash
cloudflared tunnel --url http://localhost:8000
```

**Benefit**: True edge ML deployment story. The same device that collects data also runs inference.

---

### Phase D: Live Accuracy Leaderboard (Week 4-5)

**Goal**: A public page showing real-time accuracy comparison.

#### Dashboard Enhancements

Add a new page: **Leaderboard** (`/leaderboard`)

```
╔══════════════════════════════════════════════════════════════════╗
║           LIVE ACCURACY LEADERBOARD (Last 30 Days)              ║
╠══════════════════════════════════════════════════════════════════╣
║  Rank  │  Source         │  Day-1 RMSE  │  Day-1 MAE  │  n_days ║
╠════════╪═════════════════╪══════════════╪═════════════╪═════════╣
║   🥇   │  ThermoSense    │   0.68°C     │   0.52°C    │   30    ║
║   🥈   │  OpenWeatherMap │   1.24°C     │   0.98°C    │   30    ║
║   🥉   │  Open-Meteo     │   1.31°C     │   1.05°C    │   30    ║
║   4    │  AccuWeather    │   1.45°C     │   1.12°C    │   30    ║
║   5    │  Google Weather │   1.52°C     │   1.19°C    │   30    ║
╚════════╧═════════════════╧══════════════════════════════════════╝

ThermoSense beats Google Weather by 55% on Day-1 RMSE
Location: Bangalore (12.9716°N, 77.5946°E)
Sensor: DHT22 on Raspberry Pi Zero 2 W
Updated: 2026-05-07 21:00 IST
```

**Why this is powerful for resume**:
- Concrete, verifiable claim: "55% better than Google Weather"
- Live data, not cherry-picked
- Anyone can visit the URL and see current accuracy
- Shows end-to-end deployment, not just model training

---

### Phase E: Statistical Rigor (Week 5-6)

**Goal**: Make claims defensible with proper statistical testing.

#### Minimum Viable Evidence

To claim "ThermoSense beats commercial apps", you need:

1. **At least 30 days** of parallel forecasts (statistical minimum for t-test)
2. **Paired t-test** or Wilcoxon signed-rank test comparing daily errors
3. **Confidence interval** on the improvement (e.g., "ThermoSense RMSE is 0.4-0.8°C lower than Google, p < 0.01")
4. **Effect size** (Cohen's d) to show the improvement is practically meaningful

**Implementation** (`src/evaluation/statistical_tests.py`):

```python
from scipy import stats
import numpy as np

def compare_forecasters(errors_thermosense: np.ndarray, errors_baseline: np.ndarray):
    """
    Paired comparison of forecast errors.
    
    Returns:
        dict with t-statistic, p-value, effect size, and confidence interval
    """
    diff = errors_baseline - errors_thermosense  # positive = we're better
    
    # Paired t-test (or Wilcoxon if non-normal)
    t_stat, p_value = stats.ttest_rel(errors_baseline, errors_thermosense)
    
    # Effect size (Cohen's d for paired samples)
    d = diff.mean() / diff.std()
    
    # 95% CI on the mean improvement
    ci_low, ci_high = stats.t.interval(
        0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff)
    )
    
    return {
        "mean_improvement_c": round(diff.mean(), 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 6),
        "effect_size_d": round(d, 3),
        "ci_95_low": round(ci_low, 3),
        "ci_95_high": round(ci_high, 3),
        "n_samples": len(diff),
        "significant": p_value < 0.05
    }
```

**Resume-ready statement**: "ThermoSense achieves 0.68°C RMSE on Day-1 forecasts, a 45% improvement over Google Weather (p < 0.01, n=90, Cohen's d=1.2)."

---

### Phase F: Expand Scope for Impact (Week 6-8)

**Goal**: Increase the project's reach and applicability.

#### Option 1: Multi-Location Network

Deploy sensors at 3-5 locations (friends/family homes):
- Different microclimates (urban, suburban, near water)
- Train location-specific bias corrections
- Show the system generalizes across sites

**Resume statement**: "Deployed ThermoSense to 5 locations across Bangalore; average 42% RMSE improvement over commercial apps."

#### Option 2: Agriculture/Cold Chain Application

Partner with a local farmer or cold storage facility:
- Temperature monitoring for crop frost protection
- Alert system when predicted overnight low drops below threshold
- Quantify economic value: "Prevented X crop damage events worth ₹Y"

**Resume statement**: "Deployed ThermoSense for agricultural frost prediction; system issued accurate alerts for 8/9 frost events, saving estimated ₹50,000 in crop losses."

#### Option 3: Energy Optimization

If you have smart home devices:
- Use forecasts to pre-cool/pre-heat home before price spikes
- Integrate with smart thermostat API
- Quantify energy savings

**Resume statement**: "Integrated ThermoSense with smart thermostat; 12% reduction in cooling costs via predictive HVAC scheduling."

---

## Part 3: Updated Success Criteria

### Minimum Bar (Must Have)

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Real sensor deployed | DHT22/DS18B20 running ≥30 days | Photo + data logs |
| Live predictions | Daily 3-day forecasts generated automatically | API endpoint returning fresh data |
| Baseline comparison | ≥3 commercial services tracked daily | Database with parallel forecasts |
| RMSE improvement | ThermoSense beats best commercial app by ≥20% | Leaderboard showing live metrics |
| Statistical significance | p < 0.05 on improvement | t-test results displayed |
| Public deployment | Accessible via URL | Working link |

### Excellence Bar (Should Have)

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| 90-day track record | Continuous operation without gaps | Uptime logs |
| Multi-location | ≥3 sensors | Dashboard showing all locations |
| Real users | ≥10 people checking forecasts regularly | Analytics / feedback |
| Documentation | Blog post / video explaining the project | Published content |
| Open source adoption | ≥5 GitHub stars, ≥1 fork | GitHub metrics |

### Exceptional Bar (Nice to Have)

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Practical application | Agriculture/energy/health use case | User testimonial |
| Media coverage | HackerNews / Reddit / tech blog mention | Links |
| Conference talk | Presented at local meetup or PyCon | Recording |
| Academic citation | Used in a research paper | Citation link |

---

## Part 4: Revised Project Structure

```
thermosense/
├── README.md                     # Updated with live leaderboard badge
├── PLAN.md                       # This file
│
├── hardware/                     # NEW: Raspberry Pi sensor code
│   ├── sensor_daemon.py          # Read DHT22, store to SQLite
│   ├── uploader.py               # Sync to cloud API
│   ├── alerter.py                # Push notifications on anomalies
│   ├── requirements.txt
│   └── install.sh                # systemd service setup
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/                      # API JSON responses
│   ├── processed/                # Merged parquet
│   ├── features/                 # Feature matrix
│   └── baselines/                # NEW: Daily forecasts from Google/AccuWeather/etc.
│
├── src/
│   ├── data/
│   │   ├── fetcher.py
│   │   ├── preprocess.py
│   │   └── baseline_collector.py # NEW: Scrape/API commercial forecasts
│   │
│   ├── features/
│   │   └── engineer.py
│   │
│   ├── models/
│   │   ├── sarima_model.py
│   │   ├── lgbm_model.py
│   │   ├── tft_model.py
│   │   └── ensemble.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── statistical_tests.py  # NEW: t-tests, effect sizes, CIs
│   │
│   └── api/
│       ├── main.py
│       └── routes/
│           ├── forecast.py
│           ├── history.py
│           ├── metrics.py
│           ├── pipeline.py
│           └── leaderboard.py    # NEW: Live accuracy comparison
│
├── frontend/
│   └── src/
│       └── pages/
│           ├── Dashboard.js
│           ├── Forecast.js
│           ├── History.js
│           ├── Metrics.js
│           ├── Pipeline.js
│           └── Leaderboard.js    # NEW: Public accuracy dashboard
│
├── scripts/
│   ├── run_pipeline.py
│   ├── train_models.py
│   ├── collect_baselines.py      # NEW: Daily cron job
│   └── generate_report.py        # NEW: Weekly accuracy report
│
├── deployment/
│   ├── railway.toml              # NEW: Cloud deployment config
│   ├── Dockerfile
│   └── cloudflare_tunnel.sh      # NEW: Edge deployment
│
└── docs/
    ├── images/
    └── BLOG_POST.md              # NEW: Write-up for publication
```

---

## Part 5: Resume Bullet Points (Target)

After completing this transformation, you can write:

### Weak (Current State)
> "Built a temperature forecasting ML pipeline using SARIMA, LightGBM, and Temporal Fusion Transformer with a React dashboard."

### Strong (After Transformation)

> **ThermoSense - Hyperlocal Weather Intelligence System**
> - Deployed an IoT sensor network (Raspberry Pi + DHT22) to collect real-time temperature data from 3 locations
> - Developed an ensemble ML model (SARIMA + LightGBM + TFT) achieving **45% lower RMSE** than Google Weather on Day-1 forecasts (p < 0.01, n=90 days)
> - Built a public live leaderboard comparing ThermoSense vs 4 commercial weather services, updated daily
> - Served 500+ API requests/day for local users via FastAPI + Railway deployment
> - Open-sourced with 15 GitHub stars; featured on Hacker News

### For Specific Roles

**Data Scientist / ML Engineer**:
> "Designed a hyperlocal temperature forecasting system using Temporal Fusion Transformers; achieved 0.68°C RMSE by engineering a novel 'API bias' feature that captures systematic microclimate offsets. Validated improvement over commercial baselines with paired statistical testing."

**Full-Stack / MLOps Engineer**:
> "Built end-to-end ML system: DHT22 sensor → Raspberry Pi edge collection → cloud API (FastAPI) → React dashboard. Deployed via Railway with 99.5% uptime over 90 days. Automated daily data ingestion, model retraining, and baseline comparison."

**Embedded / IoT Engineer**:
> "Designed and deployed a solar-powered Raspberry Pi weather station with sub-1°C sensor accuracy. Implemented local SQLite storage with cloud sync, handling network outages gracefully. Edge inference reduces API calls by 80%."

---

## Part 6: Implementation Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Hardware setup | DHT22 + Pi deployed, sensor_daemon running, first 7 days of data |
| 2 | Baseline collection | AccuWeather/Google/OWM daily forecasts being stored |
| 3 | Cloud deployment | Railway or Cloudflare Tunnel live |
| 4 | Leaderboard page | Public comparison dashboard |
| 5 | 30-day milestone | First statistically valid comparison (p-value computed) |
| 6 | Polish + documentation | Blog post draft, README updated with live badge |
| 7 | Multi-location (optional) | Second sensor deployed |
| 8 | 60-day milestone | Robust statistical results, ready for publication |

---

## Part 7: Quick Wins (If Short on Time)

If you cannot do the full transformation, here are partial improvements ranked by impact:

### Tier 1: Minimal Effort, Maximum Impact

1. **Use a public PWS**: Find a Weather Underground station near you and use its data instead of deploying your own sensor. Still demonstrates the bias-correction concept.

2. **Collect 30 days of parallel baselines**: Write a simple cron job that fetches AccuWeather and Google Weather forecasts daily. After 30 days, you have a real comparison.

3. **Deploy to Railway (1 hour)**: Your current API works. Just deploy it and get a public URL.

### Tier 2: Moderate Effort

4. **Add statistical testing**: Implement the `compare_forecasters()` function and display p-values on the Metrics page.

5. **Create a leaderboard page**: Even with synthetic baselines, showing "ThermoSense vs Open-Meteo vs OWM" as a live dashboard is more compelling than static metrics.

### Tier 3: Full Effort

6. **Deploy real hardware**: The Raspberry Pi + DHT22 setup is genuinely impressive and differentiates you from everyone who just calls APIs.

---

## Part 8: Key Metrics to Track

### Daily Metrics (Automated)

| Metric | Stored In | Updated |
|--------|-----------|---------|
| Sensor 9 PM reading | `daily_actuals` table | Nightly |
| ThermoSense Day-1 forecast | `daily_forecasts` table | Nightly |
| Google Day-1 forecast | `daily_forecasts` table | Nightly |
| AccuWeather Day-1 forecast | `daily_forecasts` table | Nightly |
| Open-Meteo Day-1 forecast | `daily_forecasts` table | Nightly |

### Weekly Metrics (Dashboard)

| Metric | Calculation | Display |
|--------|-------------|---------|
| ThermoSense RMSE (7d) | sqrt(mean((pred - actual)²)) | Leaderboard |
| Best commercial RMSE (7d) | min(Google, AccuWeather, ...) | Leaderboard |
| Improvement % | (1 - ts_rmse / best_commercial_rmse) * 100 | Headline stat |
| Uptime | hours_online / (7*24) | Status badge |

### Monthly Metrics (Report)

| Metric | Why It Matters |
|--------|----------------|
| Rolling 30-day RMSE per source | Trend over time |
| p-value (ThermoSense vs best) | Statistical significance |
| Effect size (Cohen's d) | Practical significance |
| Sensor data completeness % | Data quality indicator |
| Worst prediction day | Identifies failure modes |

---

## Part 9: What NOT to Do

### Anti-Patterns to Avoid

1. **Don't keep adding features to the dashboard** - shipping beats polishing.

2. **Don't train more models** - you have enough models. The bottleneck is real data, not model architectur.

3. **Don't optimize for RMSE on synthetic data** - the current 0.221°C is meaningless without real sensor validation.

4. **Don't claim "beats commercial apps" without evidence** - your README says "84% improvement" but this is against your own ARIMA baseline, not Google.

5. **Don't over-engineer the hardware** - a simple DHT22 on a breadboard is enough. You're not building a production weather station.

---

## Part 10: Final Checklist Before Putting on Resume

Before listing ThermoSense on your resume, verify:

- [ ] **Real sensor**: Deployed for ≥30 days, not simulated
- [ ] **Real baselines**: Commercial app forecasts collected daily
- [ ] **Real comparison**: ThermoSense vs ≥2 commercial services
- [ ] **Statistical test**: p-value < 0.05 on improvement claim
- [ ] **Public URL**: Anyone can visit and verify claims
- [ ] **Data available**: Raw sensor data downloadable or visible
- [ ] **No cherry-picking**: Results shown for entire period, not selected days

If any of these are missing, your claims are **unverifiable** and a savvy interviewer will notice.

---

## Conclusion

This project has excellent bones - the ML pipeline, feature engineering, and dashboard are genuinely well-built. But right now it's a **demo**, not a **solution**.

The transformation from "impressive tech demo" to "impactful project" requires:

1. **Real data** from a deployed sensor
2. **Real comparison** against services people actually use
3. **Real deployment** where predictions are consumed
4. **Real evidence** with statistical rigor

Complete these, and you'll have one of the strongest ML projects in any portfolio. The hardware cost is ~$25 and the time investment is 4-6 weeks of part-time work.

The question isn't whether you *can* build a system that beats Google Weather for your specific location - you almost certainly can, because microclimate bias is real. The question is whether you'll do the work to *prove* it.

---

## References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Open-Meteo documentation: https://open-meteo.com/en/docs
- DHT22 sensor datasheet: https://www.sparkfun.com/datasheets/Sensors/Temperature/DHT22.pdf
- Weather Underground PWS network: https://www.wunderground.com/pws/overview
