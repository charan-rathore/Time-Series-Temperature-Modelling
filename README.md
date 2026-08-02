# ThermoSense

**Hyperlocal temperature forecasts for your exact location - not the nearest weather station.**

Live dashboard: [thermosense-black.vercel.app](https://thermosense-black.vercel.app)

Latest Open-Meteo Bangalore demo metrics: [docs/demo-results.md](docs/demo-results.md) · [Watch the live training demo](#live-demo)

---

## Why this project exists

Commercial weather apps (Google Weather, AccuWeather, OpenWeatherMap, and similar) do not measure temperature at your balcony, courtyard, or rooftop. They report a value from a regional model or the nearest official station - often kilometers away, at a different elevation, and in a different microclimate.

That gap is usually systematic, not random. Urban heat islands, building geometry, vegetation, and local humidity create a persistent offset between “the app” and what you actually feel. ThermoSense exists to measure that offset at your location and correct for it.

---

## What this project is

ThermoSense is an end-to-end IoT + ML product that:

1. Collects ground-truth temperature/humidity from a physical sensor (Raspberry Pi + DHT22), or starts from historical/API data while you add hardware later
2. Pulls commercial weather API forecasts for the same coordinates
3. Learns the local bias (`sensor − API`) and trains forecasting models on your site’s data
4. Serves a live dashboard and API with 3-day hyperlocal forecasts and a public accuracy leaderboard against commercial baselines

Typical use cases: personal hyperlocal forecasts, proving that a location-specific model beats generic apps with real metrics, portfolio/demo of applied ML + IoT, or a starting point for agriculture/energy alerts tied to *your* conditions.

Deeper system design, data lifecycle, models, and API surface live in:

- [docs/architecture.md](docs/architecture.md)
- [docs/api.md](docs/api.md)
- [hardware/README.md](hardware/README.md) (sensor wiring and Pi install)
- [deployment/README.md](deployment/README.md) (Vercel, Docker, Railway, etc.)

---

## Why use ThermoSense (value proposition)

| Generic weather apps | ThermoSense |
|----------------------|-------------|
| One-size-fits-region forecast | Forecast corrected for **your** microclimate |
| No ground truth at your site | Optional physical sensor closes the loop |
| Accuracy claims without your data | Live leaderboard vs Open-Meteo / OWM / AccuWeather on **your** observations |
| Black-box apps | Trainable models (SARIMA, LightGBM, ensemble) you control and retrain |
| No feedback path | Dashboard feedback + continuous daily pipeline |

**Clear advantage:** instead of trusting a grid-cell average, ThermoSense learns the bias between commercial APIs and your location, then applies that correction going forward - and shows the scoreboard so you can verify the improvement.

---

## Live demo

Watch ThermoSense on the production site: open the **Pipeline** page (last item in the left sidebar), train SARIMA / LightGBM / Ensemble, then read the Metrics to see why lower MAE and RMSE matter for hyperlocal forecasts.

<video src="docs/videos/thermosense-training-demo.mp4" controls width="100%" poster=""></video>

[Download / open the demo video](docs/videos/thermosense-training-demo.mp4) · Live app: [thermosense-black.vercel.app](https://thermosense-black.vercel.app)

## Setup guide

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the React dashboard)
- Git
- Optional: Raspberry Pi + DHT22 for live sensor ground truth (~$25)

### 1. Clone and install

```bash
git clone https://github.com/charan-rathore/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional (TFT deep-learning model - needs PyTorch):

```bash
pip install torch pytorch-forecasting pytorch-lightning
```

### 2. Configure location and secrets

```bash
cp .env.example .env
```

Set your site in `config/config.yaml`:

```yaml
location:
  name: "Your City"
  lat: 12.9716
  lon: 77.5946
  timezone: "Asia/Kolkata"
```

Optional keys in `.env` (only needed to compare against commercial baselines on the leaderboard):

```bash
OWM_API_KEY=your_openweathermap_key           # free tier: openweathermap.org
ACCUWEATHER_API_KEY=your_accuweather_key      # free tier: developer.accuweather.com
```

Open-Meteo (primary weather source) needs **no API key**.

### 3. Backfill data and train models

First-time historical pull (~365 days):

```bash
python scripts/run_pipeline.py --mode backfill
```

Daily incremental update (manual or cron):

```bash
python scripts/run_pipeline.py --mode daily
```

Train the core models:

```bash
# Fast path (~2 minutes): SARIMA + LightGBM + Ensemble
python scripts/train_models.py --models sarima lgbm ensemble

# Include TFT if you installed PyTorch (~10 minutes)
python scripts/train_models.py --models sarima lgbm tft ensemble
```

### 4. Run locally

```bash
# Build the dashboard
cd frontend && npm install && npm run build && cd ..

# API + dashboard (http://localhost:8000)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Development hot-reload (optional second terminal):

```bash
cd frontend && npm start          # http://localhost:3000 (proxies API to :8000)
```

Useful checks:

```bash
curl http://localhost:8000/api/health
pytest tests/ -v
mlflow ui --port 5000             # optional experiment UI
```

### 5. Hardware sensor (optional but recommended)

Full wiring, systemd install, and uploader config: [hardware/README.md](hardware/README.md).

Short path on a Raspberry Pi:

```bash
cd hardware
pip3 install -r requirements.txt
python3 sensor_daemon.py --simulate   # dry run
python3 sensor_daemon.py              # real DHT22 on GPIO 4
sudo ./install.sh                     # start on boot

export THERMOSENSE_API_URL=https://thermosense-black.vercel.app
python3 uploader.py --continuous
```

Place the sensor shaded, ventilated, ~1.5-2 m above ground, and record exact GPS coordinates for fair API comparison.

### 6. Deploy to production

**Vercel (current production host)**

```bash
npm i -g vercel
vercel link
vercel --prod
```

Or import the GitHub repo at [vercel.com/new](https://vercel.com/new).

Production URL: https://thermosense-black.vercel.app  
Health: https://thermosense-black.vercel.app/api/health

Other options (Docker, Railway, Cloudflare Tunnel): [deployment/README.md](deployment/README.md).

### 7. Daily operations (cron)

```bash
# After the 9 PM sensor snapshot
0 22 * * * cd /path/to/thermosense && .venv/bin/python scripts/run_pipeline.py --mode daily

# Collect commercial baselines for the leaderboard
0 18 * * * cd /path/to/thermosense && .venv/bin/python -m src.data.baseline_collector --collect

# Weekly retrain
0 0 * * 0 cd /path/to/thermosense && .venv/bin/python scripts/train_models.py --models sarima lgbm ensemble
```

---

## Contact

If any query, please reach out to:

- **Charan Rathore**
- Email: ra7hore.charan@gmail.com
- Phone: 6303460570

---

## License

MIT - see [LICENSE](LICENSE) if present in the repo.
