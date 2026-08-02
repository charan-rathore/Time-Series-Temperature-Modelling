# ThermoSense Deployment Guide

This folder contains deployment configurations for various platforms.

## Quick Start

### Option 1: Vercel (Recommended)

ThermoSense runs as a FastAPI app on Vercel Fluid Compute, with the React
dashboard built into `public/` and served from the CDN.

```bash
# Install CLI
npm i -g vercel

# Link + deploy
vercel link
vercel --prod
```

Or import the GitHub repo at [vercel.com/new](https://vercel.com/new).
Vercel detects FastAPI via `main.py` / `pyproject.toml` and runs
`scripts/vercel_build.sh` to build the dashboard.

Optional env vars (Project Settings → Environment Variables):

| Variable | Description | Required |
|----------|-------------|----------|
| `OWM_API_KEY` | OpenWeatherMap API key | Optional (for baseline comparison) |
| `ACCUWEATHER_API_KEY` | AccuWeather API key | Optional (for baseline comparison) |
| `THERMOSENSE_API_URL` | Public URL of deployed API | Optional (for self-reference) |
| `LOCATION_NAME` | Display name for the sensor site | Optional |
| `LOCATION_LAT` / `LOCATION_LON` | Coordinates | Optional |

### Option 2: Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

Or connect your GitHub repo in the [Railway Dashboard](https://railway.app).

### Option 3: Render

```bash
# Push to GitHub, then in Render dashboard:
# 1. New > Web Service
# 2. Connect your repo
# 3. Render auto-detects render.yaml
```

### Option 4: Docker

```bash
docker build -t thermosense -f deployment/Dockerfile .
docker run -p 8000:8000 thermosense

# Or use docker-compose
docker-compose -f deployment/docker-compose.yml up -d
```

### Option 5: Raspberry Pi Edge Deployment

Run ThermoSense directly on your Pi with Cloudflare Tunnel:

```bash
cd /path/to/thermosense
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Quick tunnel (temporary URL)
./deployment/cloudflare_tunnel.sh quick

# Or set up a permanent tunnel
./deployment/cloudflare_tunnel.sh setup
./deployment/cloudflare_tunnel.sh install
```

## Files

| File | Purpose |
|------|---------|
| `../vercel.json` | Vercel function config (maxDuration, includeFiles) |
| `../pyproject.toml` | Vercel FastAPI entrypoint + build script |
| `../main.py` | Root entrypoint for Vercel detection |
| `../scripts/vercel_build.sh` | Builds React dashboard into `public/` |
| `railway.toml` | Railway deployment config |
| `railway.json` | Railway build config |
| `render.yaml` | Render.com blueprint |
| `Dockerfile` | Docker container build |
| `docker-compose.yml` | Docker Compose orchestration |
| `cloudflare_tunnel.sh` | Cloudflare Tunnel setup for edge deployment |

## Architecture

```
+---------------------------------------------------------------+
|                     Cloud Deployment                          |
|  +-------------+  +-------------+  +-------------+            |
|  |   Vercel    |  |   Railway   |  |   Render    |  Docker    |
|  |(Recommended)|  |             |  |             |            |
|  +------+------+  +------+------+  +------+------+            |
|         +----------------+----------------+                   |
|                          |                                    |
|                 +--------v--------+                           |
|                 | FastAPI + React |                           |
|                 |    Dashboard    |                           |
|                 +--------+--------+                           |
+--------------------------+------------------------------------+
                           |
         +-----------------+-----------------+
         v                 v                 v
+-----------------+ +---------------+ +---------------+
| Raspberry Pi    | |  Open-Meteo   | |  AccuWeather  |
| DHT22 Sensor    | |     API       | |     API       |
+-----------------+ +---------------+ +---------------+
```

## Health Check

```bash
curl https://your-deployment-url.com/api/health
# Should return: {"status":"ok","version":"2.0.0"}
```

## Updating

### Vercel
```bash
vercel --prod
# or: git push (with Git integration connected)
```

### Railway
```bash
railway up
```

### Render
Push to GitHub - auto-deploys on merge to main.

### Docker
```bash
docker-compose -f deployment/docker-compose.yml down
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d
```
