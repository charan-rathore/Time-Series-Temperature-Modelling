# ThermoSense Deployment Guide

This folder contains deployment configurations for various platforms.

## Quick Start

### Option 1: Railway (Recommended for Cloud)

Railway offers a generous free tier (500 hours/month) and easy deployment.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Or connect your GitHub repo in the [Railway Dashboard](https://railway.app).

### Option 2: Render

```bash
# Push to GitHub, then in Render dashboard:
# 1. New > Web Service
# 2. Connect your repo
# 3. Render auto-detects render.yaml
```

### Option 3: Docker

```bash
# Build and run locally
docker build -t thermosense -f deployment/Dockerfile .
docker run -p 8000:8000 thermosense

# Or use docker-compose
docker-compose -f deployment/docker-compose.yml up -d
```

### Option 4: Raspberry Pi Edge Deployment

Run ThermoSense directly on your Pi with Cloudflare Tunnel:

```bash
# Start the API
cd /path/to/thermosense
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Quick tunnel (temporary URL)
./deployment/cloudflare_tunnel.sh quick

# Or set up a permanent tunnel
./deployment/cloudflare_tunnel.sh setup
./deployment/cloudflare_tunnel.sh install
```

## Environment Variables

Set these in your deployment platform:

| Variable | Description | Required |
|----------|-------------|----------|
| `OWM_API_KEY` | OpenWeatherMap API key | Optional (for baseline comparison) |
| `ACCUWEATHER_API_KEY` | AccuWeather API key | Optional (for baseline comparison) |
| `THERMOSENSE_API_URL` | Public URL of deployed API | Optional (for self-reference) |

## Files

| File | Purpose |
|------|---------|
| `railway.toml` | Railway deployment config |
| `railway.json` | Railway build config |
| `render.yaml` | Render.com blueprint |
| `Dockerfile` | Docker container build |
| `docker-compose.yml` | Docker Compose orchestration |
| `cloudflare_tunnel.sh` | Cloudflare Tunnel setup for edge deployment |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Deployment                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Railway   │ OR │   Render    │ OR │   Docker    │      │
│  │   (Free)    │    │   (Free)    │    │  (Self-host)│      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │  FastAPI + React │                        │
│                   │    Dashboard     │                        │
│                   └────────┬────────┘                        │
└────────────────────────────┼────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌───────────────┐  ┌───────────────┐
│ Raspberry Pi    │  │  Open-Meteo   │  │  AccuWeather  │
│ DHT22 Sensor    │  │     API       │  │     API       │
│ (Edge Device)   │  │  (Baseline)   │  │  (Baseline)   │
└─────────────────┘  └───────────────┘  └───────────────┘
```

## Health Check

After deployment, verify your API is running:

```bash
curl https://your-deployment-url.com/api/health
# Should return: {"status":"ok","version":"2.0.0"}
```

## Updating

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
