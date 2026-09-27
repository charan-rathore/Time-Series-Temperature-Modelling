# ThermoSense API Reference

Interactive docs are also available at `/docs` on a running server
(e.g. https://thermosense-black.vercel.app/docs).

All endpoints are prefixed with `/api`. The public Vercel deployment is read-only: write/training endpoints return 503. On a persistent self-hosted deployment, `POST /api/sensor/readings`, manual feedback and pipeline POSTs require a configured `THERMOSENSE_API_KEY` and matching `X-API-Key` header. The sample forecast below illustrates the response format; it is not a current production prediction.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast?days=3` | 3-day temperature forecast |
| `POST` | `/api/forecast/feedback` | Submit actual observation (persistent self-hosted service only) |
| `GET` | `/api/history?start=2024-06-01&end=2024-07-11` | Historical data |
| `GET` | `/api/metrics?window_days=30` | Model accuracy metrics |
| `GET` | `/api/leaderboard?window_days=30&horizon=1` | Leaderboard framework; no current live ranking |
| `GET` | `/api/statistics?window_days=30` | Statistical significance tests |
| `POST` | `/api/sensor/readings` | Receive sensor uploads (API key + persistent service only) |
| `GET` | `/api/pipeline/status` | System health |
| `POST` | `/api/pipeline/backfill` | Trigger data backfill (API key + persistent service only) |
| `POST` | `/api/pipeline/train` | Disabled until validated prospective evaluation |
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
