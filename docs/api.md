# ThermoSense API Reference

Interactive docs are also available at `/docs` on a running server
(e.g. https://thermosense-black.vercel.app/docs).

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
