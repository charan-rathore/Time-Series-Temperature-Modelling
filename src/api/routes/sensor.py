"""
Sensor API routes - Receive and store readings from the Raspberry Pi sensor

Endpoints:
    POST /api/sensor/readings  - Upload batch of sensor readings
    GET  /api/sensor/latest    - Get most recent sensor reading
    GET  /api/sensor/history   - Get sensor reading history
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import os
import hmac
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, Field

from src.data.baseline_collector import store_actual, _DB_PATH, init_database
import yaml

router = APIRouter()


class SensorReading(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    temp_c: float = Field(..., description="Temperature in Celsius")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity percentage")
    source: str = Field("dht22_sensor", description="Data source identifier")


class ReadingsBatch(BaseModel):
    readings: List[SensorReading] = Field(..., min_length=1, max_length=1000)


class ReadingsResponse(BaseModel):
    accepted: int
    rejected: int
    acknowledged: int = Field(..., description="Input readings safely processed, including outside the daily 9 PM window")
    message: str


@router.post("/readings", response_model=ReadingsResponse)
def upload_readings(batch: ReadingsBatch, x_api_key: Optional[str] = Header(None)):
    """
    Upload a batch of sensor readings.
    
    This endpoint is called by the Raspberry Pi uploader to sync local readings
    to the cloud database. Readings are stored as daily actuals for the 9 PM
    comparison window.
    """
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Sensor storage is not configured for durable cloud writes")
    expected_key = os.environ.get("THERMOSENSE_API_KEY")
    if not expected_key or not x_api_key or not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(401, "Sensor upload requires a configured API key")
    init_database(_DB_PATH)
    with open(Path(__file__).resolve().parents[3] / "config" / "config.yaml") as config_file:
        site_timezone = ZoneInfo(yaml.safe_load(config_file)["location"]["timezone"])

    accepted = 0
    rejected = 0
    valid_readings = 0
    
    readings_by_date = {}
    
    for reading in batch.readings:
        try:
            ts = datetime.fromisoformat(reading.timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                raise ValueError("timestamp needs timezone")
            if not (-40 <= reading.temp_c <= 80) or (reading.humidity_pct is not None and not 0 <= reading.humidity_pct <= 100) or reading.source != "dht22_sensor":
                rejected += 1
                continue
        except ValueError:
            rejected += 1
            continue
        
        valid_readings += 1
        local_ts = ts.astimezone(site_timezone)
        reading_date = local_ts.date()
        distance = abs((local_ts - datetime.combine(reading_date, datetime.min.time(), site_timezone).replace(hour=21)).total_seconds())

        if distance <= 1800:
            if reading_date not in readings_by_date:
                readings_by_date[reading_date] = reading
            else:
                existing_ts = datetime.fromisoformat(
                    readings_by_date[reading_date].timestamp.replace("Z", "+00:00")
                )
                existing_local = existing_ts.astimezone(site_timezone)
                existing_distance = abs((existing_local - datetime.combine(reading_date, datetime.min.time(), site_timezone).replace(hour=21)).total_seconds())
                if distance < existing_distance:
                    readings_by_date[reading_date] = reading
    
    for reading_date, reading in readings_by_date.items():
        if store_actual(
            _DB_PATH,
            reading_date,
            reading.temp_c,
            reading.humidity_pct,
            source=reading.source,
        ):
            accepted += 1
        else:
            rejected += 1
    
    skipped = valid_readings - len(readings_by_date)
    acknowledged = valid_readings if rejected == 0 and accepted == len(readings_by_date) else 0
    
    return ReadingsResponse(
        accepted=accepted,
        rejected=rejected,
        acknowledged=acknowledged,
        message=f"Processed {len(batch.readings)} readings. "
                f"{accepted} stored as daily actuals, {skipped} outside 9 PM window.",
    )


@router.get("/latest")
def get_latest_reading():
    """Get the most recent sensor reading stored as a daily actual."""
    import sqlite3
    
    if not _DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No sensor data available")
    
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("""
        SELECT date, sensor_temp_c, sensor_humidity_pct, recorded_at, source
        FROM daily_actuals
        ORDER BY date DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="No sensor data available")
    
    return dict(row)


@router.get("/history")
def get_sensor_history(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(30, ge=1, le=365, description="Maximum records to return"),
):
    """Get sensor reading history."""
    import sqlite3
    
    if not _DB_PATH.exists():
        return {"readings": [], "count": 0}
    
    start = date.fromisoformat(start_date) if start_date else date.today() - timedelta(days=limit)
    end = date.fromisoformat(end_date) if end_date else date.today()
    
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("""
        SELECT date, sensor_temp_c, sensor_humidity_pct, api_temp_c, recorded_at, source
        FROM daily_actuals
        WHERE date >= ? AND date <= ?
        ORDER BY date DESC
        LIMIT ?
    """, (start.isoformat(), end.isoformat(), limit))
    
    readings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {
        "readings": readings,
        "count": len(readings),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
