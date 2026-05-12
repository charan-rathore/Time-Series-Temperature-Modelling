#!/usr/bin/env python3
"""
ThermoSense Sensor Daemon — Raspberry Pi + DHT22

Continuously reads temperature and humidity from a DHT22 sensor,
stores readings in a local SQLite database, and exposes a simple
HTTP endpoint for the uploader to query.

Hardware Setup:
    - DHT22 data pin connected to GPIO 4 (BCM numbering)
    - DHT22 VCC to 3.3V or 5V
    - DHT22 GND to GND
    - 10K pull-up resistor between data and VCC (some modules have this built-in)

Run on Raspberry Pi:
    python3 sensor_daemon.py

Install as systemd service:
    sudo cp thermosense-sensor.service /etc/systemd/system/
    sudo systemctl enable thermosense-sensor
    sudo systemctl start thermosense-sensor
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import Adafruit_DHT
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    print("[WARN] Adafruit_DHT not available. Running in simulation mode.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sensor_daemon.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent / "readings.db"
DEFAULT_GPIO_PIN = 4
DEFAULT_READ_INTERVAL_SECONDS = 300  # 5 minutes
DEFAULT_HTTP_PORT = 8081


def init_database(db_path: Path) -> None:
    """Create the readings table if it doesn't exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temp_c REAL NOT NULL,
            humidity_pct REAL,
            source TEXT DEFAULT 'dht22_sensor',
            synced INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_readings_timestamp 
        ON sensor_readings(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_readings_synced 
        ON sensor_readings(synced)
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized: {db_path}")


def read_dht22(gpio_pin: int, retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Read temperature and humidity from DHT22 sensor.
    
    Returns None if reading fails after all retries.
    """
    if not SENSOR_AVAILABLE:
        return simulate_reading()
    
    for attempt in range(1, retries + 1):
        humidity, temperature = Adafruit_DHT.read_retry(
            Adafruit_DHT.DHT22, gpio_pin, retries=5, delay_seconds=2
        )
        
        if humidity is not None and temperature is not None:
            if -40 <= temperature <= 80 and 0 <= humidity <= 100:
                return {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "temp_c": round(temperature, 2),
                    "humidity_pct": round(humidity, 2),
                    "source": "dht22_sensor",
                }
            else:
                logger.warning(f"Invalid reading: temp={temperature}, humidity={humidity}")
        
        logger.warning(f"Read attempt {attempt}/{retries} failed")
        time.sleep(2)
    
    return None


def simulate_reading() -> Dict[str, Any]:
    """
    Generate a simulated reading for testing without hardware.
    
    Simulates a realistic temperature pattern based on time of day
    with some random variation.
    """
    import random
    
    now = datetime.now()
    hour = now.hour
    
    base_temp = 25.0
    if 6 <= hour < 12:
        temp_offset = (hour - 6) * 0.8
    elif 12 <= hour < 18:
        temp_offset = 4.8 - (hour - 12) * 0.3
    else:
        temp_offset = -2.0 + random.uniform(-1, 1)
    
    temperature = base_temp + temp_offset + random.uniform(-0.5, 0.5)
    humidity = 60 + random.uniform(-15, 15) + (10 if 18 <= hour or hour < 6 else 0)
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temp_c": round(temperature, 2),
        "humidity_pct": round(min(100, max(0, humidity)), 2),
        "source": "simulated",
    }


def store_reading(reading: Dict[str, Any], db_path: Path) -> int:
    """Store a reading in the database. Returns the row ID."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        INSERT INTO sensor_readings (timestamp, temp_c, humidity_pct, source)
        VALUES (?, ?, ?, ?)
        """,
        (reading["timestamp"], reading["temp_c"], reading["humidity_pct"], reading["source"]),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_latest_reading(db_path: Path) -> Optional[Dict[str, Any]]:
    """Get the most recent reading from the database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, timestamp, temp_c, humidity_pct, source, synced
        FROM sensor_readings
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_unsynced_readings(db_path: Path, limit: int = 1000) -> list:
    """Get readings that haven't been synced to the cloud."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, timestamp, temp_c, humidity_pct, source
        FROM sensor_readings
        WHERE synced = 0
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def mark_as_synced(db_path: Path, reading_ids: list) -> int:
    """Mark readings as synced. Returns count of updated rows."""
    if not reading_ids:
        return 0
    
    conn = sqlite3.connect(str(db_path))
    placeholders = ",".join("?" * len(reading_ids))
    cursor = conn.execute(
        f"UPDATE sensor_readings SET synced = 1 WHERE id IN ({placeholders})",
        reading_ids,
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def get_9pm_reading(db_path: Path, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the reading closest to 9 PM (21:00) for a given date.
    
    This is the target reading for comparison with forecasts.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    start = f"{date}T20:30:00Z"
    end = f"{date}T21:30:00Z"
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, timestamp, temp_c, humidity_pct, source
        FROM sensor_readings
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ? || 'T21:00:00Z'))
        LIMIT 1
        """,
        (start, end, date),
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_stats(db_path: Path, hours: int = 24) -> Dict[str, Any]:
    """Get statistics for the last N hours."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        SELECT 
            COUNT(*) as count,
            AVG(temp_c) as avg_temp,
            MIN(temp_c) as min_temp,
            MAX(temp_c) as max_temp,
            AVG(humidity_pct) as avg_humidity,
            SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as unsynced_count
        FROM sensor_readings
        WHERE timestamp >= ?
        """,
        (cutoff,),
    )
    row = cursor.fetchone()
    conn.close()
    
    return {
        "period_hours": hours,
        "reading_count": row[0] or 0,
        "avg_temp_c": round(row[1], 2) if row[1] else None,
        "min_temp_c": round(row[2], 2) if row[2] else None,
        "max_temp_c": round(row[3], 2) if row[3] else None,
        "avg_humidity_pct": round(row[4], 2) if row[4] else None,
        "unsynced_count": row[5] or 0,
    }


class SensorHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to expose sensor data."""
    
    db_path = DEFAULT_DB_PATH
    
    def log_message(self, format, *args):
        logger.debug(f"HTTP: {format % args}")
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "sensor_available": SENSOR_AVAILABLE})
        
        elif self.path == "/latest":
            reading = get_latest_reading(self.db_path)
            if reading:
                self._send_json(reading)
            else:
                self._send_json({"error": "No readings available"}, 404)
        
        elif self.path == "/unsynced":
            readings = get_unsynced_readings(self.db_path)
            self._send_json({"count": len(readings), "readings": readings})
        
        elif self.path == "/stats":
            stats = get_stats(self.db_path)
            self._send_json(stats)
        
        elif self.path.startswith("/9pm"):
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            date = query.get("date", [None])[0]
            reading = get_9pm_reading(self.db_path, date)
            if reading:
                self._send_json(reading)
            else:
                self._send_json({"error": f"No 9 PM reading for {date or 'today'}"}, 404)
        
        else:
            self._send_json({
                "endpoints": [
                    "GET /health - Check daemon status",
                    "GET /latest - Get most recent reading",
                    "GET /unsynced - Get readings not yet synced to cloud",
                    "GET /stats - Get 24h statistics",
                    "GET /9pm?date=YYYY-MM-DD - Get 9 PM reading for a date",
                ],
            })
    
    def do_POST(self):
        if self.path == "/mark_synced":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()
            data = json.loads(body)
            ids = data.get("ids", [])
            count = mark_as_synced(self.db_path, ids)
            self._send_json({"marked_synced": count})
        else:
            self._send_json({"error": "Not found"}, 404)


def run_http_server(port: int, db_path: Path):
    """Run the HTTP server in a separate thread."""
    SensorHTTPHandler.db_path = db_path
    server = HTTPServer(("0.0.0.0", port), SensorHTTPHandler)
    logger.info(f"HTTP server listening on port {port}")
    server.serve_forever()


def main_loop(gpio_pin: int, interval: int, db_path: Path):
    """Main sensor reading loop."""
    logger.info(f"Starting sensor loop: GPIO={gpio_pin}, interval={interval}s, db={db_path}")
    
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    while True:
        try:
            reading = read_dht22(gpio_pin)
            
            if reading:
                row_id = store_reading(reading, db_path)
                logger.info(
                    f"Stored reading #{row_id}: "
                    f"{reading['temp_c']}°C, {reading['humidity_pct']}% humidity"
                )
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.error(f"Failed to read sensor ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical("Too many consecutive failures. Check sensor connection!")
        
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            consecutive_failures += 1
        
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="ThermoSense Sensor Daemon")
    parser.add_argument(
        "--gpio", type=int, default=DEFAULT_GPIO_PIN,
        help=f"GPIO pin for DHT22 data (default: {DEFAULT_GPIO_PIN})"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_READ_INTERVAL_SECONDS,
        help=f"Reading interval in seconds (default: {DEFAULT_READ_INTERVAL_SECONDS})"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--http-port", type=int, default=DEFAULT_HTTP_PORT,
        help=f"HTTP server port (default: {DEFAULT_HTTP_PORT})"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Force simulation mode even if Adafruit_DHT is available"
    )
    
    args = parser.parse_args()
    
    if args.simulate:
        global SENSOR_AVAILABLE
        SENSOR_AVAILABLE = False
        logger.info("Simulation mode enabled via --simulate flag")
    
    db_path = Path(args.db)
    init_database(db_path)
    
    http_thread = threading.Thread(
        target=run_http_server,
        args=(args.http_port, db_path),
        daemon=True,
    )
    http_thread.start()
    
    try:
        main_loop(args.gpio, args.interval, db_path)
    except KeyboardInterrupt:
        logger.info("Shutting down sensor daemon...")
        sys.exit(0)


if __name__ == "__main__":
    main()
