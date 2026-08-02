#!/usr/bin/env python3
"""
ThermoSense Uploader - Syncs sensor readings to cloud API

Periodically fetches unsynced readings from the local sensor database
and uploads them to the ThermoSense cloud API. Handles network failures
gracefully with exponential backoff.

Run as cron job or systemd timer:
    */15 * * * * /path/to/python /path/to/uploader.py

Or run continuously:
    python3 uploader.py --continuous --interval 900
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[ERROR] requests library required. Install with: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("uploader.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_SENSOR_URL = "http://localhost:8081"
DEFAULT_CLOUD_API_URL = os.environ.get("THERMOSENSE_API_URL", "http://localhost:8000")
DEFAULT_BATCH_SIZE = 100
DEFAULT_UPLOAD_INTERVAL = 900  # 15 minutes


def fetch_unsynced_readings(sensor_url: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetch unsynced readings from the local sensor daemon."""
    try:
        resp = requests.get(f"{sensor_url}/unsynced", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("readings", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch unsynced readings: {e}")
        return []


def mark_readings_synced(sensor_url: str, reading_ids: List[int]) -> bool:
    """Mark readings as synced in the local sensor database."""
    if not reading_ids:
        return True
    
    try:
        resp = requests.post(
            f"{sensor_url}/mark_synced",
            json={"ids": reading_ids},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to mark readings as synced: {e}")
        return False


def upload_readings_to_cloud(
    cloud_url: str,
    readings: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> tuple:
    """
    Upload readings to the ThermoSense cloud API.
    
    Returns (success_count, failed_ids).
    """
    if not readings:
        return 0, []
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    endpoint = urljoin(cloud_url, "/api/sensor/readings")
    
    try:
        payload = {
            "readings": [
                {
                    "timestamp": r["timestamp"],
                    "temp_c": r["temp_c"],
                    "humidity_pct": r.get("humidity_pct"),
                    "source": r.get("source", "dht22_sensor"),
                }
                for r in readings
            ]
        }
        
        resp = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        
        return result.get("accepted", len(readings)), []
    
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 422:
            logger.warning(f"Some readings rejected by API: {e.response.text}")
            return 0, [r["id"] for r in readings]
        logger.error(f"HTTP error uploading readings: {e}")
        return 0, [r["id"] for r in readings]
    
    except requests.RequestException as e:
        logger.error(f"Network error uploading readings: {e}")
        return 0, [r["id"] for r in readings]


def upload_batch(
    sensor_url: str,
    cloud_url: str,
    api_key: Optional[str],
    batch_size: int,
) -> Dict[str, int]:
    """
    Fetch, upload, and mark a batch of readings.
    
    Returns statistics about the operation.
    """
    stats = {
        "fetched": 0,
        "uploaded": 0,
        "marked_synced": 0,
        "failed": 0,
    }
    
    readings = fetch_unsynced_readings(sensor_url, limit=batch_size)
    stats["fetched"] = len(readings)
    
    if not readings:
        logger.debug("No unsynced readings to upload")
        return stats
    
    logger.info(f"Uploading {len(readings)} readings to {cloud_url}")
    
    success_count, failed_ids = upload_readings_to_cloud(cloud_url, readings, api_key)
    stats["uploaded"] = success_count
    stats["failed"] = len(failed_ids)
    
    successful_ids = [r["id"] for r in readings if r["id"] not in failed_ids]
    
    if successful_ids:
        if mark_readings_synced(sensor_url, successful_ids):
            stats["marked_synced"] = len(successful_ids)
            logger.info(f"Marked {len(successful_ids)} readings as synced")
        else:
            logger.warning("Failed to mark readings as synced locally")
    
    return stats


def run_once(
    sensor_url: str,
    cloud_url: str,
    api_key: Optional[str],
    batch_size: int,
) -> Dict[str, int]:
    """Run a single upload cycle."""
    total_stats = {
        "fetched": 0,
        "uploaded": 0,
        "marked_synced": 0,
        "failed": 0,
        "batches": 0,
    }
    
    while True:
        batch_stats = upload_batch(sensor_url, cloud_url, api_key, batch_size)
        
        total_stats["fetched"] += batch_stats["fetched"]
        total_stats["uploaded"] += batch_stats["uploaded"]
        total_stats["marked_synced"] += batch_stats["marked_synced"]
        total_stats["failed"] += batch_stats["failed"]
        total_stats["batches"] += 1
        
        if batch_stats["fetched"] < batch_size:
            break
    
    return total_stats


def run_continuous(
    sensor_url: str,
    cloud_url: str,
    api_key: Optional[str],
    batch_size: int,
    interval: int,
):
    """Run upload cycles continuously at the specified interval."""
    logger.info(f"Starting continuous upload: interval={interval}s")
    
    while True:
        try:
            stats = run_once(sensor_url, cloud_url, api_key, batch_size)
            
            if stats["uploaded"] > 0:
                logger.info(
                    f"Upload cycle complete: "
                    f"{stats['uploaded']} uploaded, "
                    f"{stats['marked_synced']} marked synced, "
                    f"{stats['failed']} failed"
                )
        
        except Exception as e:
            logger.exception(f"Error in upload cycle: {e}")
        
        logger.debug(f"Sleeping for {interval} seconds...")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="ThermoSense Uploader")
    parser.add_argument(
        "--sensor-url", type=str, default=DEFAULT_SENSOR_URL,
        help=f"Local sensor daemon URL (default: {DEFAULT_SENSOR_URL})"
    )
    parser.add_argument(
        "--cloud-url", type=str, default=DEFAULT_CLOUD_API_URL,
        help=f"Cloud API URL (default: {DEFAULT_CLOUD_API_URL})"
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("THERMOSENSE_API_KEY"),
        help="API key for cloud authentication (or set THERMOSENSE_API_KEY env var)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Readings per upload batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously instead of one-shot"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_UPLOAD_INTERVAL,
        help=f"Upload interval in seconds for continuous mode (default: {DEFAULT_UPLOAD_INTERVAL})"
    )
    
    args = parser.parse_args()
    
    if args.continuous:
        run_continuous(
            args.sensor_url,
            args.cloud_url,
            args.api_key,
            args.batch_size,
            args.interval,
        )
    else:
        stats = run_once(
            args.sensor_url,
            args.cloud_url,
            args.api_key,
            args.batch_size,
        )
        
        print(json.dumps(stats, indent=2))
        
        if stats["failed"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
