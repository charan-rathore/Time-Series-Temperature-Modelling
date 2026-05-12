"""History endpoint for ThermoSense API."""

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "daily_merged.parquet"


class HistoricalPoint(BaseModel):
    date: str
    actual_temp_c: Optional[float] = None
    api_temp_c: Optional[float] = None
    api_bias: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    is_sensor_reading: Optional[bool] = None


class HistoryResponse(BaseModel):
    location: str
    start_date: str
    end_date: str
    records: List[HistoricalPoint]
    total_records: int


@router.get("", response_model=HistoryResponse, summary="Get historical temperature data")
def get_history(
    request: Request,
    start: str = "2024-06-01",
    end: Optional[str] = None,
):
    """
    Return historical temperature records and API bias values.

    Query params:
    - start: Start date in YYYY-MM-DD format (default: 2024-06-01)
    - end: End date in YYYY-MM-DD format (default: today)
    """
    try:
        start_date = date.fromisoformat(start)
    except ValueError:
        raise HTTPException(status_code=400, detail="start must be YYYY-MM-DD")

    try:
        end_date = date.today() if end is None else date.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="end must be YYYY-MM-DD")

    config = request.app.state.config
    location_name = config["location"]["name"]

    records = []
    if PROCESSED_PATH.exists():
        try:
            df = pd.read_parquet(PROCESSED_PATH)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
            filtered = df[mask].copy()

            for _, row in filtered.iterrows():
                records.append(HistoricalPoint(
                    date=str(row["date"].date()),
                    actual_temp_c=_safe_float(row.get("temp_c")),
                    api_temp_c=_safe_float(row.get("temp_c_api")),
                    api_bias=_safe_float(row.get("api_bias") if "api_bias" in row.index else None),
                    humidity_pct=_safe_float(row.get("humidity_pct")),
                    pressure_hpa=_safe_float(row.get("pressure_hpa")),
                    is_sensor_reading=bool(row.get("is_sensor_reading", False)),
                ))
        except Exception as e:
            print(f"[history] Error reading processed data: {e}")

    return HistoryResponse(
        location=location_name,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        records=records,
        total_records=len(records),
    )


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return round(f, 2)
    except (ValueError, TypeError):
        return None
