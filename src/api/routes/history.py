"""History endpoint for ThermoSense API."""

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class HistoricalPoint(BaseModel):
    date: str
    actual_temp_c: Optional[float]
    api_temp_c: Optional[float]
    api_bias: Optional[float]


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

    end_date = date.today() if end is None else date.fromisoformat(end)

    config = request.app.state.config
    location_name = config["location"]["name"]

    # Placeholder — replace with actual parquet read from data/processed/
    return HistoryResponse(
        location=location_name,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        records=[],
        total_records=0,
    )
