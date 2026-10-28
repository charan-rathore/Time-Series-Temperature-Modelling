"""Opt-in authenticated raw history; disabled unless operator configures both secrets."""
import hmac
import os
from dataclasses import asdict
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Query, Depends
from pydantic import BaseModel, ConfigDict, Field
from src.history.observations import Observation, summarize
from src.history.egres import EgresHistory, HistoryConflict, HistoryLimit

router=APIRouter()

class Reading(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    location: str
    source: str
    observed_at: str
    temperature_c: float=Field(allow_inf_nan=False)
    humidity_pct: Optional[float]=Field(default=None,allow_inf_nan=False)
    provenance: str='live'
    reference: Optional[str]=None

class Batch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    readings: list[Reading]=Field(min_length=1,max_length=1000)

def store(authorization: Optional[str]=Header(None)):
    key=os.environ.get('THERMOSENSE_HISTORY_TOKEN','')
    dsn=os.environ.get('THERMOSENSE_HISTORY_DSN','')
    if not key or not dsn:raise HTTPException(503,'Raw history is not configured')
    if not authorization or not hmac.compare_digest(authorization.encode('utf-8'),('Bearer '+key).encode('utf-8')):
        raise HTTPException(401,'History authorization required')
    return EgresHistory(dsn)

@router.post('/readings')
def append(batch:Batch, db=Depends(store)):
    try:
        # Validate entire batch before any write. An interrupted write can be retried safely.
        rows=[Observation.create(**r.model_dump()) for r in batch.readings]
    except ValueError as e:raise HTTPException(422,str(e))
    try:
        written=sum(db.append(r) for r in rows)
    except HistoryConflict as e:raise HTTPException(409,str(e))
    except Exception:raise HTTPException(503,'History write unavailable; retry the batch')
    return dict(stored=written,duplicates=len(rows)-written,
                batch_atomic=False,message='Only fully acknowledged batches may be marked synced')

@router.get('/readings')
def read(location:str,start:str,end:str,source:str,provenance:str='live',
         expected_interval_seconds:int=Query(300,ge=1,le=86400),db=Depends(store)):
    try:
        rows=db.read(location=location,start=start,end=end,source=source,provenance=provenance)
        return dict(readings=[asdict(r) for r in rows],summary=summarize(rows,expected_interval_seconds),
                    date_basis='UTC',retention='append-only; no automatic deletion')
    except (ValueError,HistoryLimit) as e:raise HTTPException(422,str(e))
    except Exception:raise HTTPException(503,'History read unavailable')
