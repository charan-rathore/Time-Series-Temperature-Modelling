"""Immutable source-labelled history. Never synthesizes missing observations."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta, date
import hashlib
import json
import math
import re

_IDENTIFIER = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')

def utc(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('observed_at must include a timezone')
    return t.astimezone(timezone.utc)

def stamp(t):
    return t.isoformat(timespec='microseconds').replace('+00:00','Z')

def identifier(value):
    if not isinstance(value,str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('invalid location or source identifier')
    return value

def partition_key(location, day):
    identifier(location); date.fromisoformat(day)
    return location+'|'+day

@dataclass(frozen=True)
class Observation:
    id: str
    location: str
    source: str
    observed_at: str
    original_observed_at: str
    received_at: str
    temperature_c: float
    humidity_pct: float | None
    provenance: str
    reference: str | None

    @classmethod
    def create(cls, *, location, source, observed_at, temperature_c, humidity_pct=None,
               provenance='live', reference=None):
        identifier(location); identifier(source)
        t=utc(observed_at); now=datetime.now(timezone.utc)
        if t>now+timedelta(minutes=5): raise ValueError('observation is in the future')
        if provenance not in ('live','backfill'): raise ValueError('invalid provenance')
        if provenance=='backfill' and (not isinstance(reference,str) or not reference.strip()):
            raise ValueError('backfill requires a source reference')
        if reference is not None and (not isinstance(reference,str) or len(reference)>2048): raise ValueError('invalid reference')
        def numeric(value, low, high):
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('invalid measurement')
            return float(value)
        temperature_c=numeric(temperature_c,-100,100)
        if humidity_pct is not None: humidity_pct=numeric(humidity_pct,0,100)
        normalized=stamp(t)
        key=json.dumps([location,source,normalized],separators=(',',':'))
        return cls(hashlib.sha256(key.encode()).hexdigest(),location,source,normalized,
                   observed_at,stamp(now),temperature_c,humidity_pct,provenance,reference)

    def payload(self): return json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False)
    def same_measurement(self, other):
        # A timezone-equivalent replay has the same identity. Retain the first original timestamp/receipt.
        return all(getattr(self,k)==getattr(other,k) for k in
                   ('id','location','source','observed_at','temperature_c','humidity_pct','provenance','reference'))

def summarize(rows, expected_interval_seconds):
    if not 1<=expected_interval_seconds<=86400: raise ValueError('invalid expected interval')
    if len({(r.location,r.source,r.provenance) for r in rows})>1:
        raise ValueError('summaries require one location, source and provenance')
    rows=sorted(rows,key=lambda r:r.observed_at)
    gaps=[]
    for a,b in zip(rows,rows[1:]):
        elapsed=(utc(b.observed_at)-utc(a.observed_at)).total_seconds()
        missing=max(0,math.floor(elapsed/expected_interval_seconds)-1)
        if missing: gaps.append(dict(after=a.observed_at,before=b.observed_at,missing_intervals=missing))
    return dict(count=len(rows),observed_change_c=(rows[-1].temperature_c-rows[0].temperature_c) if len(rows)>1 else None,
                missing_intervals=sum(g['missing_intervals'] for g in gaps),gaps=gaps,
                coverage_scope='estimated cadence gaps between observed endpoints only; no inferred values or plant behavior')
