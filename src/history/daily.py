"""Deterministic local-day projection. Never writes raw history or serving data."""
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from .observations import identifier, utc

ALGORITHM_VERSION='daily-nearest-local-target-v1'


def project_daily(rows, *, location, source, timezone_name, start, end,
                  provenance='live', target_hour=21, window_seconds=3600):
    """Select one actual observation per local day; report every absent day.

    Scope is one source/location/provenance, not an implicit sensor-reference merge.
    Tie: earlier measurement UTC time, then stable ID. Receipt time never selects.
    """
    identifier(location);identifier(source)
    try:
        zone=ZoneInfo(timezone_name);first=date.fromisoformat(start);last=date.fromisoformat(end)
    except (ValueError,TypeError,ZoneInfoNotFoundError) as e:raise ValueError('invalid date or timezone') from e
    if first>last or (last-first).days>365:raise ValueError('projection range must be 1 to 366 local dates')
    if type(target_hour) is not int or not 0<=target_hour<=23:raise ValueError('invalid target hour')
    if type(window_seconds) is not int or not 0<=window_seconds<=43200:raise ValueError('invalid selection window')
    if provenance not in ('live','backfill'):raise ValueError('invalid provenance')
    scoped={}
    for row in rows:
        if (row.location,row.source,row.provenance)!=(location,source,provenance):continue
        if row.id in scoped and not row.same_measurement(scoped[row.id]):raise ValueError('conflicting observation identity')
        prior=scoped.get(row.id)
        if prior is None or (row.received_at,row.original_observed_at)<(prior.received_at,prior.original_observed_at):
            scoped[row.id]=row
    buckets={}
    for row in scoped.values():
        local=utc(row.observed_at).astimezone(zone)
        if first<=local.date()<=last:buckets.setdefault(local.date(),[]).append(row)
    days=[];day=first
    while day<=last:
        naive=datetime.combine(day,time(target_hour))
        a=naive.replace(tzinfo=zone,fold=0);b=naive.replace(tzinfo=zone,fold=1)
        valid=[t for t in (a,b) if t.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)==naive]
        result=dict(date=day.isoformat(),status='missing',temperature_c=None,observation_id=None)
        if not valid:result['status']='nonexistent_target_time'
        elif a.utcoffset()!=b.utcoffset():result['status']='ambiguous_target_time'
        else:
            target=a.astimezone(timezone.utc)
            candidates=[]
            for row in buckets.get(day,[]):
                observed=utc(row.observed_at);offset=abs((observed-target).total_seconds())
                if offset<=window_seconds:candidates.append((offset,observed,row.id,row))
            if candidates:
                offset,observed,_,row=min(candidates,key=lambda item:item[:3])
                result.update(status='observed',temperature_c=row.temperature_c,humidity_pct=row.humidity_pct,
                              observation_id=row.id,observed_at=row.observed_at,original_observed_at=row.original_observed_at,
                              received_at=row.received_at,local_observed_at=observed.astimezone(zone).isoformat(),
                              offset_seconds=offset,reference=row.reference)
        days.append(result);day+=timedelta(days=1)
    # Hash exactly the scoped originals, including receipt/reference, so inputs are recoverable.
    payloads=sorted(row.payload() for row in scoped.values())
    input_hash=hashlib.sha256(('\n'.join(payloads)).encode()).hexdigest()
    return dict(algorithm=ALGORITHM_VERSION,input_sha256=input_hash,location=location,source=source,
                provenance=provenance,timezone=timezone_name,target_hour=target_hour,
                window_seconds=window_seconds,days=days,
                observed_days=sum(d['status']=='observed' for d in days),
                missing_days=sum(d['status']=='missing' for d in days))


def project_history(store, *, location, source, timezone_name, start, end,
                    provenance='live', target_hour=21, window_seconds=3600):
    """Read bounded UTC partitions then project local days without cross-date loss.

    At most 29 local dates per call leaves one UTC padding day at each end
    within the existing 31-date read cap. Larger histories are explicit batches.
    """
    # Validate the complete projection contract before accessing a database.
    project_daily([],location=location,source=source,timezone_name=timezone_name,start=start,end=end,
                  provenance=provenance,target_hour=target_hour,window_seconds=window_seconds)
    first=date.fromisoformat(start);last=date.fromisoformat(end)
    if (last-first).days>28:raise ValueError('stored projection is limited to 29 local dates per batch')
    rows=store.read(location=location,source=source,provenance=provenance,
                    start=(first-timedelta(days=1)).isoformat(),end=(last+timedelta(days=1)).isoformat())
    return project_daily(rows,location=location,source=source,timezone_name=timezone_name,start=start,end=end,
                         provenance=provenance,target_hour=target_hour,window_seconds=window_seconds)
