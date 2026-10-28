#!/usr/bin/env python3
"""Explicit opt-in raw Pi history upload. Does not modify legacy synced flags.

Requires a configured authenticated history endpoint. Local source timestamps
must include timezone, or --source-timezone must name the sensor's true zone.
A failed/partial batch is safely replayed; the server preserves immutable keys.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import requests


def upload(db_path, endpoint, token, location, source_timezone=None, limit=1000, after_id=0):
    if not token: raise ValueError('THERMOSENSE_HISTORY_TOKEN is required')
    parsed=urlparse(endpoint)
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost','::1')):
        raise ValueError('history endpoint must use HTTPS except loopback tests')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('endpoint cannot include credentials, query or fragment')
    if not 1<=limit<=1000 or after_id<0:raise ValueError('invalid batch bounds')
    # Read-only URI: do not accidentally create a missing source database.
    with sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        rows=c.execute('SELECT id,timestamp,temp_c,humidity_pct,source FROM sensor_readings WHERE id > ? ORDER BY id LIMIT ?',
                       (after_id,limit)).fetchall()
    payload=[]
    for r in rows:
        t=datetime.fromisoformat(r['timestamp'].replace('Z','+00:00'))
        if t.tzinfo is None:
            if not source_timezone:raise ValueError('naive sensor timestamps require explicit --source-timezone')
            zone=ZoneInfo(source_timezone)
            a=t.replace(tzinfo=zone,fold=0);b=t.replace(tzinfo=zone,fold=1)
            if a.utcoffset()!=b.utcoffset():raise ValueError('ambiguous/nonexistent local sensor timestamp')
            t=a
        payload.append(dict(location=location,source=r['source'],observed_at=t.isoformat(),
                            temperature_c=r['temp_c'],humidity_pct=r['humidity_pct'],provenance='live'))
    if not payload:return dict(acknowledged=0,last_id=after_id)
    response=requests.post(endpoint,json={'readings':payload},headers={'Authorization':'Bearer '+token},
                           timeout=30,allow_redirects=False)
    if response.status_code!=200:raise ValueError('history upload not acknowledged; retry unchanged batch')
    result=response.json()
    if not isinstance(result,dict) or type(result.get('stored')) is not int or type(result.get('duplicates')) is not int:
        raise ValueError('invalid history acknowledgement')
    if result['stored']<0 or result['duplicates']<0 or result['stored']+result['duplicates']!=len(rows):
        raise ValueError('partial history acknowledgement; retry unchanged batch')
    return dict(acknowledged=len(rows),last_id=rows[-1]['id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--location',required=True)
    p.add_argument('--source-timezone');p.add_argument('--after-id',type=int,default=0);p.add_argument('--limit',type=int,default=1000)
    a=p.parse_args()
    print(upload(a.db,a.endpoint,os.getenv('THERMOSENSE_HISTORY_TOKEN'),a.location,a.source_timezone,a.limit,a.after_id))

if __name__=='__main__':main()
