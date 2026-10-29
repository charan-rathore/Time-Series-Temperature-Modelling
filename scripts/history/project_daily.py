#!/usr/bin/env python3
"""Explicit read-only daily projection. Does not change serving or raw data.

THERMOSENSE_HISTORY_DSN must point to the operator's private Egres endpoint.
Prints a source-labelled versioned projection; does not schedule collection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.history.daily import project_history
from src.history.egres import EgresHistory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['location','source','timezone','start','end']:p.add_argument('--'+flag,required=True)
    p.add_argument('--provenance',choices=['live','backfill'],default='live')
    args=p.parse_args()
    store=EgresHistory(os.environ.get('THERMOSENSE_HISTORY_DSN'))
    result=project_history(store,location=args.location,source=args.source,timezone_name=args.timezone,
                           start=args.start,end=args.end,provenance=args.provenance)
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
