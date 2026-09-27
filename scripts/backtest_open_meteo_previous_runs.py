#!/usr/bin/env python3
"""Historical fixed-lead Open-Meteo forecast versus unverified legacy observations.

Previous Runs API predicts each valid-time temperature 24/48/72h in advance;
no same-time archive observation is used as a future forecast input.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import load_legacy_csv

URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def compare(payload: dict, legacy: pd.DataFrame) -> dict:
    hourly = payload['hourly']
    times = pd.to_datetime(hourly['time'])
    mask = times.hour == 21
    frame = pd.DataFrame({'date': times[mask].normalize()})
    for h in (1, 2, 3):
        key = f'temperature_2m_previous_day{h}'
        frame[f'forecast_day{h}'] = np.asarray(hourly[key], dtype=float)[mask]
    observations = legacy[['date', 'temp_c']].rename(columns={'temp_c': 'unverified_legacy_c'})
    paired = frame.merge(observations, on='date', how='inner').sort_values('date')
    if len(paired) < 20:
        raise ValueError('Need at least 20 paired dates')
    result = {
        'kind': 'retrospective_fixed_lead_forecast_comparison_to_unverified_legacy',
        'source': URL,
        'requested_coordinates': [12.9716, 77.5946],
        'returned_grid_coordinates': [payload.get('latitude'), payload.get('longitude')],
        'returned_grid_elevation_m': payload.get('elevation'),
        'docs': 'https://open-meteo.com/en/docs/previous-runs-api',
        'n_legacy_rows': len(legacy),
        'n_paired_days': len(paired),
        'period': [str(paired.date.iloc[0].date()), str(paired.date.iloc[-1].date())],
        'models': {},
        'warning': 'Historical forecast runs are fixed 24/48/72h lead, but the repo legacy observations lack verified coordinates, instrument, siting and timestamps. This is a retrospective vendor-only benchmark, not ThermoSense model superiority or a live sensor test.'
    }
    for h in (1,2,3):
        subset = paired[['unverified_legacy_c', f'forecast_day{h}']].dropna()
        err = subset[f'forecast_day{h}'] - subset.unverified_legacy_c
        result['models'][f'open_meteo_{h}day'] = {
            'n': len(subset), 'mae_c': round(float(np.abs(err).mean()), 3),
            'rmse_c': round(float(np.sqrt(np.mean(err**2))), 3),
            'mean_error_c': round(float(err.mean()), 3),
        }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',default='2024-06-02')
    parser.add_argument('--end',default='2024-07-11')
    parser.add_argument('--offline-raw',type=Path)
    args=parser.parse_args()
    if args.offline_raw:
        payload=json.loads(args.offline_raw.read_text())
    else:
        params={'latitude':12.9716,'longitude':77.5946,
                'hourly':','.join(f'temperature_2m_previous_day{h}' for h in (1,2,3)),
                'start_date':args.start,'end_date':args.end,'timezone':'Asia/Kolkata'}
        response=requests.get(URL,params=params,timeout=40)
        response.raise_for_status()
        payload=response.json()
        out=Path('data/raw')/f'previous_runs_{args.start}_{args.end}.json'
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({'requested_url':response.url,'response':payload},indent=2))
        print(f'Saved fixed-lead forecast response: {out}')
    if 'response' in payload: payload=payload['response']
    print(json.dumps(compare(payload,load_legacy_csv()),indent=2))

if __name__=='__main__': main()
