#!/usr/bin/env python3
"""Walk-forward fixed-lead bias correction against unverified legacy observations.

At target date t for horizon h, offset uses only verified past comparison
rows with date < t-h, strictly before the nominal 24h-multiple issue time.
This is an experimental baseline, not a verified sensor-backed production model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.backtest_open_meteo_previous_runs import URL
from src.data.preprocess import load_legacy_csv


def evaluate(payload: dict, legacy: pd.DataFrame, warmup: int = 10, window: int = 7) -> dict:
    payload = payload['response'] if 'response' in payload else payload
    hourly = payload['hourly']
    times = pd.to_datetime(hourly['time'])
    mask = times.hour == 21
    frame = pd.DataFrame({'date': times[mask].normalize()})
    obs = legacy[['date', 'temp_c']].rename(columns={'temp_c': 'actual_c'})
    for h in (1,2,3):
        frame[f'forecast_day{h}'] = np.asarray(hourly[f'temperature_2m_previous_day{h}'],dtype=float)[mask]
    paired = frame.merge(obs,on='date',how='inner').sort_values('date').reset_index(drop=True)
    if len(paired)<warmup+10: raise ValueError('Need 10 or more untouched test days')
    result={'kind':'experimental_causal_bias_correction_on_unverified_legacy',
            'forecast_source':URL,
            'requested_coordinates':[12.9716,77.5946],
            'returned_grid_coordinates':[payload.get('latitude'),payload.get('longitude')],
            'returned_grid_elevation_m':payload.get('elevation'),
            'legacy_source':'repo data/legacy/temperature-data-for-TSA.csv',
            'window':window,'warmup':warmup,'horizons':{},
            'warning':'Retrospective fixed-lead Open-Meteo forecasts and repo legacy values, not independently verified DHT22 readings. Valid-time offset assumes the legacy date refers to 21:00 local and the previous-day forecast issue timing is no later than that date. Not proof of superiority on a live site or against other vendors.'}
    for h in (1,2,3):
        raw,corrected,actual,dates=[],[],[],[]
        key=f'forecast_day{h}'
        for i in range(warmup,len(paired)):
            issue_date=paired.date.iloc[i]-pd.Timedelta(days=h)
            previous=paired.loc[paired.date < issue_date].dropna(subset=[key,'actual_c'])
            if len(previous)<window or pd.isna(paired[key].iloc[i]): continue
            bias=(previous.actual_c-previous[key]).iloc[-window:].mean()
            raw.append(float(paired[key].iloc[i]))
            corrected.append(float(paired[key].iloc[i]+bias))
            actual.append(float(paired.actual_c.iloc[i]))
            dates.append(str(paired.date.iloc[i].date()))
        if not dates: continue
        actual=np.asarray(actual);raw=np.asarray(raw);corrected=np.asarray(corrected)
        result['horizons'][f'day{h}']={
          'n_matched':len(dates),'first_target':dates[0],'last_target':dates[-1],
          'raw_mae_c':round(float(np.mean(np.abs(actual-raw))),3),
          'corrected_mae_c':round(float(np.mean(np.abs(actual-corrected))),3),
          'raw_rmse_c':round(float(np.sqrt(np.mean((actual-raw)**2))),3),
          'corrected_rmse_c':round(float(np.sqrt(np.mean((actual-corrected)**2))),3),
          'mean_adjustment_c':round(float(np.mean(corrected-raw)),3),
        }
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-raw',type=Path)
    args=parser.parse_args()
    if args.offline_raw:
        payload=json.loads(args.offline_raw.read_text())
    else:
        import requests
        params={'latitude':12.9716,'longitude':77.5946,
                'hourly':','.join(f'temperature_2m_previous_day{h}' for h in (1,2,3)),
                'start_date':'2024-06-02','end_date':'2024-07-11','timezone':'Asia/Kolkata'}
        response=requests.get(URL,params=params,timeout=40)
        response.raise_for_status()
        payload=response.json()
    print(json.dumps(evaluate(payload,load_legacy_csv()),indent=2))
