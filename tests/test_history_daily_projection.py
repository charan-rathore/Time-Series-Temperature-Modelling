from datetime import date
import pytest
from src.history.observations import Observation
from src.history.daily import project_daily

def reading(t,temp=25.,source='sensor',provenance='live'):
    return Observation.create(location='site',source=source,observed_at=t,temperature_c=temp,
                              provenance=provenance,reference='archive:record' if provenance=='backfill' else None)

def project(rows,**kw):
    return project_daily(rows,location='site',source='sensor',timezone_name='Asia/Kolkata',start='2026-10-26',end='2026-10-28',**kw)

def test_local_day_and_closest_original_timestamp():
    a=reading('2026-10-27T15:20:00Z',20.);b=reading('2026-10-27T15:35:00Z',26.)
    out=project([b,a]);day=out['days'][1]
    assert day['status']=='observed' and day['temperature_c']==26.
    assert day['observation_id']==b.id and day['offset_seconds']==300
    assert day['observed_at']==b.observed_at and day['received_at']==b.received_at
    assert day['local_observed_at'].startswith('2026-10-27T21:05:00')

def test_missing_is_not_forward_filled_and_no_cross_midnight_guess():
    out=project([reading('2026-10-26T20:00:00Z')]) # local Oct27 01:30, outside window
    assert all(x['status']=='missing' and x['temperature_c'] is None for x in out['days'])

def test_tie_is_earlier_measurement_not_input_order():
    a=reading('2026-10-27T15:25:00Z',20.);b=reading('2026-10-27T15:35:00Z',30.)
    assert project([a,b])==project([b,a])
    assert project([b,a])['days'][1]['observation_id']==a.id

def test_source_and_backfill_not_mixed():
    rows=[reading('2026-10-27T15:30:00Z',99.,source='reference'),reading('2026-10-27T15:30:00Z',40.,provenance='backfill')]
    assert project(rows)['days'][1]['status']=='missing'
    assert project(rows,provenance='backfill')['days'][1]['temperature_c']==40.

def test_duplicate_replay_and_conflicting_identity():
    a=reading('2026-10-27T15:30:00Z')
    assert project([a,a])==project([a])
    with pytest.raises(ValueError,match='conflict'):project([a,reading('2026-10-27T15:30:00Z',99.)])

@pytest.mark.parametrize('kw',[dict(start='2026-10-29',end='2026-10-28'),dict(start='bad',end='2026-10-28'),dict(timezone_name='No/Zone'),dict(target_hour=25),dict(window_seconds=-1)])
def test_invalid_contract(kw):
    args=dict(location='site',source='sensor',timezone_name='Asia/Kolkata',start='2026-10-26',end='2026-10-28');args.update(kw)
    with pytest.raises(ValueError):project_daily([],**args)

def test_dst_ambiguous_and_nonexistent_target_not_guessed():
    a=project_daily([],location='site',source='sensor',timezone_name='America/New_York',start='2026-11-01',end='2026-11-01',target_hour=1)
    assert a['days'][0]['status']=='ambiguous_target_time'
    b=project_daily([],location='site',source='sensor',timezone_name='America/New_York',start='2026-03-08',end='2026-03-08',target_hour=2)
    assert b['days'][0]['status']=='nonexistent_target_time'

def test_bounded_storage_reads_neighbor_utc_dates():
    from src.history.daily import project_history
    from unittest.mock import Mock
    store=Mock();store.read.return_value=[reading('2026-10-27T15:30:00Z')]
    out=project_history(store,location='site',source='sensor',timezone_name='Asia/Kolkata',start='2026-10-27',end='2026-10-27')
    assert out['observed_days']==1
    assert store.read.call_args.kwargs['start']=='2026-10-26'
    assert store.read.call_args.kwargs['end']=='2026-10-28'
    with pytest.raises(ValueError):project_history(store,location='site',source='sensor',timezone_name='Asia/Kolkata',start='2026-10-01',end='2026-10-30')
    assert store.read.call_count==1


def test_real_egres_projection_matches_original_rows():
    import os,uuid
    from src.history.daily import project_history
    from src.history.egres import EgresHistory
    dsn=os.getenv('THERMOSENSE_TEST_EGRES_DSN')
    if not dsn:pytest.skip('requires isolated real Egres')
    store=EgresHistory(dsn);store.initialize();location='projection-'+uuid.uuid4().hex
    rows=[Observation.create(location=location,source='sensor',observed_at=t,temperature_c=temp)
          for t,temp in [('2026-10-27T15:10:00Z',24.),('2026-10-27T15:35:00Z',26.)]]
    for row in reversed(rows):store.append(row)
    actual=project_history(store,location=location,source='sensor',timezone_name='Asia/Kolkata',start='2026-10-26',end='2026-10-28')
    expected=project_daily(rows,location=location,source='sensor',timezone_name='Asia/Kolkata',start='2026-10-26',end='2026-10-28')
    assert actual==expected and actual['observed_days']==1 and actual['missing_days']==2

def test_equivalent_replay_receipts_have_deterministic_projection():
    from dataclasses import replace
    a=reading('2026-10-27T15:30:00Z')
    b=replace(a,received_at='2026-10-28T20:00:00.000000Z')
    assert project([a,b])==project([b,a])
