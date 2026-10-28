from datetime import datetime, timezone
import pytest
from src.history.observations import Observation, summarize, partition_key

def obs(t='2026-10-28T08:00:00Z', **kw):
    return Observation.create(location='bangalore_central', source='dht22_sensor', observed_at=t, temperature_c=25., humidity_pct=60., **kw)

def test_timestamp_normalization_and_stable_id():
    a=obs(); b=obs('2026-10-28T13:30:00+05:30')
    assert a.observed_at==b.observed_at and a.id==b.id
    assert a.original_observed_at!=b.original_observed_at
    assert a.observed_at.endswith('Z')

@pytest.mark.parametrize('kw',[{'temperature_c':float('nan')},{'temperature_c':float('inf')},{'temperature_c':True},{'humidity_pct':101},{'location':''},{'source':'unknown/source'}])
def test_invalid_fields(kw):
    args=dict(location='site',source='sensor',observed_at='2026-10-28T00:00:00Z',temperature_c=25.)
    args.update(kw)
    with pytest.raises(ValueError): Observation.create(**args)

def test_naive_time_rejected():
    with pytest.raises(ValueError):obs('2026-10-28T08:00:00')

def test_future_rejected():
    with pytest.raises(ValueError):obs('2999-01-01T00:00:00Z')

def test_backfill_needs_reference():
    with pytest.raises(ValueError):obs(provenance='backfill')
    assert obs(provenance='backfill',reference='archive:sha256:abc').provenance=='backfill'

def test_missing_intervals_no_fabricated_values():
    a=obs(); b=obs('2026-10-28T08:15:00Z')
    result=summarize([b,a],300)
    assert result['count']==2 and result['missing_intervals']==2
    assert result['observed_change_c']==0
    assert result['gaps'][0]['missing_intervals']==2

def test_empty_and_single_are_not_fake_trends():
    assert summarize([],300)['observed_change_c'] is None
    assert summarize([obs()],300)['observed_change_c'] is None

def test_separate_sources_not_combined():
    other=Observation.create(location='bangalore_central',source='reference',observed_at='2026-10-28T08:05:00Z',temperature_c=40.)
    with pytest.raises(ValueError): summarize([obs(),other],300)

def test_partition_utc_boundary():
    a=obs('2026-10-28T00:10:00+05:30')
    assert partition_key(a.location,a.observed_at[:10]).endswith('2026-10-27')

def test_minor_cadence_jitter_is_not_a_missing_reading():
    assert summarize([obs(),obs('2026-10-28T08:05:01Z')],300)['missing_intervals']==0

def test_backfill_and_live_cannot_silently_mix():
    with pytest.raises(ValueError):summarize([obs(),obs(provenance='backfill',reference='archive:hash')],300)
