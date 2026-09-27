import pandas as pd
import pytest
from scripts.audit_observations import compare


def test_untouched_comparison_never_calibrates_on_reporting_days():
    days = pd.date_range('2024-06-01', periods=20)
    archive = pd.DataFrame({'date': days, 'temp_c': [20.0] * 20})
    legacy = pd.DataFrame({'date': days, 'temp_c': [22.0] * 14 + [25.0] * 6})
    result = compare(archive, legacy)
    assert result['calibration_days'] == 14
    assert result['calibration_offset_c'] == 2.0
    assert result['holdout_archive_mae_c'] == 5.0
    assert result['holdout_bias_corrected_same_time_mae_c'] == 3.0
    assert 'not_forecast' in result['kind']


def test_refuses_tiny_sample():
    days = pd.date_range('2024-06-01', periods=3)
    data = pd.DataFrame({'date': days, 'temp_c': [20.0] * 3})
    with pytest.raises(ValueError):
        compare(data, data)


def test_walkforward_forecast_never_sees_target():
    from scripts.backtest_legacy import backtest
    dates = pd.date_range('2024-06-01', periods=30)
    original = pd.DataFrame({'date': dates, 'temp_c': [20.0] * 30})
    changed = original.copy()
    changed.loc[29, 'temp_c'] = 100.0
    before = backtest(original)
    after = backtest(changed)
    # The previous target change may affect the last two origins, so compare
    # only the day-1 score, which must include a strictly prior baseline.
    assert before['horizons']['day1']['persistence_mae_c'] == 0
    assert after['horizons']['day1']['persistence_mae_c'] > 0
    assert after['horizons']['day1']['untouched_targets'] == 10


def test_previous_runs_uses_exact_local_9pm_and_fixed_leads():
    from scripts.backtest_open_meteo_previous_runs import compare
    times = [f'2024-06-{i:02d}T{h:02d}:00' for i in range(1,22) for h in (20,21)]
    payload={'hourly':{'time':times,
              **{f'temperature_2m_previous_day{h}':[99 if t.endswith('20:00') else 25+h for t in times] for h in (1,2,3)}}}
    obs=pd.DataFrame({'date':pd.date_range('2024-06-01',periods=21),'temp_c':[25]*21})
    result=compare(payload,obs)
    assert result['n_paired_days']==21
    assert result['models']['open_meteo_1day']['mae_c']==1
    assert result['models']['open_meteo_3day']['mae_c']==3


def test_bias_correction_does_not_consume_issue_date_or_target():
    from scripts.evaluate_bias_correction import evaluate
    dates=pd.date_range('2024-06-01',periods=30)
    payload={'hourly':{'time':[f'{d.date()}T21:00' for d in dates],
           **{f'temperature_2m_previous_day{h}':[20.0]*30 for h in (1,2,3)}}}
    actual=pd.DataFrame({'date':dates,'temp_c':[22.0]*30})
    before=evaluate(payload,actual,warmup=10,window=7)
    assert before['horizons']['day1']['corrected_mae_c']==0
    changed=actual.copy();changed.loc[28,'temp_c']=100.0
    after=evaluate(payload,changed,warmup=10,window=7)
    # At day 30 the 24h old forecast must not have seen day 29's observed value.
    # Its early impact is confined to truth on day 29, not the day-30 offset.
    assert after['horizons']['day1']['corrected_mae_c']>0
    assert after['horizons']['day1']['n_matched']==20
