import time
import pandas as pd
from src.quality.checks import find_gaps


def test_real_gap_scan_preserves_dates_and_has_linear_vectorized_budget():
    dates = pd.date_range('1900-01-01', periods=60000, freq='D').delete([13, 14, 8000, 40001])
    start = time.perf_counter()
    result = find_gaps(dates)
    elapsed = time.perf_counter() - start
    assert result == [
        {'after': '1900-01-13', 'missing_days': 2},
        {'after': '1921-11-26', 'missing_days': 1},
        {'after': '2009-07-08', 'missing_days': 1},
    ]
    assert elapsed < .06, elapsed


def test_gap_scan_matches_original_with_missing_unsorted_and_subdaily_dates():
    import numpy as np
    def original(dates, limit):
        d = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
        return [{'after': prev.date().isoformat(), 'missing_days': int((cur-prev).days-1)}
                for prev, cur in zip(d[:-1],d[1:]) if (cur-prev).days-1 >= limit]
    rng = np.random.default_rng(20261025)
    for size in [0,1,2,10,100,365]:
        dates = pd.to_datetime(rng.integers(0,3000,size), unit='h', origin='2025-01-01').tolist()
        for limit in [0,1,2,7]:
            assert find_gaps(dates,limit) == original(dates,limit)
    dates = ['2025-01-01', None, '2025-01-10']
    assert find_gaps(dates) == original(dates,1)


def test_actual_health_summary_keeps_gap_output(monkeypatch):
    from src.quality import checks
    dates = pd.date_range('2025-01-01', periods=365).delete([12, 13, 100])
    df = pd.DataFrame({'date': dates, 'temp_c': [20 + i % 7 for i in range(len(dates))]})
    result = checks.summarize(df, today=pd.Timestamp('2026-01-01'))
    assert result['rows'] == 362
    assert result['gaps'] == [
        {'after': '2025-01-12', 'missing_days': 2},
        {'after': '2025-04-10', 'missing_days': 1},
    ]
