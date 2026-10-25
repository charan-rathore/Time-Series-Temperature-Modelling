# Draft: vectorize calendar-gap scanning

`find_gaps` created Python Timestamp objects and subtracted pairs for every observed day. Compute the same adjacent timedelta day counts with pandas, then create dictionaries only for qualifying gaps. Sorting, thresholds, dates and response shape are unchanged.

## Executed checks

- Actual 60,000-date helper, source-only revert: 60 ms budget fails at 142.90 ms; candidate passes.
- Seven-trial median: 120.52 ms baseline to 3.34 ms candidate.
- Ordinary 365-day input, 100 calls: 94.20 ms to 33.66 ms.
- Seeded differential checks across empty, singleton, unsorted, duplicate, subdaily and missing dates, with four gap thresholds, agree with the exact original algorithm.
- Actual `summarize` health path preserves expected row count and precise gaps for a 365-day frame.
- Final focused suite: 15 PASS. Full locked suite before the last health-path control: 335 PASS / 1 known saved-ensemble freshness FAIL. Not merge-ready.

Local measurements, not production endpoint latency. No data repair, evaluation, model, schema, dependency or uploader changes. Full-green merge gate remains blocked.
