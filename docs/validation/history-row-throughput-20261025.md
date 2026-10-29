# Draft: avoid per-record pandas Series in history responses

The history handler created a Series for every filtered CSV row. Use pandas record dictionaries instead. Field lookup, rounding, null handling, sensor flags, filtering and response models are unchanged.

## Executed evidence

- Actual history handler reads a 10,000-row CSV and builds every response model. Source-only baseline fails the 350 ms budget at 450.85 ms; final repeat 422.44 ms. Candidate passes.
- Seven-trial median, 10,000 rows: 400.41 ms to 104.39 ms.
- Ordinary 365-row history, ten calls: 160.08 ms to 42.76 ms.
- Complete response JSON is byte-identical between original and candidate at both sizes. SHA-256: 365 rows `a6f741dbe54a2c8e7a6f6073cbfcacd3ee80efea70db55d2c11d15326215c253`; 10,000 rows `1b2a583138102c8c429cc8393334b5669ae3b5204b725784bae1e74f9f59ad43`.
- Actual FastAPI TestClient route with real CSV covers precise values, rounding, missing metrics, boolean flags and full response structure. It passes on original and candidate.
- Final focused suite: 2 PASS. Locked full suite: 335 PASS / 1 known saved-ensemble freshness FAIL. Not merge-ready.

These are local handler measurements, not production network latency. No API response, schema, model, dependency, evaluation or uploader change. Full-green merge gate remains blocked.
