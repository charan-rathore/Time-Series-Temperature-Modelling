# Raw-history daily projection change record

## Starting state

Public base: `27cb2034c75635af6153aacc8be717c7fa9cf788`. Dependency: separately reviewed Egres history packet SHA256 `8af80ee972e7207bc4cc7253ccb44ce747f27e620125ce490e83f53a3edf54a1`. This increment does not require the optional uploader-checkpoint extension. It is not a replacement for the approved history packet.

Architecture diagrams distinguish current storage from proposed history-to-daily-view-to-ledger flow. They were delivered before this stage's implementation. This change implements only the local-day view, not ledger, plant insights, deployment or active collection.

## What changes

`src/history/daily.py` projects immutable raw observations into an explicit local-date range. It selects the closest measurement to local 21:00 within one hour, within the same local date; ties choose earlier UTC measurement then stable ID. One location, source and live/backfill provenance at a time. Missing days stay absent-valued; no filling or mixing reference readings into sensor ground truth. Original measurement time and receipt time remain distinct. Conflicting identities fail. DST ambiguous/nonexistent target times are flagged, not guessed. Equivalent replay receipts select deterministic earliest receipt.

Each result records algorithm version and SHA256 over scoped original payloads. `project_history` reads adjacent UTC partitions to avoid local-midnight loss, capped to 29 local dates per batch within the database adapter's existing31date cap. Over-cap reads fail explicitly. Read-only CLI: `scripts/history/project_daily.py`. No API registration, forecasting selection, model training, daily SQLite writes or existing UI changes.

## Executed tests and benchmarks

Python3.12.15, exact declared pandas2.3.3 lock plus official Egresv0.2610.0: full383tests passed, zero skipped; qualifier exit0; clean restart retained original observation payload. New14projection tests include original timestamp/local-date selection, missing/no-fill, stable ties/replays, source/backfill separation, conflict rejection, invalid bounds, DST transitions, storage date padding and actual Egres parity. Initial missing-module RED retained, final GREEN uses new code. No test expectation/tolerance/model/source-data change. Tracked feature parquet regenerated during tests was restored.

Synthetic projection benchmark:7832rows, fixed seed28102026,12counterbalanced process-local paired measurements against an independent brute-force local-day reference. Selected observation IDs exactly match. Median reference187.395ms; candidate76.041ms. This measures new projection code only; the reference is not previous production behavior and these numbers are not a forecast/Egres speedup. Input SHA and raw samples accompany review.

Existing offline actual HTTP/model forecast check, same pandas environment,4alternating process pairs,35timed requests after5warmups per process: baseline median-of-process-medians25.751ms, candidate26.001ms (+0.97%). Individual process distributions overlap. All payloads identical except generation time. This is inconclusive host-level timing noise, not a formal no-slowdown guarantee. No forecast path imports/calls this module. Independent publication review must keep that limit, rather than count functional passes as a performance proof. Frontend untouched; prior four suites/30tests and build evidence applies.

## Restore and data boundaries

Before any publication, record the eventual candidate commit, parent commit and receipt hashes here. Until committed, the exact reviewed patch/hash in the delivery manifest identifies this increment; no pushed commit is claimed.

Safe code rollback after publication is a new revert commit for this specific increment, retaining the separately reviewed history dependency. Do not reset/force-push shared history. The known public starting point is the base above, but it predates the new history feature and is not a claimed equivalent operational fallback. Do not remove raw observation data or revert its database as part of code rollback. This projection is read-only and creates no database migration. Database backup/restore requires an independently verified backup and explicit operational procedure; a git commit is not a data backup.

Keep baseline/candidate runtime, model/input hashes, raw quality/timing samples and test receipts with the release record. If measured quality or latency degrades beyond an agreed budget, stop promotion and revert the isolated code change only after verifying its exact dependencies. No automatic production rollback or new hosting permission is granted by this document.

## Remaining boundaries

Durable private Egres host, transport/authentication, storage capacity, backups/restore and actual collection remain unconfigured. No energy savings, plant behavior, future accuracy, live sensor availability or comprehensive no-regression guarantee is asserted. Daily projection does not itself improve the current forecasts until a separately tested serving/evaluation integration is made. Forecast ledger and source-aware ground-truth evaluation remain next architecture stages, not silently included here.
