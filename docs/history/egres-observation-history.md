# Raw observation history with Egres

Status: implemented and tested locally, not deployed or collecting. The combined change pins pandas to 2.3.3 and passes the complete Python suite with real Egres: 369 tests passed, zero skipped. Existing daily SQLite forecasts, statistics, sensor endpoints and UI remain unchanged. Publication and operational activation are separate steps.

## Purpose and data contract

Preserve each incoming temperature/humidity observation instead of reducing cloud uploads to the existing daily comparison window. This independent append-only workload uses only Egres's supported equality, insert, primary-key and literal LIMIT SQL. It does not replace existing SQLite queries, which need ranges, sorting, joins, aggregates and upserts.

Stored fields are location, source, original observation timestamp, normalized UTC timestamp, receipt timestamp, temperature, optional humidity, provenance (live/backfill), and backfill source reference. Identity is location + source + normalized observation time. Identical replay is acknowledged without another record. Conflicting values or provenance at that identity return 409, never overwrite history. A timezone-equivalent replay retains the first original timestamp and receipt.

`live` identifies collection provenance, not proof of hardware authenticity or recent arrival. Historical reference values must not be relabelled as sensor measurements. Backfill requires a source reference, but the software cannot independently verify a claimed reference. No past records are invented or automatically imported.

## API

- POST `/api/observations/readings`: up to 1000 validated observations.
- GET `/api/observations/readings`: location, source, start/end UTC dates, provenance and expected interval seconds.
- Both require `Authorization: Bearer <operator token>` and fail closed with 503 unless `THERMOSENSE_HISTORY_DSN` and `THERMOSENSE_HISTORY_TOKEN` are configured. Never commit these secrets.
- Install optional dependencies with `pip install '.[history]'`. Provision schema explicitly using `EgresHistory(dsn).initialize()` before serving. Requests do not create schema.

GET covers at most 31 UTC dates, caps results at 10000 records and fails explicitly if a partition/result exceeds its cap. It never silently truncates a trend. Day partitions are equality-filtered in Egres and timestamps sorted in application code. There is no automatic deletion, downsampling or TTL.

The summary reports observed endpoint temperature change and cadence-based missing-interval estimates between first and last observations. It does not infer values, fill gaps, estimate missing data outside those endpoints or assert plant operating behavior. Slight cadence jitter below a whole missing interval is not counted. Actual-vs-forecast or plant-behavior analysis is not implemented; its intended meaning remains to be defined.

Batch validation precedes writes, but writes are not atomic across the batch. A failure can leave a prefix stored; retry safely deduplicates that prefix. Never mark a failed batch synced. The acknowledgement states this explicitly.

## Pi uploader

`scripts/upload_observation_history.py` reads the existing Pi SQLite database read-only, independently of legacy synced flags. Supply database path, authenticated endpoint, location, optional limit and `--after-id`. Save the returned last_id only after a fully acknowledged batch; retrying the same range is idempotent. A conflict or partial/network failure must not advance the checkpoint. The script is not scheduled automatically and does not silently backfill a historical backlog.

Offset-less timestamps require explicit `--source-timezone` from the true device configuration. Ambiguous/nonexistent local times are rejected, never guessed. HTTPS is required outside loopback. Redirects are not followed with the bearer token. The legacy daily uploader is unchanged.

## Deployment boundary

This Egres release does not implement authentication or TLS. Bind it to loopback/private networking only. A remotely reachable application endpoint needs HTTPS, its history token and protected private database transport. A DSN is configuration, not durable hosting. A Vercel function must not be treated as a permanent database host.

No hosting, production credentials, capacity policy, backups, restore drill or collection schedule has been configured by this change. These must be established before activation. No production migration or actual collection has happened. Clean process restart coverage is not a power-loss/crash, disk-full or backup-recovery guarantee.

## Verification and reproducibility

Use a disposable sample-only official Egres v0.2610.0 Linux x86_64 binary verified against publisher SHA256SUMS. Set `THERMOSENSE_EGRES_BINARY` to its path and run:

```
python scripts/history/qualify.py
```

The qualifier starts fresh temporary sample storage on loopback ports, runs the entire Python suite with real Egres, stops/restarts the database and checks retention of the original observation payload. It preserves a failing suite's nonzero exit code, cleans temporary storage and writes a JUnit report to a temporary path (override with `THERMOSENSE_HISTORY_TEST_REPORT`). It never uses a production DSN.

Tests cover validation, UTC date boundaries, source and live/backfill separation, missing intervals/jitter, invalid batches, storage failures, malformed/non-ASCII authentication, out-of-order writes, duplicates/concurrent replay/conflicts, capped reads, HTTP-to-real-database behavior, 80 seeded observations compared with SQLite, uploader acknowledgements/redirect safety and clean-restart retention.

Executed October 28, 2026 on Python 3.12.15: 369 Python tests passed, zero skipped, qualifier exit 0; restart retained the original source/timestamps/value payload. Existing frontend: four suites, 30 tests passed; production build succeeded. Warnings remain for existing deprecations/model compatibility/numeric precision. Passing tests are not a guarantee of zero errors.

Pandas is pinned to exact 2.3.3 in requirements.txt, requirements-ci.txt and pyproject.toml. The preference-preserving regenerated requirements-ci.lock changes only pandas 3.1.0 to 2.3.3 and adds its required pytz 2026.5. This restores saved-ensemble metric reproducibility without altering models, source data, expected scores or test tolerances.

### Forecast quality and timing limits

Saved-model evaluation on the fixed final 30-day holdout produced identical inputs/features, predictions, target dates and per-horizon metrics across main/pandas3.1.0, main/pandas2.3.3 and history/pandas2.3.3. Day1: 29 pairs, MAE 0.986048 C, RMSE 1.195498 C. Day2: 28 pairs, MAE 1.080710 C, RMSE 1.227909 C. Day3: 27 pairs, MAE 1.039034 C, RMSE 1.204115 C. No models were retrained for this comparison. Original training pandas provenance is not recorded in the inspected registry; matching metrics do not establish it.

A separate offline serving input shows day2 prediction changing from 22.17 C with pandas3.1.0 to 22.13 C with pandas2.3.3, with day1/day3 unchanged. Therefore the pin is not universally output-equivalent and neither better nor worse future forecast accuracy has been proved. Fixed-holdout parity does not erase that separate input delta.

Offline HTTP/model forecast timing used five warmups and 35 timed requests per process. Six alternating process pairs comparing old dependency environment/main with combined code gave median-of-process-medians 25.555 ms versus 23.639 ms. Four alternating pairs at identical pandas2.3.3 gave main 24.910 ms versus history 24.236 ms and identical response payloads except generation time. There was no measured slowdown in these samples. This is not an Egres speed-up claim or a broad performance guarantee: history collection was inactive on the existing forecast path.

## Sources

- https://github.com/manishrjain/egres
- https://egresql.io/docs
- https://github.com/manishrjain/egres/releases/download/v0.2610.0/SHA256SUMS
- https://raw.githubusercontent.com/charan-rathore/Time-Series-Temperature-Modelling/27cb2034c75635af6153aacc8be717c7fa9cf788/src/data/baseline_collector.py
- https://raw.githubusercontent.com/charan-rathore/Time-Series-Temperature-Modelling/27cb2034c75635af6153aacc8be717c7fa9cf788/src/api/routes/locations.py
