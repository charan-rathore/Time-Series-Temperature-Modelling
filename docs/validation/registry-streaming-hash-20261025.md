# Draft: bound model artifact hashing memory

`file_sha256` loaded the entire model/data artifact into memory only to hash it. Stream 1 MiB blocks while preserving the SHA-256 algorithm, digest, return type and artifact identity.

No registry format, policy, promotion, evaluation or model change.

## Executed evidence

- Actual 20 MiB filesystem artifact with an independent expected hash: baseline peak 20,976,427 bytes fails the 3 MiB budget; candidate passes.
- Empty and non-block-aligned files have identical digests.
- Focused registry suite: 10 PASS.
- Locked full suite: 335 PASS, 1 known saved-ensemble freshness FAIL. Merge remains blocked until the full gate is clear.
- Separate seven-trial median for the 20 MiB hash: 16.80 ms baseline to 16.62 ms candidate. Measured peak 20,976,058 to 2,102,377 bytes.

These are local filesystem Python measurements, not production model-throughput claims. No API, network or deployment proof claimed.
