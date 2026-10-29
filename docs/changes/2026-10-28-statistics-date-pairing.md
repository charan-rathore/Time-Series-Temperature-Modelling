# Correct SQLite statistical date pairing

Starting public base:27cb2034c75635af6153aacc8be717c7fa9cf788. Independent of history/projection/ledger patches. Publication must check current main/PR state and apply only this increment, not copy an old tree. Existing architecture diagrams already show the separate SQLite daily comparison store; this fixes its statistical-read boundary, not a new dataflow or serving/model switch.

## Failure and change

Both statistical comparison endpoints independently read source error lists and truncated them to equal length. Sparse sources could therefore compare different dates as paired samples and generate invalid paired tests. The new SQL joins target date/horizon and the same SQLite daily actual in one statement/snapshot per baseline. Absolute errors and Diebold-Mariano signed errors come from those same aligned rows, not four separate reads. Disjoint or fewer-than-three intersections cannot yield comparison statistics. Existing missing-source404 and insufficient-pairs400 behavior is preserved. Summary source lists keep their prior independent-population semantics.

Removed duplicated absolute-error SQL; it now wraps the existing signed reader. Removed truncation and duplicate signed-error reads in comparison handlers. No schema, database path, writes, model/forecast output, external API or frontend changes. No Egres dependency. This does not certify the provenance, issue/capture time or target semantics of legacy SQLite records; it only aligns the actual stored dates. It does not create a verified prospective-accuracy or energy claim. Existing generated statistical resume statements are not independently validated here. No new missing-value filling.

## Executed evidence

Seven regression tests: disjoint dates, partial overlap with exact absolute/signed arrays at both endpoints, insufficient overlap, horizon/window/actual filtering, missing-source404, reader compatibility and no implicit missing-database creation. Three initial RED failures retained; final341PASS0FAIL0SKIP using Python3.12.15 and pandas2.3.3, full suite in isolated public-base checkout. No existing-test expectation/tolerance changes. New partial overlap expected values derive from known inserted forecasts and actuals. Tracked test-generated feature parquet restored.

Same-host alternating40paired SQLite read/transform samples on365fully-aligned dates/3sources/3horizons: old four read path2.006ms median, aligned single-query path0.831ms; arrays exactly identical on this complete fixture. This measures the scoped reader path, not whole API or production speedup. Sparse fixture output intentionally changes because false pairings are removed. Existing HTTP forecast core benchmark is separately attached; unchanged responses except generation time and overlapping host timing samples do not prove a universal non-degradation guarantee.

## Restore boundary

Record exact eventual candidate/parent commit IDs before promotion. Safe code rollback is a new revert of this isolated fix, never shared-history reset/force push. Reverting restores the known false-pairing bug, so stop statistical claims if rollback is required. The base commit is a code reference, not a database backup. No migration or existing data modification occurs. No database restore/drop or production deployment authorized by this document. Attached manifest identifies exact source and patch hashes; source hash is distinct from version labels.
