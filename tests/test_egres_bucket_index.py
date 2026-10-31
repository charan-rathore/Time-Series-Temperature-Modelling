"""Bucket secondary index: reads filter on bucket, so it must be indexed.

Without the index Egres scans the full table per day-partition query
(measured 17 ms median at 240k rows vs 0.7 ms with the index, v0.2610.0).
Set THERMOSENSE_TEST_EGRES_DSN to a disposable loopback DB only.
"""
import os
import psycopg
import pytest
from src.history.egres import EgresHistory

DSN = os.getenv('THERMOSENSE_TEST_EGRES_DSN')

@pytest.mark.skipif(not DSN, reason='requires isolated real Egres instance')
def test_initialize_creates_bucket_index():
    db = EgresHistory(DSN)
    db.initialize()
    # Egres reports an existing same-name index as 42P07 (index already exists):
    # a failing duplicate create proves initialize() created the index.
    with db.connect() as c, pytest.raises(psycopg.errors.DuplicateTable):
        c.execute('CREATE INDEX raw_observations_bucket ON raw_observations (bucket)')
    # initialize() stays idempotent with the index present.
    db.initialize()
