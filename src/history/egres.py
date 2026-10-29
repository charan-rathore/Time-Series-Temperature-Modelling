"""Text-only pgwire adapter for Egres v0.2610.0's equality SQL subset.

No shared connection or exposed credentials. The operator must provide private
network transport: this Egres release does not authenticate or terminate TLS.
"""
import json
from datetime import date, timedelta
from .observations import Observation, partition_key

class HistoryConflict(ValueError): pass
class HistoryLimit(ValueError): pass

class EgresHistory:
    def __init__(self, dsn):
        if not dsn: raise ValueError('history DSN is required')
        self.dsn=dsn

    def connect(self):
        import psycopg
        return psycopg.connect(self.dsn, autocommit=True, connect_timeout=3,
                               options='', prepare_threshold=None)

    def initialize(self):
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS raw_observations (id TEXT PRIMARY KEY, bucket TEXT, payload TEXT)')

    def append(self, observation):
        import psycopg
        with self.connect() as c:
            try:
                c.execute('INSERT INTO raw_observations (id,bucket,payload) VALUES (%s,%s,%s)',
                          (observation.id,partition_key(observation.location,observation.observed_at[:10]),observation.payload()))
                return True
            except psycopg.errors.UniqueViolation:
                row=c.execute('SELECT payload FROM raw_observations WHERE id = %s',(observation.id,)).fetchone()
                if row and Observation(**json.loads(row[0])).same_measurement(observation):return False
                raise HistoryConflict('same observation identity has different contents') from None

    def read(self, *, location, start, end, source, provenance='live', max_rows=10000):
        first=date.fromisoformat(start);last=date.fromisoformat(end)
        if first>last or (last-first).days>30:raise ValueError('range must be 1 to 31 UTC dates')
        if not 1<=max_rows<=10000:raise ValueError('invalid row cap')
        if provenance not in ('live','backfill'):raise ValueError('invalid provenance')
        from .observations import identifier
        identifier(source)
        result=[]
        with self.connect() as c:
            day=first
            while day<=last:
                # LIMIT is supported; cap each partition before decoding payloads.
                rows=c.execute('SELECT payload FROM raw_observations WHERE bucket = %s LIMIT '+str(max_rows+1),
                               (partition_key(location,day.isoformat()),)).fetchall()
                if len(rows)>max_rows:raise HistoryLimit('partition too large; narrow date range')
                for row in rows:
                    o=Observation(**json.loads(row[0]))
                    if o.source==source and o.provenance==provenance:result.append(o)
                if len(result)>max_rows:raise HistoryLimit('too many observations; narrow date range')
                day+=timedelta(days=1)
        return sorted(result,key=lambda o:o.observed_at)
