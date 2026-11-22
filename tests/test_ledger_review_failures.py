"""Review-driven failures use a separately preserved candidate."""
import os
import pytest
from src.evaluation import ledger as L


def observation(source='station'):
    return {'kind':'observation','source':source,'target':L.TARGET,'date':'2026-11-01','value':21.0,'issued_at':'2026-11-02T10:00:00Z'}


@pytest.mark.parametrize('recovery_failure',['truncate','sync'])
def test_failed_recovery_preserves_both_errors(tmp_path,monkeypatch,recovery_failure):
    path=tmp_path/'ledger.jsonl'
    L.append(observation(),path)
    original_write=L.os.write
    calls=0
    def broken_write(fd,payload):
        nonlocal calls
        calls+=1
        if calls==1:return original_write(fd,payload[:11])
        raise OSError('original write failure')
    def failed_recovery(*args):raise OSError('recovery failure')
    monkeypatch.setattr(L.os,'write',broken_write)
    monkeypatch.setattr(L.os,'ftruncate' if recovery_failure=='truncate' else 'fsync',failed_recovery)
    with pytest.raises(L.LedgerError,match='recovery could not be confirmed') as caught:
        L.append(observation('next'),path)
    assert 'original write failure' in str(caught.value.write_error)
    assert 'recovery failure' in str(caught.value.recovery_error)


def test_path_replacement_before_write_is_rejected(tmp_path,monkeypatch):
    path=tmp_path/'ledger.jsonl'
    L.append(observation(),path)
    saved=L._read_records
    def replacing_read(stream):
        records=saved(stream)
        replacement=tmp_path/'replacement'
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement,path)
        return records
    monkeypatch.setattr(L,'_read_records',replacing_read)
    with pytest.raises(L.LedgerError,match='replaced'):
        L.append(observation('next'),path)
    monkeypatch.setattr(L,'_read_records',saved)
    assert len(L.read(path))==1


@pytest.mark.parametrize('field,value',[('supersedes',[]),('date',[]),('issued_at',None),('source',{}),('hash',[])])
def test_malformed_record_is_reported_not_crashed(field,value):
    record=observation()
    record['prev']=L.GENESIS
    record[field]=value
    record['hash']=L._hash(L.GENESIS,record) if field!='hash' else value
    assert L.verify_records([record])


def test_first_append_synchronizes_parent_directory(tmp_path,monkeypatch):
    import stat
    synced=[]
    original=L.os.fsync
    def record_sync(fd):
        synced.append(stat.S_ISDIR(L.os.fstat(fd).st_mode))
        return original(fd)
    monkeypatch.setattr(L.os,'fsync',record_sync)
    L.append(observation(),tmp_path/'ledger.jsonl')
    assert synced==[False,True]


def test_failed_directory_sync_reports_uncertain_commit(tmp_path,monkeypatch):
    import stat
    original=L.os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(L.os.fstat(fd).st_mode):raise OSError('directory sync failure')
        return original(fd)
    monkeypatch.setattr(L.os,'fsync',fail_directory)
    path=tmp_path/'ledger.jsonl'
    with pytest.raises(L.LedgerCommitUncertainError,match='directory durability is unconfirmed'):
        L.append(observation(),path)
    assert len(L.read(path))==1
    assert L.verify(path)==[]


def test_verification_checks_each_record_once(monkeypatch):
    records=[]
    for i in range(1000):
        record=observation(str(i))
        record['prev']=records[-1]['hash'] if records else L.GENESIS
        record['hash']=L._hash(record['prev'],record)
        records.append(record)
    calls=0
    original=L._ValidationState.check
    def count_check(self,record):
        nonlocal calls
        calls+=1
        return original(self,record)
    monkeypatch.setattr(L._ValidationState,'check',count_check)
    assert L.verify_records(records)==[]
    assert calls==len(records)


def test_huge_integer_is_a_verifier_diagnostic():
    record=observation()
    record['value']=10**1000
    record['prev']=L.GENESIS
    record['hash']=L._hash(L.GENESIS,record)
    assert any('finite number' in error for error in L.verify_records([record]))


@pytest.mark.parametrize('outcome',['commit','recovery','rolled-back'])
def test_daily_runner_stops_on_uncertain_commit(tmp_path,monkeypatch,capsys,outcome):
    from scripts import ledger_daily as daily
    monkeypatch.setattr(daily,'verify',lambda path:[])
    monkeypatch.setattr(daily,'read',lambda path:[])
    monkeypatch.setattr(daily,'airport_observations',lambda path:[])
    monkeypatch.setattr(daily,'fetch_open_meteo',lambda get:None)
    monkeypatch.setattr(daily,'plan',lambda *args:[observation(),observation('second')])
    calls=[]
    def uncertain(record,path):
        calls.append(record)
        if outcome=='commit':
            raise L.LedgerCommitUncertainError('directory durability is unconfirmed')
        if outcome=='rolled-back':
            raise L.LedgerWriteError('rollback completed')
        raise L.LedgerRecoveryError(OSError('write'),OSError('recovery'))
    monkeypatch.setattr(daily,'append',uncertain)
    assert daily.main(['--ledger',str(tmp_path/'ledger.jsonl')])==1
    assert len(calls)==1
    assert 'refused' not in capsys.readouterr().out


def test_invalid_predecessor_does_not_become_accepted_semantic_state():
    records=[]
    for source,value,hour in [('first',21.0,10),('invalid',None,12),('third',22.0,11)]:
        record=observation(source)
        record.update(value=value,issued_at=f'2026-11-02T{hour}:00:00Z')
        record['prev']=records[-1]['hash'] if records else L.GENESIS
        record['hash']=L._hash(record['prev'],record)
        records.append(record)
    errors=L.verify_records(records)
    assert any('record 2:' in error for error in errors)
    assert not any('record 3:' in error for error in errors)


def test_post_write_path_failure_has_uncertain_outcome(tmp_path,monkeypatch):
    path=tmp_path/'ledger.jsonl'
    original=L._check_path
    calls=0
    def path_check(fd,path):
        nonlocal calls
        calls+=1
        if calls==3:raise L.LedgerError('injected post-write pathname replacement')
        return original(fd,path)
    monkeypatch.setattr(L,'_check_path',path_check)
    with pytest.raises(L.LedgerCommitUncertainError,match='pathname'):
        L.append(observation(),path)
    monkeypatch.setattr(L,'_check_path',original)
    assert len(L.read(path))==1
    assert L.verify(path)==[]


def test_rolled_back_io_failure_is_typed_and_preserves_prefix(tmp_path,monkeypatch):
    path=tmp_path/'ledger.jsonl'
    L.append(observation(),path)
    before=path.read_bytes()
    def fail_write(fd,payload):raise OSError('injected write error')
    monkeypatch.setattr(L.os,'write',fail_write)
    with pytest.raises(L.LedgerWriteError,match='rollback completed') as caught:
        L.append(observation('next'),path)
    assert isinstance(caught.value.__cause__,OSError)
    assert path.read_bytes()==before
    assert L.verify(path)==[]
