#!/usr/bin/env python3
"""Run full Python suite with an isolated real Egres, then verify clean-restart retention.

Usage: THERMOSENSE_EGRES_BINARY=/absolute/path/to/egres python scripts/history/qualify.py
Only sample data. Uses fresh temporary storage, private loopback offset ports and cleans up the server.
Requires the official checksum-verified Egres v0.2610.0 binary, pytest and history optional dependencies.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.history.egres import EgresHistory
from src.history.observations import Observation


def main():
    binary=Path(os.environ['THERMOSENSE_EGRES_BINARY']).resolve()
    with tempfile.TemporaryDirectory(prefix='thermosense-egres-') as directory:
        root=Path(directory);offset=500
        for offset in range(500,900):
            sockets=[]
            try:
                for base in (3210,4210,5210):
                    s=socket.socket();s.bind(('127.0.0.1',base+offset));sockets.append(s)
                break
            except OSError:continue
            finally:
                for s in sockets:s.close()
        else:raise RuntimeError('no free test port set')
        dsn=f'host=127.0.0.1 port={3210+offset} user=sample dbname=egres sslmode=disable'
        env=os.environ.copy();env['THERMOSENSE_TEST_EGRES_DSN']=dsn
        log=open(root/'server.log','a')
        command=[str(binary),'--cache=total_bytes=256MiB',f'--store=dir_path={root}/store',
                 f'--wal=dir={root}/wal',f'--listen=offset={offset}','--io=auto_calibrate=false']
        process=None
        def start():
            p=subprocess.Popen(command,stdout=log,stderr=log)
            for _ in range(100):
                if p.poll() is not None:raise RuntimeError('Egres startup failed')
                try:
                    with EgresHistory(dsn).connect() as c:c.execute('SELECT 1')
                    time.sleep(.6) # listener alone precedes single-node election
                    return p
                except Exception:time.sleep(.1)
            p.terminate();p.wait(timeout=15);raise RuntimeError('Egres not ready')
        try:
            process=start()
            store=EgresHistory(dsn);store.initialize()
            sentinel=Observation.create(location='retention_test',source='sample',observed_at='2026-10-28T08:00:00Z',temperature_c=25.)
            store.append(sentinel)
            result=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml='+str(Path(os.environ.get('THERMOSENSE_HISTORY_TEST_REPORT',tempfile.gettempdir()+'/thermosense-history-suite.xml')))],cwd=ROOT,env=env)
            process.terminate();process.wait(timeout=15);process=None
            process=start()
            rows=store.read(location='retention_test',source='sample',start='2026-10-28',end='2026-10-28')
            assert len(rows)==1 and rows[0].payload()==sentinel.payload()
            print('Clean-restart retention and original timestamp/source payload verified',flush=True)
            return result.returncode
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()

if __name__=='__main__':raise SystemExit(main())
