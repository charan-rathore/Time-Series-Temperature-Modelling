"""Benchmark actual healthy training route; replace thread launch, not route logic."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api.routes import pipeline
source = subprocess.check_output(["git", "show", f"{sys.argv[1] if len(sys.argv)>1 else 'origin/main'}:src/api/routes/pipeline.py"], text=True)
start = source.index('def run_training(')
end = source.index('\n\n@router.get', start)
namespace = dict(vars(pipeline))
exec(source[start:end], namespace)
class FakeThread:
    def __init__(self, **kwargs): pass
    def start(self): pass
namespace['threading'] = SimpleNamespace(Thread=FakeThread)
saved = pipeline.threading
pipeline.threading = SimpleNamespace(Thread=FakeThread)
pipeline._job_store.clear()
payloads = [pipeline.TrainRequest(models=['sarima']), pipeline.TrainRequest()]
def run(fn, n):
    t = time.perf_counter()
    for i in range(n): fn(payloads[i % 2])
    return (time.perf_counter()-t)*1000
try:
    run(namespace['run_training'], 10000); run(pipeline.run_training, 10000)
    before=[]; after=[]
    for i in range(15):
        if i % 2:
            after.append(run(pipeline.run_training, 50000)); before.append(run(namespace['run_training'], 50000))
        else:
            before.append(run(namespace['run_training'], 50000)); after.append(run(pipeline.run_training, 50000))
    print(json.dumps({'callsPerSample':50000,'samples':15,'beforeMs':before,'afterMs':after,'beforeMedianMs':statistics.median(before),'afterMedianMs':statistics.median(after),'ratio':statistics.median(after)/statistics.median(before)},indent=2))
    if statistics.median(after)>statistics.median(before)*1.05: raise SystemExit('Healthy route slowdown above 5% timing guard')
finally: pipeline.threading=saved
