"""Compare actual forecast loop with baseline using controlled predictor costs."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models import loader
from src.models.ensemble import EnsembleStacker
baseline = sys.argv[1] if len(sys.argv) > 1 else 'origin/main'
def old_class(file, name):
    namespace={'__name__':'src.models.perf_baseline','__package__':'src.models','__file__':str(Path('src/models')/(file+'.py'))}
    exec(subprocess.check_output(['git','show',f'{baseline}:src/models/{file}.py'],text=True),namespace)
    return namespace[name], namespace
old_stack,_=old_class('ensemble','EnsembleStacker')
old_manager,old_namespace=old_class('loader','ModelManager')
features=pd.DataFrame({'temp_c':[20.,21.,22.]})
for namespace in (vars(loader),old_namespace):
    namespace['load_processed']=lambda:features
    namespace['build_feature_matrix']=lambda *a,**kw:features

def setup(manager_type,stack_type):
    manager=manager_type();manager.is_loaded=True
    manager.sarima=SimpleNamespace(is_fitted=True,predict=lambda **kw:np.full(kw['steps'],20.))
    manager.ensemble=stack_type({'base_models':['sarima']});manager.ensemble.is_fitted=True
    manager.ensemble.meta_models={h:SimpleNamespace(predict=lambda x,value=h+20:np.array([value])) for h in (1,2,3)}
    return manager
before=setup(old_manager,old_stack);after=setup(loader.ModelManager,EnsembleStacker)
def run(manager,n):
    start=time.perf_counter()
    for _ in range(n):manager.forecast(3)
    return (time.perf_counter()-start)*1000
run(before,1000);run(after,1000);a=[];b=[]
for i in range(15):
    if i%2:b.append(run(after,3000));a.append(run(before,3000))
    else:a.append(run(before,3000));b.append(run(after,3000))
print(json.dumps({'forecastsPerSample':3000,'daysPerForecast':3,'samples':15,'beforeMs':a,'afterMs':b,'beforeMedianMs':statistics.median(a),'afterMedianMs':statistics.median(b),'ratio':statistics.median(b)/statistics.median(a)},indent=2))
if statistics.median(b)>statistics.median(a)*1.05:raise SystemExit('Forecast CPU regression beyond 5% timing-noise guard')
