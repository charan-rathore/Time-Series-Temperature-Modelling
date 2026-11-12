"""The audit must never see the future: features at row t are unchanged when
every value after t is replaced, and the climatology table built for an
origin only uses rows before it."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "scripts/walk_forward_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_features_use_no_future_rows():
    d = audit.load()
    t = 500
    base = audit.features(d).iloc[t]
    future = d.copy()
    cols = ["temp_c"] + [c for c in audit.WEATHER if c in d]
    future.loc[t + 1:, cols] = 999.0
    after = audit.features(future).iloc[t]
    assert np.allclose(base.values, after.values, equal_nan=True)


def test_climatology_ignores_rows_after_origin():
    d = audit.load()
    upto = 600
    a = audit.climatology(d, upto)
    future = d.copy()
    future.loc[upto:, "temp_c"] = 999.0
    assert np.allclose(a, audit.climatology(future, upto))


def test_block_bootstrap_can_sample_every_observation():
    n, L = 365, 14
    starts = audit.block_starts(np.random.default_rng(0), n, L, 4000)
    covered = np.zeros(n, bool)
    covered[(starts[..., None] + np.arange(L)).ravel()] = True
    assert starts.max() == n - L
    assert covered.all()
