import json

import pytest

from src.models import registry as reg


def _entry(rmse, n=174, artifacts=None, data="d1", model="lgbm"):
    return reg.make_version(model, artifacts or {"a.pkl": "aa"}, data, "2024 to 2026",
                            {"rmse": rmse, "n_observations": n}, git_sha="abc", created_at="t")


def test_version_id_is_deterministic_and_sensitive_to_inputs():
    assert _entry(1.5)["version"] == _entry(1.5)["version"]
    assert _entry(1.5)["version"] != _entry(1.5, artifacts={"a.pkl": "bb"})["version"]
    assert _entry(1.5)["version"] != _entry(1.5, data="d2")["version"]


def test_first_version_becomes_champion():
    r = {"versions": [], "champions": {}}
    out = reg.register(r, _entry(1.5))
    assert out["promoted"] and r["champions"]["lgbm"] == _entry(1.5)["version"]


def test_challenger_must_win_by_margin_to_replace_champion():
    r = {"versions": [], "champions": {}}
    reg.register(r, _entry(1.50, artifacts={"a.pkl": "v1"}))
    near = reg.register(r, _entry(1.49, artifacts={"a.pkl": "v2"}))      # 0.7% better
    assert near["registered"] and not near["promoted"]
    clear = reg.register(r, _entry(1.40, artifacts={"a.pkl": "v3"}))     # 6.7% better
    assert clear["promoted"]
    assert r["champions"]["lgbm"].startswith("lgbm-")
    assert len(r["versions"]) == 3                                        # history kept


def test_less_evidence_never_dethrones_champion():
    r = {"versions": [], "champions": {}}
    reg.register(r, _entry(1.50, n=174, artifacts={"a.pkl": "v1"}))
    out = reg.register(r, _entry(0.50, n=29, artifacts={"a.pkl": "v2"}))  # great score, tiny sample
    assert not out["promoted"]


def test_reregistering_the_same_version_is_a_noop():
    r = {"versions": [], "champions": {}}
    reg.register(r, _entry(1.5))
    again = reg.register(r, _entry(1.5))
    assert not again["registered"] and len(r["versions"]) == 1


def test_models_are_tracked_independently():
    r = {"versions": [], "champions": {}}
    reg.register(r, _entry(1.5))
    reg.register(r, _entry(1.3, model="sarima"))
    assert set(r["champions"]) == {"lgbm", "sarima"}


def test_save_load_round_trip_and_hashing(tmp_path):
    p = tmp_path / "registry.json"
    assert reg.load(p) == {"versions": [], "champions": {}}
    r = {"versions": [], "champions": {}}
    reg.register(r, _entry(1.5))
    reg.save(r, p)
    assert reg.load(p) == r
    f = tmp_path / "m.pkl"
    f.write_bytes(b"x")
    h1 = reg.artifact_hashes(tmp_path, ["m.pkl"])
    f.write_bytes(b"y")
    assert reg.artifact_hashes(tmp_path, ["m.pkl"]) != h1
