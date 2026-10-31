"""Artifact identity must bind to the exact bytes a model was deserialized
from. A file swapped on disk between deserialization and any independent
re-hash must never let an old in-memory model wear a new artifact's
identity (the load(path)-then-rehash(path) race)."""

import hashlib
from pathlib import Path

from src.models.loader import _load_with_identity


def test_identity_binds_to_deserialized_bytes_even_when_path_is_swapped(tmp_path):
    original = tmp_path / "model.pkl"
    original.write_bytes(b"version-one")
    seen = {}

    def load_fn(path):
        # The loader reads the path it is handed...
        seen["bytes"] = Path(path).read_bytes()
        # ...and an attacker/racing retrain replaces the ORIGINAL artifact
        # before any post-hoc hash of that path could run.
        original.write_bytes(b"version-two")

    digest = _load_with_identity(original, load_fn)

    assert seen["bytes"] == b"version-one"
    assert digest == hashlib.sha256(b"version-one").hexdigest()[:16]
    assert digest != hashlib.sha256(b"version-two").hexdigest()[:16]


def test_identity_matches_a_fresh_untouched_file(tmp_path):
    original = tmp_path / "model.pkl"
    payload = b"stable-bytes" * 1000
    original.write_bytes(payload)
    digest = _load_with_identity(original, lambda path: Path(path).read_bytes())
    assert digest == hashlib.sha256(payload).hexdigest()[:16]


def test_snapshot_is_cleaned_up(tmp_path):
    original = tmp_path / "model.pkl"
    original.write_bytes(b"x")
    before = set(Path("/tmp").glob("ts-model-*"))
    _load_with_identity(original, lambda path: None)
    after = set(Path("/tmp").glob("ts-model-*"))
    assert after == before


def test_tft_branch_preserves_family_load_and_claims_no_identity(tmp_path, monkeypatch):
    """TFT compatibility regression at the actual family contract.

    TFTModel.load(path) derives path.with_suffix('.ckpt') and
    path.with_suffix('.meta.pkl'); models/tft.pkl itself need not exist.
    The loader must keep handing it the family path (so siblings resolve)
    and must NOT record a single-file artifact hash it cannot verify -
    TFT provenance is unknown by design, never guessed.

    Executed with a stub TFTModel: qualifies the loader wiring only, not a
    real torch checkpoint load (no TFT checkpoint exists in this repo).
    """
    import src.models.loader as loader

    (tmp_path / "tft.ckpt").write_bytes(b"checkpoint")
    (tmp_path / "tft.meta.pkl").write_bytes(b"meta")
    monkeypatch.setattr(loader, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(loader, "_TFT_AVAILABLE", True)

    received = {}

    class StubTFT:
        def __init__(self, cfg):
            self.is_fitted = True

        def load(self, path):
            received["path"] = path
            # the family contract: siblings must resolve from the given path
            assert (Path(path).with_suffix(".ckpt")).exists()
            assert (Path(path).with_suffix(".meta.pkl")).exists()

    monkeypatch.setattr(loader, "TFTModel", StubTFT)

    manager = loader.ModelManager()
    manager.load_models({"models": {}})

    assert received["path"] == str(tmp_path / "tft.pkl")
    assert manager.tft is not None
    assert not any(name.startswith("tft") for name in manager.artifact_hashes)


def test_snapshot_is_cleaned_up_when_deserialization_fails(tmp_path):
    import pytest
    original = tmp_path / "model.pkl"
    original.write_bytes(b"not a model")
    snapshots = []
    def fail(path):
        snapshots.append(Path(path))
        raise RuntimeError("deserialization failed")
    with pytest.raises(RuntimeError, match="deserialization failed"):
        _load_with_identity(original, fail)
    assert len(snapshots) == 1
    assert not snapshots[0].exists()
