import hashlib
import tracemalloc
from src.models.registry import file_sha256


def test_actual_large_artifact_hash_has_bounded_peak_memory(tmp_path):
    artifact = tmp_path / "large-model.bin"
    block = bytes(range(256)) * 4096
    expected = hashlib.sha256()
    with artifact.open("wb") as stream:
        for _ in range(20):
            stream.write(block)
            expected.update(block)
    tracemalloc.start()
    try:
        actual = file_sha256(artifact)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected.hexdigest()
    assert peak < 3 * 1024 * 1024, peak


def test_actual_empty_and_non_block_aligned_artifacts_keep_sha256(tmp_path):
    for name, payload in [("empty", b""), ("tail", bytes(range(255)) * 5000 + b"tail")]:
        artifact = tmp_path / name
        artifact.write_bytes(payload)
        assert file_sha256(artifact) == hashlib.sha256(payload).hexdigest()
