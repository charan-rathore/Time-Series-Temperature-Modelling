"""Job-claim atomicity and log-buffer safety for the pipeline routes."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.routes import pipeline


@pytest.fixture(autouse=True)
def clean_state():
    with pipeline._job_lock:
        pipeline._job_store.clear()
    with pipeline._log_lock:
        pipeline._log_lines.clear()
    yield


def test_claim_job_rejects_second_running_claim():
    assert pipeline._claim_job("backfill") is True
    assert pipeline._claim_job("backfill") is False


def test_claim_allows_reclaim_after_finish():
    assert pipeline._claim_job("daily") is True
    with pipeline._job_lock:
        pipeline._job_store["daily"]["status"] = "completed"
    assert pipeline._claim_job("daily") is True


def test_concurrent_posts_have_exactly_one_winner():
    started = threading.Event()
    release = threading.Event()

    def fake_run(job_id, cmd):
        started.set()
        release.wait(timeout=5)
        with pipeline._job_lock:
            pipeline._job_store[job_id]["status"] = "completed"

    original = pipeline._run_script
    pipeline._run_script = fake_run
    try:
        client = TestClient(app, raise_server_exceptions=False)
        results = []

        def post():
            results.append(client.post("/api/pipeline/backfill", json={}).status_code)

        first = threading.Thread(target=post)
        first.start()
        assert started.wait(timeout=5)
        post()  # second POST while the first worker is alive
        first.join()
        release.set()
        assert sorted(results) == [200, 409]
    finally:
        pipeline._run_script = original


def test_train_claim_survives_invalid_payload():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/pipeline/train", json={"models": ["not-a-model"]})
    assert response.status_code == 400
    with pipeline._job_lock:
        assert "train" not in pipeline._job_store, "a rejected payload must not leave a phantom claim"


def test_append_log_stays_capped_under_concurrency():
    def writer(prefix):
        for i in range(200):
            pipeline._append_log(f"{prefix}-{i}")

    threads = [threading.Thread(target=writer, args=(f"w{n}",)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with pipeline._log_lock:
        snapshot = list(pipeline._log_lines)
    assert len(snapshot) <= pipeline._LOG_MAX
    assert all(line.startswith("[") for line in snapshot)
