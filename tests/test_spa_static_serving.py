"""Regression tests for SPA fallback serving (path traversal containment)."""

import importlib
import shutil

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module

PROJECT_ROOT = main_module._PROJECT_ROOT
SPA_SHELL = "<html>spa shell</html>"


@pytest.fixture
def spa_client():
    """Register the SPA fallback route by giving the app a real public/ dir."""
    public = PROJECT_ROOT / "public"
    assert not public.exists(), "test expects no pre-existing public/ directory"
    public.mkdir()
    (public / "index.html").write_text(SPA_SHELL)
    (public / "static").mkdir()
    reloaded = importlib.reload(main_module)
    try:
        yield TestClient(reloaded.app, raise_server_exceptions=False)
    finally:
        shutil.rmtree(public, ignore_errors=True)
        importlib.reload(main_module)


def test_spa_fallback_serves_real_frontend_files(spa_client):
    response = spa_client.get("/index.html")
    assert response.status_code == 200
    assert response.text == SPA_SHELL


def test_spa_fallback_serves_shell_for_unknown_paths(spa_client):
    response = spa_client.get("/some/client/side/route")
    assert response.status_code == 200
    assert response.text == SPA_SHELL


def test_spa_fallback_does_not_serve_files_outside_the_frontend_dir(spa_client):
    # README.md lives at the project root, one level above public/.
    readme = (PROJECT_ROOT / "README.md").read_text()
    for vector in ("/..%2fREADME.md", "/..%2FREADME.md", "/%2e%2e/README.md"):
        response = spa_client.get(vector)
        assert response.status_code == 200, vector
        assert response.text == SPA_SHELL, f"traversal via {vector} served outside content"
        assert response.text != readme


@pytest.fixture
def spa_client_no_static():
    """public/ with an index.html but no static/ subdirectory."""
    public = PROJECT_ROOT / "public"
    assert not public.exists(), "test expects no pre-existing public/ directory"
    public.mkdir()
    (public / "index.html").write_text(SPA_SHELL)
    yield public
    shutil.rmtree(public, ignore_errors=True)
    importlib.reload(main_module)


def test_app_serves_spa_when_frontend_has_no_static_dir(spa_client_no_static):
    # Vercel-style public/ output has no static/ folder; mounting /static
    # unconditionally must not crash the app at import time.
    reloaded = importlib.reload(main_module)
    client = TestClient(reloaded.app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == SPA_SHELL
