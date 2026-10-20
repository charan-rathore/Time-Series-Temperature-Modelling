"""Serving must read data from paths Vercel actually bundles.

Vercel's Python function bundler leaves out committed files that live under
.gitignore'd directories (data/processed/, data/features/) even when
vercel.json includeFiles lists them. The API therefore serves zeros on the
live site. These tests pin the serving paths to a directory no ignore rule
touches (data/bundle/).
"""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gitignore_rules():
    rules = []
    for line in (ROOT / ".gitignore").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            rules.append(line)
    return rules


def _is_ignored(rel: str, rules) -> bool:
    for r in rules:
        if r.endswith("/") and (rel + "/").startswith(r):
            return True
        if fnmatch.fnmatch(rel, r) or fnmatch.fnmatch(Path(rel).name, r):
            return True
    return False


def _serving_paths():
    from src.api.routes import forecast, history, pipeline

    paths = {pipeline.PROCESSED_PATH, pipeline.FEATURES_PATH,
             forecast.PROCESSED_PATH, history.PROCESSED_PATH}
    return {str(p.resolve().relative_to(ROOT)) for p in paths}


def test_serving_data_paths_are_not_gitignored():
    rules = _gitignore_rules()
    for rel in _serving_paths():
        assert not _is_ignored(rel, rules), f"{rel} is covered by .gitignore"


def test_serving_data_files_are_committed():
    for rel in _serving_paths():
        assert (ROOT / rel).exists(), f"{rel} missing from the repo"
