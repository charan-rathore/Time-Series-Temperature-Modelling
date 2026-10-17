"""Documentation contracts: link integrity and explicit current boundaries."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "architecture"


def test_guide_has_native_disclosures_and_no_remote_runtime():
    html = (DOCS / "interactive.html").read_text()
    assert html.count("<details ") == 12
    assert html.count("<summary>") == 12
    assert '@media(max-width:700px)' in html
    assert 'name="viewport"' in html
    assert not re.search(r"<script\b|<iframe\b|@import|url\(", html, re.I)
    assert not re.search(r"jev|typesafe", html, re.I)


def test_source_and_asset_links_exist():
    html = (DOCS / "interactive.html").read_text()
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        assert not target.startswith(("http:", "https:", "//"))
        assert (DOCS / target).resolve().is_file(), target


def test_fallbacks_stores_and_runtime_limits_are_not_hidden():
    html = (DOCS / "interactive.html").read_text()
    for fact in ["26°C", "Placeholder values remain possible", "not automatically retrain",
                 "No hardware availability", "local filesystem artifacts",
                 "does not normalize", "Incomplete", "process-local"]:
        assert fact.lower() in html.lower(), fact


def test_svg_has_accessible_description_and_distinct_stores():
    svg = (DOCS / "observation-to-evidence.svg").read_text()
    ET.fromstring(svg)
    assert 'role="img" aria-labelledby="title desc"' in svg
    assert "Baseline SQLite" in svg and "training parquet" in svg
    assert "API CONSUMERS (NOT A CHAIN)" in svg
    assert not re.search(r"<script\b|<foreignObject\b", svg, re.I)
