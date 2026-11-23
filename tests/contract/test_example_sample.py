"""The committed example output (examples/processed_sample/) must match what the code writes today.

A fresh copy is built with ``scripts/regen_example_sample.py`` (real
``save_bottle_files``, scripted search and text-model replies, no model and
no network) and compared with the committed sample. If a test here fails
after an output change, regenerate the sample:

    uv run python scripts/regen_example_sample.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "regen_example_sample.py"
SAMPLE_ROOT = REPO_ROOT / "examples" / "processed_sample"
REGENERATE = "regenerate it with `uv run python scripts/regen_example_sample.py`"


def _load_script():
    spec = importlib.util.spec_from_file_location("regen_example_sample", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_folder() -> Path:
    folders = [p for p in SAMPLE_ROOT.iterdir() if p.is_dir()]
    assert len(folders) == 1, f"expected one bottle folder in {SAMPLE_ROOT}, found {folders}"
    return folders[0]


def _key_paths(value, prefix: str = "") -> set[str]:
    """Every key path in a JSON value; list items share the path ``key[]``."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= _key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= _key_paths(item, f"{prefix}[]")
    return paths


def _files(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sample")
    return _load_script().regenerate(out / "processed_sample")


def test_sample_json_key_set_matches_fresh_output(fresh):
    folder = _sample_folder()
    committed = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    current = json.loads((fresh / f"{fresh.name}.json").read_text(encoding="utf-8"))
    assert set(committed) == set(current), f"top-level keys drifted; {REGENERATE}"
    assert _key_paths(committed) == _key_paths(current), f"nested keys drifted; {REGENERATE}"
    assert committed["schema_version"] == current["schema_version"] == 2


def test_sample_json_matches_fresh_output(fresh):
    """Values too, not only keys: the script pins every clock, so the output is reproducible."""
    folder = _sample_folder()
    committed = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    current = json.loads((fresh / f"{fresh.name}.json").read_text(encoding="utf-8"))
    assert committed == current, f"the bottle JSON drifted; {REGENERATE}"


def test_sample_queries_come_from_the_real_query_builder():
    """The scripted search must not invent queries: they are what build_queries makes of the label."""
    from image_processor.web_research import build_queries

    folder = _sample_folder()
    data = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    identity = {k: data["metadata"][k] for k in ("name", "distillery", "specific_release")}
    identity["batch_number"] = data["batch_number"]  # the hint value, as the enricher passes it
    assert data["web_research"]["queries"] == build_queries(identity)


def _subset_problems(expected, actual, path: str = "") -> list[str]:
    """Where ``expected`` (with "..." placeholders) is not a sub-structure of ``actual``."""
    if expected == "...":
        return []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: not an object"]
        problems: list[str] = []
        for key, value in expected.items():
            if key == "...":
                continue
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems += _subset_problems(value, actual[key], f"{path}.{key}")
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: not a list"]
        return [
            f"{path}[{i}]: no matching item"
            for i, item in enumerate(expected)
            if item != "..." and not any(not _subset_problems(item, a) for a in actual)
        ]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def test_detailed_docs_json_excerpt_matches_the_sample():
    """docs/detailed/README.md shows an excerpt of the sample JSON; every value in it must be real."""
    import re

    doc = (REPO_ROOT / "docs" / "detailed" / "README.md").read_text(encoding="utf-8")
    match = re.search(r"### JSON File.*?```json\n(.*?)```", doc, re.S)
    assert match, "JSON excerpt not found in docs/detailed/README.md"
    excerpt = json.loads(match.group(1))
    folder = _sample_folder()
    data = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    assert _subset_problems(excerpt, data) == []


def test_regenerate_refuses_to_replace_unrelated_files(tmp_path):
    """--out pointing at a folder with other files must not delete them."""
    target = tmp_path / "somewhere"
    target.mkdir()
    (target / "keep.txt").write_text("not a sample", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_script().regenerate(target)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "not a sample"


def test_sample_files_match_fresh_output(fresh):
    folder = _sample_folder()
    assert folder.name == fresh.name
    assert _files(folder) == _files(fresh), REGENERATE


def test_sample_markdown_matches_fresh_output(fresh):
    folder = _sample_folder()
    committed = (folder / f"{folder.name}.md").read_text(encoding="utf-8")
    current = (fresh / f"{fresh.name}.md").read_text(encoding="utf-8")
    assert committed == current, f"the Markdown page drifted; {REGENERATE}"


def test_sample_shows_the_provenance_cases():
    """Supported with label + OCR, MSRP Unknown with a reason, a resolved conflict, the web log."""
    folder = _sample_folder()
    data = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    fields = data["facts"]["fields"]

    proof_sources = {c["source_id"] for c in fields["proof"]["citations"]}
    assert fields["proof"]["status"] == "supported"
    assert {"L1", "O1"} <= proof_sources

    assert fields["msrp"]["status"] == "unknown"
    assert fields["msrp"]["value"] is None
    assert "no source stated an MSRP" in fields["msrp"]["reason"]
    assert data["msrp"] is None
    assert data["label_price"] == "64.99"

    conflicts = data["facts"]["conflicts"]
    assert any(c["field"] == "batch_number" and c["resolved_by"] for c in conflicts)

    outcomes = {p["outcome"] for p in data["web_research"]["pages"]}
    assert outcomes == {"cited", "fetched_not_cited", "skipped"}
    assert data["facts"]["publish"]["ready"] is False


def test_sample_evidence_hashes_match_the_facts():
    folder = _sample_folder()
    data = json.loads((folder / f"{folder.name}.json").read_text(encoding="utf-8"))
    web_sources = [s for s in data["facts"]["sources"] if s["id"].startswith("W")]
    assert web_sources
    for source in web_sources:
        text = (folder / source["evidence_path"]).read_bytes()
        assert hashlib.sha256(text).hexdigest() == source["content_sha256"]


def test_sample_holds_no_machine_paths():
    folder = _sample_folder()
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            for marker in ("/Users/", "/home/", "/private/", "/tmp/", "/var/folders/"):
                assert marker not in text, f"{path.name} contains a machine path ({marker})"


def test_hints_example_is_a_valid_hints_file():
    """examples/hints.example.toml (named in --hints help) loads with the real reader."""
    from lib.hints import HintBook

    book = HintBook.load(REPO_ROOT / "examples" / "hints.example.toml")
    assert len(book) >= 1
    assert any(book.fields_for(bottle_id) for bottle_id in ("IMG_4643",))
    assert book.source_for("IMG_4643")
