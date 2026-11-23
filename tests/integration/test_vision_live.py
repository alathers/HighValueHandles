"""Opt-in live vision checks on real bottle photos (regressions seen in the E2E run).

Skipped unless ``BOURBON_LIVE_TESTS=1`` and ``BOURBON_LIVE_VISION=1``, the
photos under ``HighValueRoom/`` exist and the model is already installed in a
running Ollama. Never pulls a model; the photos are only read.

Run::

    BOURBON_LIVE_TESTS=1 BOURBON_LIVE_VISION=1 \
        uv run --extra dev pytest tests/integration/test_vision_live.py

Override the model with ``BOURBON_MODEL``.
"""

import os
from pathlib import Path

import pytest
import requests

from image_processor.bottle_analyzer import MIXED_GROUP_WARNING, BottleAnalyzer
from lib.config import DEFAULT_VISION_MODEL
from lib.ollama_client import OllamaClient

PHOTO_DIR = Path(__file__).resolve().parents[2] / "HighValueRoom"
OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("BOURBON_MODEL") or DEFAULT_VISION_MODEL

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BOURBON_LIVE_TESTS") != "1" or os.environ.get("BOURBON_LIVE_VISION") != "1",
        reason="set BOURBON_LIVE_TESTS=1 BOURBON_LIVE_VISION=1 to run",
    ),
    pytest.mark.allow_model_server,
]

# Photo -> expectations on single-photo analysis (case-insensitive substrings;
# None = must be absent). From the labels themselves, not from model output.
CASES = {
    # Old Fitzgerald: bonded with a season on the label.
    "IMG_4645.jpeg": {"specific_release": ["bottled-in-bond", "spring 2021"]},
    # George T. Stagg: barrel proof, unfiltered.
    "IMG_4668.jpeg": {"specific_release": ["barrel proof", "unfiltered"]},
    # Woodford Reserve: "Batch Proof 121.2" is the proof, not a batch number.
    "IMG_4676.jpeg": {"batch_number_not": "121.2", "proof": ["121.2"]},
    # Blood Oath: generic class words stay out of the release.
    "IMG_4639.jpeg": {"specific_release_not": "kentucky straight bourbon whiskey"},
}


def _model_installed() -> bool:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json().get("models", [])
    except (requests.RequestException, ValueError):
        return False
    return any(t.get("name") == MODEL for t in tags)


@pytest.fixture(scope="module")
def analyzer():
    if not _model_installed():
        pytest.skip(f"{MODEL} is not installed in a running Ollama at {OLLAMA_URL}")
    return BottleAnalyzer(ollama_client=OllamaClient(base_url=OLLAMA_URL, timeout=600))


@pytest.mark.parametrize("name", sorted(CASES))
def test_single_photo_label_fields(analyzer, name):
    path = PHOTO_DIR / name
    if not path.is_file():
        pytest.skip(f"photo not available: {path}")
    mtime = path.stat().st_mtime_ns

    metadata = analyzer.analyze_with_vision_model([path], "", model=MODEL)

    expected = CASES[name]
    for field in ("specific_release", "proof"):
        for part in expected.get(field, []):
            assert part in (getattr(metadata, field) or "").lower(), (field, getattr(metadata, field))
    if "batch_number_not" in expected:
        assert expected["batch_number_not"] not in (metadata.batch_number or "")
    if "specific_release_not" in expected:
        assert expected["specific_release_not"] not in (metadata.specific_release or "").lower()
    assert metadata.abv, "ABV printed on every label must be kept"
    assert MIXED_GROUP_WARNING not in metadata.warnings  # one photo cannot mix products
    assert path.stat().st_mtime_ns == mtime
