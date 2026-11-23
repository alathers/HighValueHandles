"""OCR regression on real bottle photos tracked in the repository.

Read-only: the photos are opened, never written, and nothing is created next to
them. Skipped when Tesseract or the photos are unavailable. Expected keywords
are a subset of what the measured pipeline reads (recall 1.0 and 0.78), so
small Tesseract version drift does not break the test.
"""

import shutil
from pathlib import Path

import pytest

from image_processor.text_extractor import TextExtractor

PHOTO_DIR = Path(__file__).resolve().parents[2] / "HighValueRoom"

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def _photo(name: str) -> Path:
    path = PHOTO_DIR / name
    if not path.is_file():
        pytest.skip(f"photo not available: {path}")
    return path


@pytest.mark.parametrize(
    ("name", "keywords"),
    [
        ("IMG_4659.jpeg", {"akkeshi", "whisky", "distillery"}),  # flat box, upright
        ("IMG_4655.jpeg", {"single", "barrel"}),  # Four Roses back label
    ],
)
def test_real_photo_ocr(name, keywords):
    path = _photo(name)
    mtime = path.stat().st_mtime_ns

    result = TextExtractor().extract_detailed_with_retry(path)

    tokens = set(result.clean_text.lower().split())
    assert keywords <= tokens, result.clean_text
    assert result.rotation_applied == 0
    assert result.is_reliable
    assert result.duration_s < 10
    assert path.stat().st_mtime_ns == mtime  # never modified
