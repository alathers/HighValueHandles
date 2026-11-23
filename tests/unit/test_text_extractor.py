"""Unit tests for the Tesseract OCR pipeline in image_processor.text_extractor.

Part A needs no Tesseract (retry wiring, token rules, image loading, resizing,
region selection). Part B runs the real ``tesseract`` executable on synthetic
images rendered with Pillow's FreeType default font and is skipped when
Tesseract (or a scalable default font) is unavailable. Nothing touches the
network or the repository's photos.
"""

import io
import shutil
import time
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import numpy as np
import pytesseract
import pytest
from PIL import Image, ImageDraw, ImageFont

from image_processor.text_extractor import (
    MIN_WORD_CONFIDENCE,
    ExtractedText,
    OCRWord,
    TextExtractor,
    clean_ocr_text,
    is_plausible_token,
    load_image,
    normalize_token,
    resize_long_edge,
    strong_score,
)
from lib.error_handling import RetryException, RetryManager
from lib.exceptions import (
    ConfigurationError,
    ImageProcessingError,
    OCRError,
    RetryExhaustedError,
)


def _scalable_default_font() -> bool:
    try:
        return isinstance(ImageFont.load_default(size=72), ImageFont.FreeTypeFont)
    except TypeError:  # Pillow < 10.1 has no size argument
        return False


requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None or not _scalable_default_font(),
    reason="tesseract not installed or Pillow lacks a scalable default font",
)

LINES = ["BUFFALO TRACE", "KENTUCKY STRAIGHT", "BOURBON WHISKEY", "90 PROOF 750 ML"]
EXPECTED_WORDS = {"BUFFALO", "TRACE", "KENTUCKY", "BOURBON", "PROOF", "90"}


def _result(clean: str = "BOURBON") -> ExtractedText:
    return ExtractedText(raw_text="x", clean_text=clean, words=[])


def _render(
    size: tuple[int, int] = (1200, 900),
    font_px: int = 72,
    bg: str = "white",
    fg: str = "black",
) -> Image.Image:
    """Render the synthetic label text onto an RGB canvas."""
    font = ImageFont.load_default(size=font_px)
    image = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(image)
    y = round(font_px * 1.1)
    for line in LINES:
        draw.text((round(font_px * 1.1), y), line, fill=fg, font=font)
        y += round(font_px * 1.6)
    return image


# ---------------------------------------------------------------------------
# A. Retry wiring (no Tesseract)
# ---------------------------------------------------------------------------


class TestRetryWiring:
    """extract_with_retry must go through RetryManager.execute_with_retry."""

    def test_extract_with_retry_uses_execute_with_retry(self):
        """Regression for the RetryManager.retry() AttributeError."""
        rm = Mock(spec=RetryManager)  # spec: .retry would raise AttributeError
        rm.execute_with_retry.return_value = _result("BOURBON")
        ext = TextExtractor(rm)

        assert ext.extract_with_retry(Path("a.jpg")) == "BOURBON"
        rm.execute_with_retry.assert_called_once_with(
            ext.extract,
            Path("a.jpg"),
            detect_rotation=True,
            select_regions=True,
            error_context=ANY,
        )

    def test_orientation_corrected_disables_rotation(self):
        rm = Mock(spec=RetryManager)
        rm.execute_with_retry.return_value = _result()
        ext = TextExtractor(rm)

        ext.extract_with_retry(Path("a.jpg"), orientation_corrected=True, select_regions=False)

        _, kwargs = rm.execute_with_retry.call_args
        assert kwargs["detect_rotation"] is False
        assert kwargs["select_regions"] is False

    def test_extract_detailed_with_retry_returns_extracted_text(self):
        rm = Mock(spec=RetryManager)
        result = _result("KENTUCKY")
        rm.execute_with_retry.return_value = result

        assert TextExtractor(rm).extract_detailed_with_retry("a.jpg") is result

    def test_retry_exception_becomes_retry_exhausted(self):
        rm = Mock(spec=RetryManager)
        rm.execute_with_retry.side_effect = RetryException("x", OCRError("boom"), 1)

        with pytest.raises(RetryExhaustedError) as excinfo:
            TextExtractor(rm).extract_with_retry(Path("a.jpg"))

        assert isinstance(excinfo.value.original_error, OCRError)
        assert excinfo.value.retry_count == 1
        assert "2 attempt(s)" in excinfo.value.message

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("nope"),
            ImageProcessingError("bad"),
            ConfigurationError("no tess"),
        ],
    )
    def test_file_errors_not_wrapped(self, error):
        rm = Mock(spec=RetryManager)
        rm.execute_with_retry.side_effect = RetryException("x", error, 0)

        with pytest.raises(type(error)) as excinfo:
            TextExtractor(rm).extract_with_retry(Path("a.jpg"))
        assert excinfo.value is error

    def test_default_policy_does_not_retry_image_errors(self):
        ext = TextExtractor()
        with (
            patch.object(ext, "extract", side_effect=ImageProcessingError("bad")) as extract,
            patch("lib.error_handling.time.sleep") as sleep,
            pytest.raises(ImageProcessingError),
        ):
            ext.extract_with_retry(Path("a.jpg"))
        assert extract.call_count == 1
        sleep.assert_not_called()

    def test_default_policy_does_not_retry_missing_tesseract(self):
        ext = TextExtractor()
        with (
            patch.object(ext, "extract", side_effect=ConfigurationError("no tess")) as extract,
            pytest.raises(ConfigurationError),
        ):
            ext.extract_with_retry(Path("a.jpg"))
        assert extract.call_count == 1

    def test_default_policy_retries_ocr_errors_once(self):
        ext = TextExtractor()
        with (
            patch.object(ext, "extract", side_effect=OCRError("t")) as extract,
            patch("lib.error_handling.time.sleep") as sleep,
            pytest.raises(RetryExhaustedError),
        ):
            ext.extract_with_retry(Path("a.jpg"))
        assert extract.call_count == 2
        sleep.assert_called_once_with(1)

    def test_default_policy_recovers_after_transient_failure(self):
        ext = TextExtractor()
        with (
            patch.object(ext, "extract", side_effect=[OCRError("t"), _result("BOURBON")]),
            patch("lib.error_handling.time.sleep"),
        ):
            assert ext.extract_with_retry(Path("a.jpg")) == "BOURBON"

    def test_legacy_detect_orientation_api_removed(self):
        """The OSD/minAreaRect orientation code was wrong and is gone."""
        ext = TextExtractor()
        assert not hasattr(ext, "detect_orientation")
        assert not hasattr(ext, "correct_orientation")
        assert not hasattr(TextExtractor, "retry")


# ---------------------------------------------------------------------------
# A. Token rules (no Tesseract)
# ---------------------------------------------------------------------------


class TestTokenRules:
    @pytest.mark.parametrize(
        ("token", "confidence"),
        [
            ("BOURBON", 90),
            ("121.2", 70),
            ("55%", 70),
            ("750ml", 70),
            ("Akkeshi", 75),
            ("No", 90),
            ("ALC/VOL", 90),
            ("KY", 90),
            ("McKenna", 90),
            ("12th", 70),
        ],
    )
    def test_keeps_plausible_tokens(self, token, confidence):
        assert is_plausible_token(normalize_token(token), confidence)

    @pytest.mark.parametrize(
        ("token", "confidence"),
        [
            ("4", 95),
            ("~", 90),
            ("eRB", 90),
            ("SSS", 90),
            ("MLA", 70),
            ("BOURBON", 50),
            ("Ss", 90),
            ("xx", 99),
            ("a7b3", 90),
            ("", 99),
            ("BRRR", 90),
        ],
    )
    def test_drops_noise_tokens(self, token, confidence):
        assert not is_plausible_token(normalize_token(token), confidence)

    def test_min_confidence_threshold(self):
        assert is_plausible_token("BOURBON", MIN_WORD_CONFIDENCE)
        assert not is_plausible_token("BOURBON", MIN_WORD_CONFIDENCE - 0.1)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("“BOURBON.”", "BOURBON"),
            ("121.2", "121.2"),
            ("PROOF.", "PROOF"),
            ("|~|", ""),
            ("(KENTUCKY)", "KENTUCKY"),
        ],
    )
    def test_normalize_token(self, raw, expected):
        assert normalize_token(raw) == expected

    def test_clean_ocr_text(self):
        noisy = "‘IW | KENTUCKY ~ BOURBON BOURBON 4 — eRB 90 PROOF"  # noqa: RUF001
        assert clean_ocr_text(noisy) == "KENTUCKY BOURBON 90 PROOF"
        assert clean_ocr_text(noisy, max_chars=15) == "KENTUCKY"

    def test_clean_ocr_text_is_idempotent_and_handles_empty(self):
        once = clean_ocr_text("Kentucky  straight\nBOURBON whiskey 47% ALC/VOL")
        assert once == "Kentucky straight BOURBON whiskey 47% ALC/VOL"
        assert clean_ocr_text(once) == once
        assert clean_ocr_text("") == ""
        assert clean_ocr_text("BOURBON", max_chars=0) == ""

    def test_clean_ocr_text_hard_cuts_single_long_token(self):
        assert clean_ocr_text("DISTILLERY", max_chars=4) == "DIST"

    def test_strong_score(self):
        words = [
            OCRWord("BOURBON", 95.0, 1, 1, 1),  # 7
            OCRWord("PROOF.", 90.0, 1, 1, 1),  # normalised to 5
            OCRWord("WHISKEY", 70.0, 1, 1, 1),  # below strong confidence
            OCRWord("ALC", 99.0, 1, 1, 1),  # shorter than 4
            OCRWord("eRBq", 99.0, 1, 1, 1),  # implausible
        ]
        assert strong_score(words) == 12
        assert strong_score([]) == 0

    def test_extracted_text_is_reliable_and_str(self):
        weak = ExtractedText(raw_text="", clean_text="ack", quality_score=19)
        strong = ExtractedText(raw_text="", clean_text="BOURBON", quality_score=20)
        assert not weak.is_reliable
        assert strong.is_reliable
        assert str(strong) == "BOURBON"


# ---------------------------------------------------------------------------
# A. Image loading and resizing (no Tesseract)
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_transparent_rgba_composited_on_white(self, tmp_path):
        image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((50, 30, 150, 70), fill=(0, 0, 0, 255))
        path = tmp_path / "rgba.png"
        image.save(path)

        loaded = load_image(path)

        assert loaded.mode == "L"
        assert loaded.getpixel((0, 0)) == 255
        assert loaded.getpixel((100, 50)) == 0

    def test_exif_orientation_applied(self, tmp_path):
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 CW on display
        path = tmp_path / "exif6.jpg"
        Image.new("RGB", (300, 200), "white").save(path, exif=exif.tobytes())

        assert load_image(path).size == (200, 300)

    def test_16bit_image_normalised_not_clipped(self, tmp_path):
        gradient = np.tile(np.linspace(0, 65535, 256, dtype=np.uint16), (64, 1))
        path = tmp_path / "gray16.png"
        Image.fromarray(gradient).save(path)

        loaded = load_image(path)

        assert loaded.mode == "L"
        assert np.ptp(np.asarray(loaded)) > 100

    def test_constant_16bit_image_becomes_white(self, tmp_path):
        path = tmp_path / "flat16.png"
        Image.fromarray(np.full((10, 10), 1000, dtype=np.uint16)).save(path)

        assert np.asarray(load_image(path)).min() == 255

    @pytest.mark.parametrize(("mode", "suffix"), [("P", ".png"), ("CMYK", ".jpg"), ("1", ".png")])
    def test_other_modes_load_as_l(self, tmp_path, mode, suffix):
        path = tmp_path / f"img{suffix}"
        Image.new("RGB", (64, 32), "white").convert(mode).save(path)

        assert load_image(path).mode == "L"

    def test_corrupt_file_raises_image_processing_error(self, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not an image at all")

        with pytest.raises(ImageProcessingError):
            load_image(path)

    def test_truncated_file_raises_image_processing_error(self, tmp_path):
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), "white").save(buf, format="PNG")
        path = tmp_path / "truncated.png"
        path.write_bytes(buf.getvalue()[:60])

        with pytest.raises(ImageProcessingError):
            load_image(path)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.jpg")


class TestResizeLongEdge:
    @pytest.mark.parametrize(
        ("size", "long_edge", "expected"),
        [
            ((6000, 4500), 2000, (2000, 1500)),
            ((4500, 6000), 1200, (900, 1200)),
            ((400, 300), 2000, (1000, 750)),
            ((1500, 1000), 2000, (1500, 1000)),
            ((100, 50), 2000, (400, 200)),  # upscale capped at 4x
        ],
    )
    def test_resize(self, size, long_edge, expected):
        assert resize_long_edge(Image.new("L", size), long_edge).size == expected


class TestSelectTextRegions:
    def _array(self) -> np.ndarray:
        arr = np.zeros((400, 600), dtype=np.uint8)
        arr[100:250, 150:450] = 255  # 300x150 white label, ~19% of the image
        return arr

    def test_accepts_2d(self):
        boxes = TextExtractor().select_text_regions(self._array())

        assert boxes == [(150, 100, 300, 150)]

    def test_accepts_3d(self):
        rgb = np.stack([self._array()] * 3, axis=-1)
        boxes = TextExtractor().select_text_regions(rgb)

        assert all(isinstance(b, tuple) and len(b) == 4 for b in boxes)
        assert boxes == [(150, 100, 300, 150)]

    def test_blank_image_has_no_regions(self):
        assert TextExtractor().select_text_regions(np.full((100, 100), 255, np.uint8)) == []

    def test_bad_input_returns_empty(self):
        assert TextExtractor().select_text_regions(np.zeros((0, 0, 3), dtype=np.uint8)) == []


class TestTesseractErrors:
    def test_tesseract_missing_raises_configuration_error(self):
        with (
            patch("pytesseract.image_to_data", side_effect=pytesseract.TesseractNotFoundError()),
            pytest.raises(ConfigurationError),
        ):
            TextExtractor()._ocr_words(Image.new("L", (10, 10)))

    def test_tesseract_timeout_raises_ocr_error(self):
        with (
            patch(
                "pytesseract.image_to_data", side_effect=RuntimeError("Tesseract process timeout")
            ),
            pytest.raises(OCRError),
        ):
            TextExtractor()._ocr_words(Image.new("L", (10, 10)))

    def test_ocr_words_filters_empty_and_negative_confidence(self):
        data = {
            "text": ["", "BOURBON", "  ", "noise", "PROOF"],
            "conf": ["-1", "91.5", "-1", "-1", 88],
            "block_num": [0, 1, 1, 1, 2],
            "par_num": [0, 1, 1, 1, 1],
            "line_num": [0, 1, 1, 1, 1],
        }
        with patch("pytesseract.image_to_data", return_value=data) as image_to_data:
            words = TextExtractor(tesseract_timeout=7)._ocr_words(Image.new("L", (10, 10)))

        assert words == [OCRWord("BOURBON", 91.5, 1, 1, 1), OCRWord("PROOF", 88.0, 2, 1, 1)]
        assert image_to_data.call_args.kwargs["timeout"] == 7
        assert "--psm 11" in image_to_data.call_args.kwargs["config"]

    def test_rotation_vote_keeps_strong_upright(self):
        ext = TextExtractor()
        upright = [OCRWord("KENTUCKY", 95.0, 1, 1, 1), OCRWord("STRAIGHT", 95.0, 1, 1, 1),
                   OCRWord("BOURBON", 95.0, 1, 1, 1)]  # fmt: skip
        with patch.object(ext, "_ocr_words") as ocr:
            assert ext.detect_rotation(Image.new("L", (10, 10)), upright) == (0, upright)
        ocr.assert_not_called()

    def test_rotation_vote_requires_minimum_gain(self):
        ext = TextExtractor()
        upright = [OCRWord("BOURBON", 95.0, 1, 1, 1)]  # score 7
        small_gain = [OCRWord("KENTUCKY", 95.0, 1, 1, 1), OCRWord("RYE", 99.0, 1, 1, 1)]  # 8
        big_gain = [OCRWord("KENTUCKY", 95.0, 1, 1, 1), OCRWord("STRAIGHT", 95.0, 1, 1, 1),
                    OCRWord("BOURBON", 95.0, 1, 1, 1)]  # 23  # fmt: skip
        with patch.object(ext, "_ocr_words", side_effect=[small_gain, big_gain]):
            rotation, words = ext.detect_rotation(Image.new("L", (10, 10)), upright)
        assert (rotation, words) == (270, big_gain)


# ---------------------------------------------------------------------------
# B. Real Tesseract on synthetic images
# ---------------------------------------------------------------------------


@requires_tesseract
class TestTesseractPipeline:
    """End-to-end OCR on rendered labels (real tesseract, no network)."""

    @staticmethod
    def _assert_text(result: ExtractedText) -> None:
        assert set(result.clean_text.split()) >= EXPECTED_WORDS, result.clean_text

    def _extract(self, path: Path, **kwargs) -> ExtractedText:
        return TextExtractor().extract(path, **kwargs)

    def test_upright(self, tmp_path):
        path = tmp_path / "upright.png"
        _render().save(path)

        result = self._extract(path)

        self._assert_text(result)
        assert result.rotation_applied == 0
        assert result.is_reliable
        assert result.mean_confidence > MIN_WORD_CONFIDENCE
        assert result.source_path == str(path)
        assert "BUFFALO" in result.raw_text

    @pytest.mark.parametrize(("pixel_rotation", "expected"), [(-90, 90), (90, 270)])
    def test_sideways_images_are_rotated(self, tmp_path, pixel_rotation, expected):
        path = tmp_path / "sideways.png"
        _render().rotate(pixel_rotation, expand=True).save(path)

        result = self._extract(path)

        assert result.rotation_applied == expected
        self._assert_text(result)

    def test_rotation_detection_can_be_disabled(self, tmp_path):
        path = tmp_path / "sideways.png"
        _render().rotate(-90, expand=True).save(path)

        result = self._extract(path, detect_rotation=False)

        assert result.rotation_applied == 0
        assert not result.is_reliable

    def test_exif_orientation_applied_before_ocr(self, tmp_path):
        exif = Image.Exif()
        exif[274] = 6
        path = tmp_path / "exif6.jpg"
        _render().rotate(90, expand=True).save(path, quality=95, exif=exif.tobytes())

        result = self._extract(path)

        assert result.rotation_applied == 0
        self._assert_text(result)

    @pytest.mark.parametrize("variant", ["rgba", "gray", "palette", "white_on_black"])
    def test_image_mode_variants(self, tmp_path, variant):
        path = tmp_path / f"{variant}.png"
        if variant == "rgba":
            # Black text on a fully transparent background.
            mask = _render(bg="black", fg="white").convert("L")
            image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
            image.putalpha(mask)
        elif variant == "gray":
            image = _render().convert("L")
        elif variant == "palette":
            image = _render().convert("P")
        else:
            image = _render(bg="black", fg="white")
        image.save(path)

        self._assert_text(self._extract(path))

    def test_tiny_image_upscaled(self, tmp_path):
        path = tmp_path / "tiny.png"
        _render(size=(400, 300), font_px=24).save(path)

        self._assert_text(self._extract(path))

    def test_blank_image(self, tmp_path):
        path = tmp_path / "blank.png"
        Image.new("RGB", (800, 600), "white").save(path)

        result = self._extract(path)

        assert result.clean_text == ""
        assert result.raw_text.strip() == ""
        assert result.words == []
        assert result.mean_confidence == 0.0
        assert not result.is_reliable

    def test_huge_image_downscaled(self, tmp_path):
        path = tmp_path / "huge.jpg"
        _render(size=(6000, 4500), font_px=360).save(path, quality=90)

        start = time.monotonic()
        result = self._extract(path)

        self._assert_text(result)
        assert result.duration_s < 10
        assert time.monotonic() - start < 15

    def test_extract_with_retry_end_to_end(self, tmp_path):
        path = tmp_path / "upright.png"
        _render().save(path)

        text = TextExtractor().extract_with_retry(path)

        assert isinstance(text, str)
        assert "BOURBON" in text.split()

    def test_extract_text_returns_clean_text(self, tmp_path):
        path = tmp_path / "upright.png"
        _render().save(path)

        assert "KENTUCKY" in TextExtractor().extract_text(path, orientation_corrected=True)
