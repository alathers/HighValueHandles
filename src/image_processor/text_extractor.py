"""Text extraction from bourbon bottle images using Tesseract OCR.

OCR is a *hint* for the vision model, not the primary reader: Tesseract does
well on flat back labels and boxes and poorly on curved, embossed or script
front labels. The pipeline below was measured on real bottle photos (keyword
recall 0.20 -> ~0.59, ~1.1s per image) and is tuned for clean, low-noise output
rather than maximum raw text:

1. Load the image, apply its EXIF orientation, composite transparency onto
   white, normalise 16-bit/float data and convert to 8-bit grayscale.
2. OCR a 1200px (long edge) copy with ``--oem 1 --psm 11`` (sparse text) via
   ``image_to_data`` so every word carries a confidence.
3. Pick the rotation by OCR voting: only when the upright pass is weak, also
   OCR the 90 and 270 degree rotations and keep one only if it is clearly
   better. Tesseract's OSD is not used (it is unreliable on bottle photos).
4. OCR the (rotated) image at 2000px plus up to three label-region crops.
5. Merge the passes, filter tokens by confidence and plausibility, and dedupe
   into ``clean_text``; the best pass is also kept verbatim as ``raw_text``.

Only POSIX-portable dependencies are used (Pillow, OpenCV, NumPy and the
``tesseract`` executable on ``PATH``).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from lib.error_handling import ErrorType, RetryException, RetryManager
from lib.exceptions import (
    ConfigurationError,
    ImageProcessingError,
    OCRError,
    RetryExhaustedError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:  # Optional HEIC/HEIF support (iPhone originals); harmless when absent.
    from pillow_heif import register_heif_opener  # type: ignore[import-not-found]

    register_heif_opener()
except ImportError:  # pragma: no cover - depends on the environment
    pass

logger = logging.getLogger(__name__)

#: Long edge (px) of the main OCR pass.
PRIMARY_LONG_EDGE = 2000
#: Long edge (px) of the secondary pass, also used for rotation voting.
SECONDARY_LONG_EDGE = 1200
#: Images whose long edge is below this are upscaled before OCR.
MIN_LONG_EDGE = 1000
#: Maximum upscale factor for tiny images.
MAX_UPSCALE = 4.0

#: Minimum Tesseract word confidence (0-100) for a token to be kept.
MIN_WORD_CONFIDENCE = 60.0
#: Minimum confidence for non-whitelisted three-letter tokens.
SHORT_WORD_CONFIDENCE = 85.0
#: Minimum confidence for a word to count towards the quality ("strong") score.
STRONG_WORD_CONFIDENCE = 80.0
#: Minimum normalised length for a word to count towards the strong score.
STRONG_WORD_MIN_LENGTH = 4
#: Minimum fraction of letters in a non-numeric token.
MIN_LETTER_RATIO = 0.75

#: Upright strong score at or above which rotation voting is skipped.
ROTATION_KEEP_SCORE = 20
#: Points a rotated pass must gain over upright to be chosen.
ROTATION_MIN_GAIN = 10
#: Rotations (PIL counter-clockwise degrees) tried by the vote. 180 is skipped:
#: handheld photos are never upside down and EXIF covers sensor orientation.
CANDIDATE_ROTATIONS = (90, 270)
#: Quality score at or above which OCR output is considered reliable.
RELIABLE_SCORE = 20

#: Per-call Tesseract timeout in seconds.
TESSERACT_TIMEOUT_S = 30
#: LSTM engine, sparse-text page segmentation.
TESSERACT_CONFIG = "--oem 1 --psm 11"

#: Label-region crops: area bounds (fraction of image), aspect bounds, count.
REGION_MIN_AREA = 0.05
REGION_MAX_AREA = 0.7
REGION_MIN_ASPECT = 0.5
REGION_MAX_ASPECT = 5.0
REGION_MAX_COUNT = 3
#: Regions covering at least this fraction of the image add nothing; skip them.
REGION_SKIP_AREA = 0.6

#: Punctuation stripped from both ends of every token.
STRIP_CHARS = "|“”‘’'\"`~_—–=<>[]{}()\\/,;:!?*^·•«»®©™°"  # noqa: RUF001 - OCR punctuation
#: Numeric tokens worth keeping: prices, proofs, ABV, volumes, ordinals, "No.5".
NUM_RE = re.compile(
    r"^(\$?\d+([.,]\d+)*(%|ml|l|cl|oz|yr|yrs)?|\d{1,3}(st|nd|rd|th)|no\.?\d+)$",
    re.IGNORECASE,
)
#: Short tokens (1-3 chars) that are allowed despite the length rules.
SHORT_TOKEN_WHITELIST = frozenset(
    {
        "no", "of", "by", "in", "to", "ky", "co", "oz", "nd", "st", "rd", "th",
        "de", "la", "le", "el", "al", "on", "at", "an", "or", "is", "it", "ii",
        "iv", "vi", "xi", "usa", "ml",
    }
)  # fmt: skip

_DECIMAL_RE = re.compile(r"\d\.\d")
_VOWEL_RE = re.compile(r"[aeiouy]")
_TRIPLE_RE = re.compile(r"(.)\1\1")
_MIXED_CASE_RE = re.compile(r"[a-z][A-Z]")
_NAME_PREFIX_RE = re.compile(r"^(Mc|Mac|O'|D')")
_TRANSPARENT_MODES = frozenset({"RGBA", "LA", "PA"})
_HIGH_DEPTH_MODES = frozenset({"I", "F"})

_TESSERACT_INSTALL_HINT = (
    "Install Tesseract OCR and make sure 'tesseract' is on PATH "
    "(macOS: brew install tesseract; Debian/Ubuntu: sudo apt install tesseract-ocr; "
    "Fedora: sudo dnf install tesseract)"
)


@dataclass(frozen=True)
class OCRWord:
    """One word recognised by Tesseract.

    Attributes:
        text: Recognised text, as returned by Tesseract (not normalised).
        confidence: Word confidence, 0-100.
        block: Tesseract block number (reading-order grouping).
        paragraph: Paragraph number within the block.
        line: Line number within the paragraph.
    """

    text: str
    confidence: float
    block: int
    paragraph: int
    line: int


@dataclass
class ExtractedText:
    """Result of OCR on one image.

    Attributes:
        raw_text: Unfiltered, line-structured text of the best OCR pass
            (words with confidence >= 0, lines joined by newlines).
        clean_text: Filtered, de-duplicated tokens joined by single spaces;
            this is what should be shown to a model. May be empty.
        words: The words that made it into ``clean_text``.
        mean_confidence: Mean confidence (0-100) of ``words``; 0.0 if none.
        quality_score: Strong score of ``words`` (see ``strong_score``).
        rotation_applied: Rotation applied before OCR, in PIL
            counter-clockwise degrees (0, 90 or 270). The file is never
            modified.
        duration_s: Wall-clock time spent on this image.
        source_path: Path of the source image.
    """

    raw_text: str
    clean_text: str
    words: list[OCRWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    quality_score: int = 0
    rotation_applied: int = 0
    duration_s: float = 0.0
    source_path: str = ""

    @property
    def is_reliable(self) -> bool:
        """Whether the OCR text is strong enough to be trusted as a hint.

        When False, callers should tell the vision model to rely on the image.
        """
        return self.quality_score >= RELIABLE_SCORE

    def __str__(self) -> str:
        """Return the cleaned text."""
        return self.clean_text


# --------------------------------------------------------------------------
# Pure helpers (no Tesseract)
# --------------------------------------------------------------------------


def normalize_token(token: str) -> str:
    """Strip surrounding punctuation from an OCR token.

    A trailing period is removed unless the token contains a decimal number
    (e.g. ``"121.2"`` keeps its digits, ``"BOURBON."`` becomes ``"BOURBON"``).

    Args:
        token: Raw OCR token.

    Returns:
        Normalised token (possibly empty).
    """
    token = token.strip(STRIP_CHARS)
    if not _DECIMAL_RE.search(token):
        token = token.rstrip(".")
    return token


def is_plausible_token(  # noqa: PLR0911 - one early return per rule
    token: str,
    confidence: float,
    min_confidence: float = MIN_WORD_CONFIDENCE,
) -> bool:
    """Decide whether a normalised OCR token looks like real label text.

    No dictionary is used, so unusual brand names survive; the rules only
    reject the typical shapes of OCR noise.

    Args:
        token: Normalised token (see ``normalize_token``).
        confidence: Tesseract confidence for the token (0-100).
        min_confidence: Minimum confidence required.

    Returns:
        True if the token should be kept.
    """
    if not token or confidence < min_confidence:
        return False
    if NUM_RE.match(token):
        return not (len(token) == 1 and token.isdigit())

    letters = sum(ch.isalpha() for ch in token)
    if letters < 2 or letters / len(token) < MIN_LETTER_RATIO:
        return False

    lower = token.lower()
    whitelisted = lower in SHORT_TOKEN_WHITELIST
    if len(lower) <= 2 and not whitelisted:
        return False
    if len(lower) == 3 and not whitelisted and confidence < SHORT_WORD_CONFIDENCE:
        return False
    if not whitelisted and not _VOWEL_RE.search(lower):
        return False
    if _TRIPLE_RE.search(lower):
        return False
    # Mixed-case garbage such as "eRB" (but not "McKenna" or all-caps tokens).
    return not (
        _MIXED_CASE_RE.search(token) and not token.isupper() and not _NAME_PREFIX_RE.match(token)
    )


def clean_ocr_text(text: str, *, max_chars: int | None = None) -> str:
    """Clean free OCR text using the token rules, ignoring confidence.

    Splits on whitespace, normalises each token, keeps plausible ones,
    collapses consecutive case-insensitive duplicates and joins with single
    spaces. Idempotent on already-clean text.

    Args:
        text: Text to clean (e.g. combined OCR from several images).
        max_chars: If set, truncate at the last space at or before this many
            characters (a single over-long first token is cut hard).

    Returns:
        Cleaned text (possibly empty).
    """
    kept: list[str] = []
    for raw in text.split():
        token = normalize_token(raw)
        if not is_plausible_token(token, 100.0):
            continue
        if kept and kept[-1].lower() == token.lower():
            continue
        kept.append(token)
    cleaned = " ".join(kept)

    if max_chars is not None and len(cleaned) > max_chars:
        if max_chars <= 0:
            return ""
        cut = cleaned.rfind(" ", 0, max_chars + 1)
        cleaned = cleaned[:cut] if cut > 0 else cleaned[:max_chars]
    return cleaned


def strong_score(words: Iterable[OCRWord]) -> int:
    """Score how much confident, plausible text a set of words contains.

    The score is the total length of normalised words that have confidence of
    at least ``STRONG_WORD_CONFIDENCE``, at least four characters and pass
    ``is_plausible_token`` at that confidence. Wrongly rotated text scores 0
    in practice, which is what makes it useful for orientation voting.

    Args:
        words: OCR words.

    Returns:
        Non-negative integer score.
    """
    score = 0
    for word in words:
        if word.confidence < STRONG_WORD_CONFIDENCE:
            continue
        token = normalize_token(word.text)
        if len(token) >= STRONG_WORD_MIN_LENGTH and is_plausible_token(
            token, word.confidence, STRONG_WORD_CONFIDENCE
        ):
            score += len(token)
    return score


def load_image(path: Path) -> Image.Image:
    """Load an image as an upright 8-bit grayscale (mode ``L``) PIL image.

    Applies the EXIF orientation, composites transparency onto white (so
    black text on a transparent background stays readable) and min-max
    normalises 16-bit / 32-bit / float images instead of clipping them.

    Args:
        path: Image file path.

    Returns:
        Grayscale PIL image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageProcessingError: If the file cannot be decoded or converted.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        opened = Image.open(path)
    except Exception as e:
        raise ImageProcessingError(f"Failed to load image {path}", str(e)) from e

    try:
        try:
            opened.load()
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image {path}", str(e)) from e
        return _to_upright_gray(opened, path)
    finally:
        opened.close()


def _to_upright_gray(image: Image.Image, path: Path) -> Image.Image:
    """Convert a loaded PIL image to upright 8-bit grayscale (see ``load_image``)."""
    try:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:  # malformed EXIF must not lose the image
            logger.debug("Ignoring unreadable EXIF orientation in %s: %s", path, e)

        if image.mode in _TRANSPARENT_MODES or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            image = canvas
        elif image.mode in _HIGH_DEPTH_MODES or image.mode.startswith("I;16"):
            arr = np.asarray(image, dtype=np.float32)
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if hi > lo:
                scaled = (np.nan_to_num(arr, nan=lo) - lo) * (255.0 / (hi - lo))
                arr8 = np.clip(scaled, 0, 255).astype(np.uint8)
            else:
                arr8 = np.full(arr.shape, 255, dtype=np.uint8)
            image = Image.fromarray(arr8)

        return image.convert("L")
    except Exception as e:
        raise ImageProcessingError(f"Failed to prepare image {path} for OCR", str(e)) from e


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale an image so its long edge suits OCR.

    Images larger than ``long_edge`` are downscaled to it; images smaller than
    ``MIN_LONG_EDGE`` are upscaled towards it (by at most ``MAX_UPSCALE``);
    anything in between is returned unchanged.

    Args:
        image: PIL image.
        long_edge: Target long edge in pixels.

    Returns:
        Resized image (or the same object if no resize was needed).
    """
    width, height = image.size
    current = max(width, height)
    if current <= 0:
        return image
    if current > long_edge:
        scale = long_edge / current
    elif current < MIN_LONG_EDGE:
        scale = min(MIN_LONG_EDGE / current, MAX_UPSCALE)
    else:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _lines_text(words: Sequence[OCRWord]) -> str:
    """Join words into lines grouped by (block, paragraph, line)."""
    lines: list[list[str]] = []
    key: tuple[int, int, int] | None = None
    for word in words:
        word_key = (word.block, word.paragraph, word.line)
        if word_key != key:
            lines.append([])
            key = word_key
        lines[-1].append(word.text)
    return "\n".join(" ".join(line) for line in lines)


def _merge_passes(passes: Sequence[Sequence[OCRWord]]) -> tuple[list[str], list[OCRWord]]:
    """Merge OCR passes into clean tokens.

    The first pass is walked in reading order (collapsing consecutive
    case-insensitive duplicates); later passes only contribute tokens not
    seen before.

    Args:
        passes: OCR passes, best first.

    Returns:
        Tuple of (clean tokens, the words they came from).
    """
    tokens: list[str] = []
    kept: list[OCRWord] = []
    seen: set[str] = set()

    for index, words in enumerate(passes):
        for word in words:
            token = normalize_token(word.text)
            if not is_plausible_token(token, word.confidence):
                continue
            lower = token.lower()
            if index == 0:
                if tokens and tokens[-1].lower() == lower:
                    continue
            elif lower in seen:
                continue
            seen.add(lower)
            tokens.append(token)
            kept.append(word)
    return tokens, kept


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------


class TextExtractor:
    """Extract label text from bourbon bottle images using Tesseract OCR.

    Attributes:
        retry_manager: RetryManager used by the ``*_with_retry`` methods. The
            default retries transient Tesseract failures (``OCRError``) once
            and never retries deterministic ones (missing file, undecodable
            image, Tesseract not installed).
        tesseract_timeout: Per-call Tesseract timeout in seconds.
    """

    def __init__(
        self,
        retry_manager: RetryManager | None = None,
        *,
        tesseract_timeout: int = TESSERACT_TIMEOUT_S,
    ):
        """Initialize text extractor.

        Args:
            retry_manager: Optional RetryManager; defaults to one OCR-only retry.
            tesseract_timeout: Per-call Tesseract timeout in seconds.
        """
        self.retry_manager = retry_manager or RetryManager(
            max_attempts=2,
            backoff_delays=[1],
            retry_on=[ErrorType.OCR_FAILURE],
        )
        self.tesseract_timeout = tesseract_timeout

    def _ocr_words(self, image: Image.Image) -> list[OCRWord]:
        """Run one Tesseract pass and return words with confidence >= 0.

        Args:
            image: Grayscale PIL image.

        Returns:
            Recognised words in Tesseract's reading order.

        Raises:
            ConfigurationError: If the tesseract executable is not available.
            OCRError: If Tesseract fails or times out.
        """
        try:
            data = pytesseract.image_to_data(
                image,
                config=TESSERACT_CONFIG,
                output_type=pytesseract.Output.DICT,
                timeout=self.tesseract_timeout,
            )
        except pytesseract.TesseractNotFoundError as e:
            raise ConfigurationError(
                "Tesseract OCR is not installed or not on PATH", _TESSERACT_INSTALL_HINT
            ) from e
        except Exception as e:
            raise OCRError("Tesseract OCR failed", str(e)) from e

        words: list[OCRWord] = []
        for i, raw_text in enumerate(data.get("text", [])):
            text = (raw_text or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if confidence < 0:
                continue
            words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                    block=int(data["block_num"][i]),
                    paragraph=int(data["par_num"][i]),
                    line=int(data["line_num"][i]),
                )
            )
        return words

    def detect_rotation(
        self,
        gray_small: Image.Image,
        upright_words: list[OCRWord],
    ) -> tuple[int, list[OCRWord]]:
        """Choose the image rotation by OCR voting.

        If the upright pass already scores ``ROTATION_KEEP_SCORE`` or more it is
        kept. Otherwise the image is also OCR'd at 90 and 270 degrees and a
        rotation is chosen only if it beats the best score so far by at least
        ``ROTATION_MIN_GAIN``.

        Args:
            gray_small: Upright grayscale image at the secondary resolution.
            upright_words: OCR words of ``gray_small`` as-is.

        Returns:
            Tuple of (rotation in PIL counter-clockwise degrees, words of the
            chosen orientation).

        Raises:
            ConfigurationError: If the tesseract executable is not available.
            OCRError: If Tesseract fails.
        """
        best_score = strong_score(upright_words)
        if best_score >= ROTATION_KEEP_SCORE:
            return 0, upright_words

        best_rotation, best_words = 0, upright_words
        for rotation in CANDIDATE_ROTATIONS:
            words = self._ocr_words(gray_small.rotate(rotation, expand=True))
            score = strong_score(words)
            if score >= best_score + ROTATION_MIN_GAIN:
                best_score, best_rotation, best_words = score, rotation, words
        return best_rotation, best_words

    def select_text_regions(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find likely label regions with Otsu thresholding and contours.

        Args:
            gray: Image array, 2-D grayscale or 3-D RGB.

        Returns:
            Up to ``REGION_MAX_COUNT`` boxes ``(x, y, width, height)``, largest
            first, excluding boxes that cover most of the image. Empty if
            nothing suitable is found or OpenCV fails.
        """
        try:
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            if gray.dtype != np.uint8:
                lo, hi = float(gray.min()), float(gray.max())
                span = hi - lo if hi > lo else 1.0
                gray = ((gray.astype(np.float32) - lo) * (255.0 / span)).astype(np.uint8)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            height, width = gray.shape[:2]
            image_area = float(height * width)
            boxes: list[tuple[int, int, int, int]] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if not REGION_MIN_AREA * image_area < area < REGION_MAX_AREA * image_area:
                    continue
                x, y, box_w, box_h = cv2.boundingRect(contour)
                aspect = box_w / max(box_h, 1)
                if not REGION_MIN_ASPECT < aspect < REGION_MAX_ASPECT:
                    continue
                if box_w * box_h >= REGION_SKIP_AREA * image_area:
                    continue
                boxes.append((int(x), int(y), int(box_w), int(box_h)))

            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return boxes[:REGION_MAX_COUNT]
        except Exception as e:  # cv2.error, unexpected shapes
            logger.debug("Text region selection failed: %s", e)
            return []

    def extract(
        self,
        image_path: Path,
        *,
        detect_rotation: bool = True,
        select_regions: bool = True,
    ) -> ExtractedText:
        """Run the full OCR pipeline on one image.

        Args:
            image_path: Image file path.
            detect_rotation: Whether to vote on 90/270 degree rotations.
            select_regions: Whether to add label-region crop passes.

        Returns:
            ExtractedText with clean and raw text plus quality information.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImageProcessingError: If the image cannot be decoded.
            ConfigurationError: If the tesseract executable is not available.
            OCRError: If Tesseract fails or times out.
        """
        start = time.monotonic()
        image_path = Path(image_path)
        gray = load_image(image_path)

        secondary = resize_long_edge(gray, SECONDARY_LONG_EDGE)
        secondary_words = self._ocr_words(secondary)

        rotation = 0
        if detect_rotation:
            rotation, secondary_words = self.detect_rotation(secondary, secondary_words)
            if rotation:
                gray = gray.rotate(rotation, expand=True)

        primary = resize_long_edge(gray, PRIMARY_LONG_EDGE)
        passes: list[list[OCRWord]] = [self._ocr_words(primary), secondary_words]

        if select_regions:
            primary_arr = np.asarray(primary)
            for x, y, box_w, box_h in self.select_text_regions(primary_arr):
                crop = Image.fromarray(primary_arr[y : y + box_h, x : x + box_w])
                passes.append(self._ocr_words(crop))

        passes.sort(key=strong_score, reverse=True)  # stable: primary wins ties
        raw_text = _lines_text(passes[0])
        tokens, kept = _merge_passes(passes)

        mean_confidence = sum(w.confidence for w in kept) / len(kept) if kept else 0.0
        result = ExtractedText(
            raw_text=raw_text,
            clean_text=" ".join(tokens),
            words=kept,
            mean_confidence=mean_confidence,
            quality_score=strong_score(kept),
            rotation_applied=rotation,
            duration_s=time.monotonic() - start,
            source_path=str(image_path),
        )
        logger.debug(
            "OCR %s: rotation=%d words=%d quality=%d reliable=%s in %.2fs",
            image_path,
            result.rotation_applied,
            len(kept),
            result.quality_score,
            result.is_reliable,
            result.duration_s,
        )
        return result

    def extract_text(
        self,
        image_path: Path,
        orientation_corrected: bool = False,
        select_regions: bool = True,
    ) -> str:
        """Extract cleaned text from an image (single attempt).

        Args:
            image_path: Image file path.
            orientation_corrected: True if the image is known to be upright
                (skips rotation voting).
            select_regions: Whether to add label-region crop passes.

        Returns:
            Cleaned OCR text, possibly empty.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImageProcessingError: If the image cannot be decoded.
            ConfigurationError: If the tesseract executable is not available.
            OCRError: If Tesseract fails or times out.
        """
        return self.extract(
            image_path,
            detect_rotation=not orientation_corrected,
            select_regions=select_regions,
        ).clean_text

    def extract_detailed_with_retry(
        self,
        image_path: Path,
        *,
        detect_rotation: bool = True,
        select_regions: bool = True,
    ) -> ExtractedText:
        """Run ``extract`` under the retry policy.

        Args:
            image_path: Image file path.
            detect_rotation: Whether to vote on 90/270 degree rotations.
            select_regions: Whether to add label-region crop passes.

        Returns:
            ExtractedText for the image.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImageProcessingError: If the image cannot be decoded.
            ConfigurationError: If the tesseract executable is not available.
            RetryExhaustedError: If OCR still fails after all attempts.
        """
        try:
            result: ExtractedText = self.retry_manager.execute_with_retry(
                self.extract,
                Path(image_path),
                detect_rotation=detect_rotation,
                select_regions=select_regions,
                error_context=f"OCR {image_path}",
            )
        except RetryException as e:
            original = e.original_error
            if isinstance(original, (FileNotFoundError, ImageProcessingError, ConfigurationError)):
                raise original from e
            raise RetryExhaustedError(
                f"OCR failed for {image_path} after {e.retry_count + 1} attempt(s)",
                e.retry_count,
                original,
            ) from original
        return result

    def extract_with_retry(
        self,
        image_path: Path,
        orientation_corrected: bool = False,
        select_regions: bool = True,
    ) -> str:
        """Extract cleaned text under the retry policy.

        This is the entry point used by ``BottleAnalyzer``.

        Args:
            image_path: Image file path.
            orientation_corrected: True if the image is known to be upright
                (skips rotation voting).
            select_regions: Whether to add label-region crop passes.

        Returns:
            Cleaned OCR text (plausible tokens in reading order), possibly empty.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImageProcessingError: If the image cannot be decoded.
            ConfigurationError: If the tesseract executable is not available.
            RetryExhaustedError: If OCR still fails after all attempts.
        """
        return self.extract_detailed_with_retry(
            image_path,
            detect_rotation=not orientation_corrected,
            select_regions=select_regions,
        ).clean_text
