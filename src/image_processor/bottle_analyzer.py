"""Analyze images and extract bourbon bottle metadata.

Processes single or multiple bottle images, groups them by bottle identity,
and creates Bottle model instances with extracted metadata.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from image_processor.image_organizer import ImageOrganizer
from image_processor.text_extractor import TextExtractor
from image_processor.vision_prompt import (
    MAX_VISION_IMAGES,
    VISION_MAX_IMAGE_EDGE,
    VISION_OPTIONS,
    VISION_SYSTEM_PROMPT,
    VisionExtraction,
    build_vision_prompt,
    parse_vision_response,
    vision_response_schema,
)
from lib.config import DEFAULT_VISION_MODEL
from lib.duplicates import DuplicateDetector
from lib.error_handling import RetryManager
from lib.exceptions import (
    FATAL_MODEL_ERRORS,
    ConfigurationError,
    VisionAnalysisError,
    VisionResponseError,
)
from lib.model_selector import ModelStage
from lib.ollama_client import OllamaClient
from models.bottle import ExtractedMetadata
from models.image import Image

if TYPE_CHECKING:
    from lib.model_selector import ModelSelector

# Maximum number of angle photos expected for a single bottle
MAX_ANGLES_PER_BOTTLE = 4

#: Attempts per bottle when the model answers with unusable output.
VISION_RESPONSE_ATTEMPTS = 2

#: Warning recorded when the model reports photos of different products.
MIXED_GROUP_WARNING = "Vision model reports photos show different products; check grouping"

logger = logging.getLogger(__name__)


def _category_name(img: Image) -> str:
    """Upper-case category name of an image (works for enums and plain strings)."""
    return str(getattr(img.category, "name", img.category)).upper()


class BottleAnalyzer:
    """Analyze bourbon bottle images and extract metadata.

    Uses TextExtractor for OCR and vision model analysis,
    groups related images, and creates Bottle instances.

    Attributes:
        text_extractor: TextExtractor for OCR and metadata extraction.
        ollama_client: Ollama client for vision model calls.
        retry_manager: RetryManager for automatic retries.
        model_selector: Optional ModelSelector for per-stage model resolution.
        last_vision_result: Raw result of the most recent vision call, or None.
    """

    def __init__(
        self,
        text_extractor: TextExtractor | None = None,
        ollama_client: OllamaClient | None = None,
        retry_manager: RetryManager | None = None,
        image_organizer: ImageOrganizer | None = None,
        model_selector: "ModelSelector | None" = None,
    ):
        """Initialize bottle analyzer.

        Args:
            text_extractor: Optional TextExtractor instance.
            ollama_client: Optional OllamaClient instance.
            retry_manager: Optional RetryManager instance.
            image_organizer: Optional ImageOrganizer instance.
            model_selector: Optional ModelSelector for per-stage model resolution.
        """
        # OCR keeps its own narrow retry policy (transient Tesseract failures only).
        self.text_extractor = text_extractor or TextExtractor()
        self.ollama_client = ollama_client or OllamaClient()
        self.retry_manager = retry_manager or RetryManager()
        self.image_organizer = image_organizer or ImageOrganizer()
        self.model_selector = model_selector
        #: Raw result of the most recent vision call (token counts for usage tracking).
        self.last_vision_result: Any = None

    def _resolve_vision_model(self, vision_model: str | None = None) -> str:
        """Pick the vision model: explicit argument, then selector, then default.

        Args:
            vision_model: Model requested by the caller (``-m``), or None.

        Returns:
            The model name to use.
        """
        if isinstance(vision_model, str) and vision_model:
            return vision_model
        if self.model_selector is not None:
            try:
                selected = self.model_selector.get_model(ModelStage.VISION_ANALYSIS)
            except Exception:
                logger.debug("Model selector could not resolve the vision model", exc_info=True)
            else:
                if isinstance(selected, str) and selected:
                    return selected
        return DEFAULT_VISION_MODEL

    def select_vision_images(self, images: list[Image], limit: int = MAX_VISION_IMAGES) -> list[Image]:
        """Choose the photos sent to the vision model for one bottle.

        Non-BILL photos come first, the primary photo leading, then input
        order. Receipts (BILL) are used only if nothing else exists. Exact
        duplicates and burst retakes (``near_duplicate_of``) are skipped so
        they do not use up an image slot, unless nothing else is left.

        Args:
            images: All photos of the bottle.
            limit: Maximum number of photos to return.

        Returns:
            At most ``limit`` images; the first is the bottle's primary photo.
        """
        if limit <= 0:
            return []
        distinct = [
            img for img in images
            if not getattr(img, "is_duplicate", False) and not getattr(img, "near_duplicate_of", None)
        ] or list(images)
        candidates = [img for img in distinct if _category_name(img) != "BILL"] or distinct
        ordered = sorted(candidates, key=lambda img: not getattr(img, "is_primary", False))
        return ordered[:limit]

    def process_single_bottle(
        self,
        images: list[Image],
        vision_model: str | None = None,
    ) -> tuple[dict, ExtractedMetadata]:
        """Process images for a single bottle and extract metadata.

        Runs OCR on every photo, then one vision call with up to
        ``MAX_VISION_IMAGES`` selected photos and their OCR text as hints.

        Args:
            images: List of Image objects for this bottle.
            vision_model: Vision model name; if None, the model selector's
                vision model, else ``DEFAULT_VISION_MODEL``.

        Returns:
            Tuple of (bottle_dict, metadata) where:
                - bottle_dict: Dictionary with bottle field values (name,
                  distillery, ...), ``images`` (primary first), ``quantity``,
                  ``model_used`` and ``confidence`` ('HIGH'/'MEDIUM'/'LOW')
                - metadata: ExtractedMetadata with the OCR text of all photos

        Raises:
            VisionAnalysisError: If the vision call failed or its answer was
                unusable (the bottle must be recorded as failed).
            VisionModelUnavailableError, OllamaConnectionError, ModelServerError:
                If the model or server cannot be used at all (fatal for a run).
        """
        logger.info("Processing bottle with %d image(s)", len(images))
        model = self._resolve_vision_model(vision_model)
        selected = self.select_vision_images(images)
        primary = selected[0] if selected else None

        # OCR every photo (also those not sent to the vision model).
        logger.debug("Extracting OCR text from %d image(s)", len(images))
        ocr_by_path: dict[str, str] = {}
        ocr_texts: list[str] = []
        for img in images:
            try:
                text = self.text_extractor.extract_with_retry(Path(img.path))
                if text.strip():
                    ocr_by_path[str(img.path)] = text.strip()
                    ocr_texts.append(text.strip())
                    logger.debug("Successfully extracted %d characters from %s", len(text.strip()), img.path)
            except ConfigurationError as e:
                # Tesseract missing/misconfigured: every remaining image would fail the same way.
                logger.error("OCR unavailable: %s", e)
                break
            except Exception as e:
                logger.warning("OCR failed for %s: %s: %s", img.path, type(e).__name__, e)

        combined_ocr = "\n".join(ocr_texts)
        logger.debug("Combined OCR text length: %d characters", len(combined_ocr))

        image_paths = [Path(img.path) for img in selected]
        hints = [ocr_by_path.get(str(img.path), "") for img in selected]
        if image_paths:
            logger.info("Analyzing %d image(s) with vision model: %s", len(image_paths), model)
        else:
            logger.warning("No images for this bottle; vision analysis has nothing to read")

        self.last_vision_result = None
        start_time = time.time()
        try:
            metadata = self.analyze_with_vision_model(image_paths, hints, model)
        except Exception as e:
            self._record_vision_usage(model, start_time, success=False, error=f"{type(e).__name__}: {e}")
            raise
        if image_paths:
            self._record_vision_usage(model, start_time, success=True)

        # Keep the OCR of every photo, and list every photo with the primary first.
        metadata.full_text = combined_ocr
        metadata.selected_text = combined_ocr
        ordered = [primary, *[img for img in images if img is not primary]] if primary else list(images)
        metadata.source_images = [str(img.path) for img in ordered]

        bottle_dict = self._build_bottle_dict(metadata, ordered)
        bottle_dict["model_used"] = metadata.vision_model or model
        bottle_dict["confidence"] = (metadata.vision_confidence or self.calculate_confidence(metadata)).upper()
        logger.info("Successfully processed bottle: %s", bottle_dict.get("name", "Unknown"))

        return bottle_dict, metadata

    def _record_vision_usage(
        self,
        model: str,
        start_time: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record one vision call with the model selector (best effort).

        Args:
            model: Model that was called.
            start_time: ``time.time()`` before the call.
            success: Whether the call produced usable metadata.
            error: Failure description, if any.
        """
        if self.model_selector is None or not hasattr(self.model_selector, "record_usage"):
            return
        result = getattr(self, "last_vision_result", None)
        try:
            self.model_selector.record_usage(
                stage=ModelStage.VISION_ANALYSIS,
                model=model,
                success=success,
                duration=time.time() - start_time,
                input_tokens=int(getattr(result, "prompt_eval_count", 0) or 0),
                output_tokens=int(getattr(result, "eval_count", 0) or 0),
                # No automatic fallback model: the resolved model is always the one used.
                fallback_used=False,
                error=error,
            )
        except Exception:
            logger.debug("Model usage tracking failed", exc_info=True)

    def group_images_by_bottle(
        self,
        images: list[Image],
        similarity_threshold: float = 0.8,
    ) -> list[list[Image]]:
        """Group images by bottle identity.

        Uses ImageOrganizer to group related images (multi-angle photos of same bottle).

        Args:
            images: List of Image objects to group.
            similarity_threshold: Minimum similarity score to consider images related.

        Returns:
            List of image groups, where each group is a list of Image objects
                for the same bottle.
        """
        return self.image_organizer.group_by_bottle(images, similarity_threshold)

    def detect_quantity(
        self,
        images: list[Image],
        duplicate_detector: DuplicateDetector | None = None,
    ) -> int:
        """Detect quantity based on hash-based duplicates and visual analysis.

        Args:
            images: List of images for this bottle.
            duplicate_detector: Optional DuplicateDetector for hash checking.

        Returns:
            Detected quantity (minimum 1).
        """
        if not images:
            return 0

        # Use provided detector or create new one
        detector = duplicate_detector or DuplicateDetector()

        # Count hash duplicates
        hash_counts: dict[str, int] = {}
        for img in images:
            try:
                img_hash = detector.generate_hash(Path(img.path))
                hash_counts[img_hash] = hash_counts.get(img_hash, 0) + 1
            except Exception as e:
                logger.warning("Failed to generate hash for %s: %s", img.path, e)

        # Maximum duplicate count suggests quantity
        max_count = max(hash_counts.values()) if hash_counts else 1

        # Visual alternative: detect multiple bottles in single frame
        # Heuristic: Use hash-based approach to detect multiple identical bottles
        # If we have more images than the max count, check if there are duplicate
        # groups that suggest multiple bottles photographed together
        if len(images) > max_count:
            # Analyze image hashes to count visually identical bottles
            # Multiple photos of the same bottle will have different hashes,
            # but multiple identical bottles photographed separately will show
            # repeating hash patterns (each bottle photographed from same angle)
            visual_count = self._estimate_quantity_from_images(images)
            # Return the higher of the metadata count or visual analysis
            # Metadata count may be wrong if not explicitly stated,
            # visual analysis helps catch implicit multiple bottles
            return max(max_count, visual_count)

        return max_count

    def calculate_confidence(self, metadata: ExtractedMetadata) -> str:
        """Calculate overall extraction confidence.

        Aggregates confidence scores from ExtractedMetadata fields.

        Args:
            metadata: ExtractedMetadata with confidence scores.

        Returns:
            Confidence level: "HIGH", "MEDIUM", or "LOW".
        """
        # Extract metadata confidence (if present)
        # Note: In the current ExtractedMetadata model, we don't have per-field confidence
        # So we use a simpler heuristic based on data completeness

        # Check metadata completeness
        fields = ["distillery", "msrp", "name", "proof", "age"]
        filled = sum(1 for field in fields if hasattr(metadata, field) and getattr(metadata, field))

        # High confidence: all required fields populated
        if filled == len(fields):
            return "HIGH"
        # Medium confidence: most required fields populated
        if filled >= len(fields) - 1:
            return "MEDIUM"
        # Low confidence: few required fields populated
        return "LOW"

    def analyze_with_vision_model(
        self,
        image_paths: Path | Sequence[Path],
        ocr_text: str | Sequence[str] = "",
        model: str | None = None,
    ) -> ExtractedMetadata:
        """Read a bottle's label data from its photos with a vision model.

        Sends all photos in one ``chat_with_images`` call with the guarded
        prompt, the JSON schema and thinking disabled. An unusable answer
        (empty, truncated or without a name, distillery or release) is retried
        once.

        Args:
            image_paths: Photos of one bottle (a single Path is accepted).
            ocr_text: OCR text per photo, aligned with ``image_paths``, or one
                string used as a single hint.
            model: Vision model name (default: selector's, else
                ``DEFAULT_VISION_MODEL``).

        Returns:
            ExtractedMetadata with the label fields, ``vision_model``,
            ``vision_confidence`` and ``warnings``. With no photos, an empty
            ExtractedMetadata carrying only the OCR text.

        Raises:
            VisionResponseError: If both attempts returned unusable output.
            VisionTimeoutError: If the model did not answer in time.
            VisionAnalysisError: For any other per-bottle vision failure.
            VisionModelUnavailableError, OllamaConnectionError, ModelServerError:
                Propagated unchanged; no further bottle can succeed.
        """
        if isinstance(image_paths, (str, Path)):
            raw_paths: Sequence[Path | str] = [image_paths]
        else:
            raw_paths = list(image_paths)
        paths = [Path(p) for p in raw_paths if str(p) not in ("", ".")]

        if isinstance(ocr_text, str):
            hints = [ocr_text]
            combined_ocr = ocr_text
        else:
            hints = [str(h or "") for h in ocr_text]
            combined_ocr = "\n".join(h.strip() for h in hints if h and h.strip())

        if not paths:
            return ExtractedMetadata(full_text=combined_ocr, selected_text=combined_ocr, source_images=[])

        model = model or self._resolve_vision_model(None)
        names = ", ".join(p.name for p in paths)
        self.last_vision_result = None
        prompt = build_vision_prompt(len(paths), hints)

        extraction: VisionExtraction | None = None
        for attempt in range(1, VISION_RESPONSE_ATTEMPTS + 1):
            result = self._call_vision_model(prompt, paths, model, names)
            self.last_vision_result = result
            try:
                extraction = self._extract_from_result(result, model, names)
                break
            except VisionResponseError as e:
                if attempt >= VISION_RESPONSE_ATTEMPTS:
                    raise
                logger.warning("%s; retrying once", e)
        assert extraction is not None  # loop either breaks with a value or raises

        logger.info(
            "Vision %s: %d image(s), %s prompt tokens, %s output tokens, %.1fs, parsed as %s",
            model,
            len(paths),
            getattr(result, "prompt_eval_count", 0),
            getattr(result, "eval_count", 0),
            float(getattr(result, "total_duration_s", 0.0) or 0.0),
            extraction.parse_mode,
        )

        warnings: list[str] = []
        # A single photo cannot mix products; same_bottle is only meaningful
        # (and only asked for) when several photos are compared.
        same_bottle = extraction.same_bottle if len(paths) > 1 else None
        if same_bottle is False:
            warnings.append(MIXED_GROUP_WARNING)
            logger.warning("%s (%s)", MIXED_GROUP_WARNING, names)

        return ExtractedMetadata(
            **extraction.fields,
            full_text=combined_ocr,
            selected_text=combined_ocr,
            source_images=[str(p) for p in paths],
            vision_model=model,
            vision_confidence=extraction.confidence,
            warnings=warnings,
            same_bottle=same_bottle,
        )

    def _call_vision_model(self, prompt: str, paths: list[Path], model: str, names: str) -> Any:
        """One ``chat_with_images`` call with the standard vision request.

        Args:
            prompt: User prompt from ``build_vision_prompt``.
            paths: Photos to send.
            model: Vision model name.
            names: Photo file names, for error messages.

        Returns:
            The client's VisionChatResult.

        Raises:
            VisionAnalysisError: Client errors other than the fatal ones,
                wrapped with the photo names and model.
        """
        try:
            return self.ollama_client.chat_with_images(
                prompt=prompt,
                image_paths=paths,
                model=model,
                system=VISION_SYSTEM_PROMPT,
                format=vision_response_schema(len(paths)),
                think=False,
                options=dict(VISION_OPTIONS),
                max_image_edge=VISION_MAX_IMAGE_EDGE,
            )
        except FATAL_MODEL_ERRORS:
            raise
        except VisionAnalysisError:
            raise
        except Exception as e:
            raise VisionAnalysisError(
                f"Vision analysis failed for {names} with model '{model}': {type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def _extract_from_result(result: Any, model: str, names: str) -> VisionExtraction:
        """Parse a vision reply, rejecting empty, truncated or anonymous answers.

        Args:
            result: VisionChatResult from the client.
            model: Vision model name (for messages).
            names: Photo file names (for messages).

        Returns:
            The parsed extraction (has at least one identifying field).

        Raises:
            VisionResponseError: If the reply is unusable.
        """
        content = getattr(result, "content", "")
        content = content if isinstance(content, str) else ""
        done_reason = getattr(result, "done_reason", None)
        if not content.strip():
            thinking = getattr(result, "thinking", "")
            thinking_chars = len(thinking) if isinstance(thinking, str) else 0
            raise VisionResponseError(
                f"Vision model '{model}' returned no content "
                f"(done_reason={done_reason}, thinking_chars={thinking_chars})"
            )
        extraction = parse_vision_response(content)
        if done_reason == "length" and extraction.parse_mode != "json":
            raise VisionResponseError(
                f"Vision model '{model}' output was cut off (done_reason=length) for {names}: {content[:300]!r}"
            )
        if not extraction.has_identity:
            raise VisionResponseError(
                f"Vision model '{model}' returned no identifying fields for {names}: {content[:300]!r}"
            )
        return extraction

    def _estimate_quantity_from_images(self, images: list[Image]) -> int:
        """Estimate quantity from visual analysis of images.

        Uses MD5 hash grouping to detect multiple identical bottles.
        When user photographs multiple identical bottles from the same angle,
        each bottle's images share similar hashes within that group.

        Args:
            images: List of images for this bottle.

        Returns:
            Estimated quantity (minimum 1).
        """
        # Heuristic: Group images by MD5 hash to find repeated patterns
        # If we see 2-3 near-identical images (hash A), then another 2-3 near-identical (hash B),
        # this suggests 2 separate bottles each photographed multiple times from the same angle
        # Example: [A1, A2, A3, B1, B2, B3] → likely 2 bottles, 3 photos each

        detector = DuplicateDetector()
        hash_counts: dict[str, int] = {}

        # Count how many times each unique hash appears
        for img in images:
            try:
                img_hash = detector.generate_hash(Path(img.path))
                hash_counts[img_hash] = hash_counts.get(img_hash, 0) + 1
            except Exception:
                pass

        # If we have duplicate hashes beyond what's needed for multi-angle photos
        # (assume 3-4 angles per bottle), suggest larger quantity
        if hash_counts:
            max_count = max(hash_counts.values())
            if max_count > MAX_ANGLES_PER_BOTTLE:
                return max_count // MAX_ANGLES_PER_BOTTLE + 1

        return 1

    def _build_bottle_dict(
        self,
        metadata: ExtractedMetadata,
        images: list[Image],
    ) -> dict:
        """Build bottle dictionary from metadata.

        Args:
            metadata: ExtractedMetadata.
            images: List of images.

        Returns:
            Dictionary with bottle field values.
        """
        bottle_dict: dict = {}

        # Required fields with fallbacks
        if hasattr(metadata, "name") and metadata.name:
            bottle_dict["name"] = metadata.name
        else:
            # Generate name from distillery + release
            distillery = getattr(metadata, "distillery", "")
            release = getattr(metadata, "specific_release", "")
            if distillery and release:
                bottle_dict["name"] = f"{distillery} {release}"
            else:
                bottle_dict["name"] = "Unknown Bottle"

        if hasattr(metadata, "distillery") and metadata.distillery:
            bottle_dict["distillery"] = metadata.distillery
        else:
            bottle_dict["distillery"] = "Unknown Distillery"

        # None when no price was visible: unknown, never a $0.00 price.
        bottle_dict["msrp"] = getattr(metadata, "msrp", None) or None

        if hasattr(metadata, "specific_release") and metadata.specific_release:
            bottle_dict["specific_release"] = metadata.specific_release
        else:
            bottle_dict["specific_release"] = ""

        # Optional fields
        for field in ["proof", "abv", "age", "batch_number", "tasting_notes"]:
            if hasattr(metadata, field):
                value = getattr(metadata, field)
                if value:
                    bottle_dict[field] = value

        # Images (as strings, in the order given: primary first)
        bottle_dict["images"] = [str(img.path) for img in images]

        # Quantity (default to 1, should be updated later)
        bottle_dict["quantity"] = 1

        return bottle_dict

    def _parse_vision_response(self, response: str) -> dict:
        """Parse a vision model response into metadata field values.

        Thin wrapper kept for compatibility; see
        ``image_processor.vision_prompt.parse_vision_response``.

        Args:
            response: Vision model response string (JSON or ``key: value`` lines).

        Returns:
            Dictionary of extracted metadata fields.
        """
        return parse_vision_response(response).fields
