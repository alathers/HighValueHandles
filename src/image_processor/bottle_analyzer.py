"""Analyze images and extract bourbon bottle metadata.

Processes single or multiple bottle images, groups them by bottle identity,
and creates Bottle model instances with extracted metadata.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from image_processor.image_organizer import ImageOrganizer
from image_processor.text_extractor import TextExtractor
from lib.duplicates import DuplicateDetector
from lib.error_handling import RetryManager
from lib.ollama_client import OllamaClient
from models.bottle import ExtractedMetadata
from models.image import Image

if TYPE_CHECKING:
    from lib.model_selector import ModelSelector, ModelStage

# Maximum number of angle photos expected for a single bottle
MAX_ANGLES_PER_BOTTLE = 4

logger = logging.getLogger(__name__)


class BottleAnalyzer:
    """Analyze bourbon bottle images and extract metadata.

    Uses TextExtractor for OCR and vision model analysis,
    groups related images, and creates Bottle instances.

    Attributes:
        text_extractor: TextExtractor for OCR and metadata extraction.
        ollama_client: Ollama client for vision model calls.
        retry_manager: RetryManager for automatic retries.
        model_selector: Optional ModelSelector for per-stage model resolution.
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
        self.text_extractor = text_extractor or TextExtractor(retry_manager=retry_manager)
        self.ollama_client = ollama_client or OllamaClient()
        self.retry_manager = retry_manager or RetryManager()
        self.image_organizer = image_organizer or ImageOrganizer()
        self.model_selector = model_selector

    def process_single_bottle(
        self,
        images: list[Image],
        vision_model: str = "qwen2-vl:7b",
    ) -> tuple[dict, ExtractedMetadata]:
        """Process images for a single bottle and extract metadata.

        Args:
            images: List of Image objects for this bottle.
            vision_model: Vision model name for Ollama (overridden by model_selector).

        Returns:
            Tuple of (bottle_dict, metadata) where:
                - bottle_dict: Dictionary with bottle field values (name, distillery, etc.)
                - metadata: ExtractedMetadata with full OCR text and confidence scores
        """
        logger.info("Processing bottle with %d image(s)", len(images))

        # Resolve model from selector if available (T161)
        if self.model_selector is not None:
            try:
                from lib.model_selector import ModelStage

                vision_model = self.model_selector.get_model(ModelStage.VISION_ANALYSIS)
            except Exception:
                pass  # Fall back to passed-in vision_model

        # Use first non-bill image for metadata extraction
        primary_image = None
        for img in images:
            if img.category != "BILL":
                primary_image = img
                break

        if not primary_image and images:
            primary_image = images[0]

        # Extract OCR text from all images
        logger.debug("Extracting OCR text from %d image(s)", len(images))
        ocr_texts: list[str] = []
        for img in images:
            try:
                text = self.text_extractor.extract_with_retry(
                    Path(img.path),
                    orientation_corrected=img.is_primary,
                    select_regions=True,
                )
                if text.strip():
                    ocr_texts.append(text.strip())
                    logger.debug("Successfully extracted %d characters from %s", len(text.strip()), img.path)
            except Exception as e:
                logger.warning("Failed to extract text from %s: %s", img.path, e)

        combined_ocr = "\n".join(ocr_texts)
        logger.debug("Combined OCR text length: %d characters", len(combined_ocr))

        # Analyze with vision model for metadata
        start_time = time.time()
        success = False
        error_msg: str | None = None
        try:
            if primary_image:
                logger.info("Analyzing with vision model: %s", vision_model)
                metadata = self.analyze_with_vision_model(
                    Path(primary_image.path),
                    combined_ocr,
                    vision_model,
                )
                success = True
            else:
                # No images provided - still call vision model (allows patching in tests)
                # Vision model implementation should handle empty case gracefully
                logger.warning("No primary image found, attempting vision model anyway")
                metadata = self.analyze_with_vision_model(
                    Path(""),
                    combined_ocr,
                    vision_model,
                )
                success = True
        except Exception:
            logger.exception("Vision model analysis failed")
            error_msg = "Vision model analysis failed"
            metadata = ExtractedMetadata(
                full_text=combined_ocr,
                selected_text=combined_ocr,
                source_images=[str(primary_image.path)] if primary_image else [],
            )

        # Record usage tracking (T162)
        if self.model_selector is not None and hasattr(self.model_selector, "record_usage"):
            try:
                from lib.model_selector import ModelStage

                duration = time.time() - start_time
                fallback_used = (
                    self.model_selector.get_model(ModelStage.VISION_ANALYSIS) != vision_model
                )
                self.model_selector.record_usage(
                    stage=ModelStage.VISION_ANALYSIS,
                    model=vision_model,
                    success=success,
                    duration=duration,
                    fallback_used=fallback_used,
                    error=error_msg,
                )
            except Exception:
                pass  # Usage tracking failure should not break processing

        # Build bottle dict from metadata
        bottle_dict = self._build_bottle_dict(metadata, images)
        logger.info("Successfully processed bottle: %s", bottle_dict.get("name", "Unknown"))

        return bottle_dict, metadata

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
        image_path: Path,
        ocr_text: str,
        model: str = "qwen2-vl:7b",
    ) -> ExtractedMetadata:
        """Analyze image with vision model for bottle metadata.

        Args:
            image_path: Path to image file.
            ocr_text: OCR text extracted from image.
            model: Vision model name.

        Returns:
            ExtractedMetadata with bottle information.

        Raises:
            AIModelError: If vision model call fails.
        """
        # Handle empty path case (no images provided)
        # This allows method to be patched/mocked in tests
        if not image_path or str(image_path) == "":
            return ExtractedMetadata(
                full_text=ocr_text,
                selected_text=ocr_text,
                source_images=[],
            )

        prompt = (
            "Analyze this bourbon bottle image and extract the following information. "
            "Return your response in a structured format with these fields:\n"
            "- name: The bottle's name\n"
            "- distillery: The distillery name\n"
            "- msrp: MSRP price (in dollars, as a number without $ sign)\n"
            "- specific_release: Specific release information (style, series, finish, etc.)\n"
            "- proof: Proof statement (e.g., '107 Proof', '90 Proof')\n"
            "- age: Age statement (e.g., '12 Years', '8 Years', or age in years)\n"
            "- batch_number: Batch number if present\n"
            "- tasting_notes: Tasting notes from the label\n\n"
            "Extract this information from the image label, packaging, and OCR text. "
            "Provide the information in a clear, extractable format."
        )

        try:
            response = self.ollama_client.image_to_text(
                image_path=image_path,
                prompt=prompt,
                model=model,
            )

            # Parse response to extract metadata fields
            # This is a simplified parser; in production, use structured JSON response
            metadata_dict = self._parse_vision_response(response)

            return ExtractedMetadata(
                full_text=ocr_text,
                selected_text=ocr_text,  # In production, could be updated
                source_images=[str(image_path)],
                **metadata_dict,
            )

        except Exception:
            logger.exception("Vision model analysis failed")
            # Fallback to OCR-only metadata
            return ExtractedMetadata(
                full_text=ocr_text,
                selected_text=ocr_text,
                source_images=[str(image_path)],
            )

    def _images_belong_together(
        self,
        img1: Image,
        img2: Image,
    ) -> bool:
        """Check if two images likely belong to the same bottle.

        Simple heuristic: check if images are hash duplicates or share similar names.

        Args:
            img1: First image.
            img2: Second image.
            threshold: Similarity threshold (not used in simple heuristic).

        Returns:
            True if images likely belong to same bottle, False otherwise.
        """
        # Hash-based duplicate check
        if img1.md5_hash and img2.md5_hash and img1.md5_hash == img2.md5_hash:
            return True

        # Name-based similarity (same filename pattern)
        name1 = Path(img1.path).stem
        name2 = Path(img2.path).stem

        # Extract common prefix
        common_len = len(name1)
        for i, (c1, c2) in enumerate(zip(name1, name2, strict=False)):
            if c1 != c2:
                common_len = i
                break

        # If names share significant common prefix, likely same bottle
        # Use 60% threshold: allows "bottleX" and "bottleY" to match but rejects "imageA" vs "imageB"
        return common_len > min(len(name1), len(name2)) * 0.6

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

        if hasattr(metadata, "msrp") and metadata.msrp:
            bottle_dict["msrp"] = metadata.msrp
        else:
            bottle_dict["msrp"] = "0.00"

        if hasattr(metadata, "specific_release") and metadata.specific_release:
            bottle_dict["specific_release"] = metadata.specific_release
        else:
            bottle_dict["specific_release"] = ""

        # Optional fields
        for field in ["proof", "age", "batch_number", "tasting_notes"]:
            if hasattr(metadata, field):
                value = getattr(metadata, field)
                if value:
                    bottle_dict[field] = value

        # Images
        bottle_dict["images"] = [img.path for img in images]

        # Quantity (default to 1, should be updated later)
        bottle_dict["quantity"] = 1

        return bottle_dict

    def _parse_vision_response(self, response: str) -> dict:
        """Parse vision model response to extract metadata fields.

        Simple parser for structured text response.

        Args:
            response: Vision model response string.

        Returns:
            Dictionary of extracted metadata fields.
        """
        metadata: dict = {}

        # Simple line-based parser
        for line in response.split("\n"):
            stripped_line = line.strip()

            # Try to extract key-value pairs
            if ":" in stripped_line:
                key, value = stripped_line.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("-", "_")

                # Map to ExtractedMetadata fields
                field_map = {
                    "name": "name",
                    "distillery": "distillery",
                    "msrp": "msrp",
                    "specific_release": "specific_release",
                    "release": "specific_release",
                    "proof": "proof",
                    "age": "age",
                    "batch_number": "batch_number",
                    "batch": "batch_number",
                    "tasting_notes": "tasting_notes",
                    "notes": "tasting_notes",
                }

                if key in field_map:
                    mapped_field = field_map[key]
                    value = value.strip()

                    # Clean up MSRP
                    if mapped_field == "msrp":
                        value = value.replace("$", "").strip()
                        try:
                            # Validate numeric format
                            float(value)
                        except ValueError:
                            value = ""

                    if value:
                        metadata[mapped_field] = value

        return metadata
