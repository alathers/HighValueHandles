"""Process bourbon bottles in configurable batches.

Handles batch slicing, lookahead for multi-angle photos, batch processing,
and user feedback integration for iterative refinement.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from image_processor.bottle_analyzer import BottleAnalyzer
from models.bottle import Correction, ExtractedMetadata
from models.image import Image

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a single batch.

    Attributes:
        batch_id: Sequential batch identifier (1, 2, 3...).
        bottle_dicts: List of processed bottle dictionaries.
        metadata_list: List of extracted metadata for each bottle.
        corrections: List of user corrections applied to this batch.
        failed_bottles: List of bottle IDs that failed processing.
    """

    batch_id: int
    bottle_dicts: list[dict] = field(default_factory=list)
    metadata_list: list[ExtractedMetadata] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    failed_bottles: list[str] = field(default_factory=list)


class BatchProcessor:
    """Process bourbon bottles in configurable batches.

    Manages batch slicing, lookahead for multi-angle photo grouping,
    batch processing orchestration, and user feedback integration.

    Attributes:
        batch_size: Number of bottles per batch, or "all" for unbatched.
        lookahead: Number of images to peek ahead for grouping.
        bottle_analyzer: BottleAnalyzer instance for processing.
    """

    def __init__(
        self,
        batch_size: int | str = 5,
        lookahead: int = 3,
        bottle_analyzer: Optional[BottleAnalyzer] = None,
        similarity_threshold: float = 0.7,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Number of bottles per batch, or "all" for unbatched.
            lookahead: Number of images to peek ahead for multi-angle grouping.
            bottle_analyzer: Optional BottleAnalyzer instance.
            similarity_threshold: Minimum similarity score for grouping (0.0-1.0).
        """
        if batch_size == "all":
            self.batch_size = batch_size
        else:
            self.batch_size = int(batch_size)
        self.lookahead = lookahead
        self.bottle_analyzer = bottle_analyzer or BottleAnalyzer()
        self.similarity_threshold = similarity_threshold

        logger.debug(
            "Initialized BatchProcessor: batch_size=%s, lookahead=%d, similarity_threshold=%.2f",
            self.batch_size,
            self.lookahead,
            self.similarity_threshold,
        )

    def slice_into_batches(
        self,
        image_groups: list[list[Image]],
    ) -> list[list[list[Image]]]:
        """Split bottle image groups into batches.

        Args:
            image_groups: List of bottle image groups from ImageOrganizer.

        Returns:
            List of batches, where each batch is a list of image groups.
        """
        if self.batch_size == 0 or str(self.batch_size).lower() == "all":
            logger.info("Batch mode: processing all images in single batch")
            return [image_groups]

        batches: list[list[list[Image]]] = []

        for i in range(0, len(image_groups), self.batch_size):
            batch = image_groups[i : i + self.batch_size]
            batches.append(batch)
            logger.debug("Created batch %d: %d image groups", len(batches), len(batch))

        logger.info("Split %d image groups into %d batches", len(image_groups), len(batches))
        return batches

    def apply_lookahead(
        self,
        current_batch: list[list[Image]],
        remaining_groups: list[list[Image]],
    ) -> tuple[list[list[Image]], list[list[Image]]]:
        """Peek ahead to include multi-angle photos in current batch.

        Problem to solve: When photographing a bottle, users often take multiple
        angles (front, back, label close-up) in sequence. Without lookahead,
        these might get split across batch boundaries. This matches each
        bottle's multi-angle shots together during processing.

        Algorithm:
        1. Peek at next N image groups (lookahead window)
        2. Compare each peeked group to current batch groups using visual similarity
        3. Move visually similar groups into current batch (they're likely same bottle)
        4. Return augmented batch and updated remaining groups

        Args:
            current_batch: Current batch of image groups.
            remaining_groups: Remaining image groups not yet processed.

        Returns:
            Tuple of (augmented_current_batch, updated_remaining_groups).
        """
        if not current_batch or not remaining_groups:
            return current_batch, remaining_groups

        # Peek at next N groups (sliding window into the unprocessed queue)
        # This is the lookahead window: we check these groups for similarity
        peek_groups = remaining_groups[: self.lookahead]

        if not peek_groups:
            return current_batch, remaining_groups

        logger.debug("Applying lookahead: checking %d next groups", len(peek_groups))

        # For each group in lookahead, check if similar to current batch
        # Track indices of groups to move (must be done in reverse order later)
        groups_to_move: list[int] = []

        for i, candidate_group in enumerate(peek_groups):
            # Use first image of each group as representative for comparison
            # This assumes images within a group are all of the same bottle
            candidate_rep = candidate_group[0] if candidate_group else None
            if not candidate_rep:
                continue

            # Compare against all groups in current batch
            for batch_group in current_batch:
                batch_rep = batch_group[0] if batch_group else None
                if not batch_rep:
                    continue

                # Check visual similarity using ORB feature matching
                # Similarity > threshold → likely same bottle, move to current batch
                similarity = self.vision_similarity_score(candidate_rep.path, batch_rep.path)

                # If similar (>threshold), move to current batch
                # Once matched, stop checking other batch groups for this candidate
                if similarity > self.similarity_threshold:
                    logger.debug(
                        "Found similar group (sim=%.2f): moving to current batch",
                        similarity,
                    )
                    groups_to_move.append(i)
                    break

        # Move identified groups from remaining to current batch
        if groups_to_move:
            augmented_batch = current_batch.copy()
            updated_remaining = remaining_groups.copy()

            # Move in reverse order to maintain indices
            # If we pop index 2 first, index 3 becomes index 2, etc.
            # Sorting reverse ensures we process higher indices first
            for idx in sorted(groups_to_move, reverse=True):
                group_to_move = remaining_groups[idx]
                augmented_batch.append(group_to_move)
                updated_remaining.pop(idx)

            logger.info(
                "Lookahead moved %d groups to batch %d",
                len(groups_to_move),
                id(current_batch),
            )
            return augmented_batch, updated_remaining

        return current_batch, remaining_groups

    def vision_similarity_score(self, img1_path: str, img2_path: str) -> float:
        """Compare two images using OpenCV ORB features.

        Algorithm (ORB feature matching):
        1. Detect ORB (Oriented FAST and Rotated BRIEF) keypoints in each image
        2. Compute binary descriptors for each keypoint (256-bit binary vectors)
        3. Match descriptors using Hamming distance (count of differing bits)
        4. Lower average distance = more similar features = higher similarity
        5. Convert distance to 0-1 similarity score

        ORB is chosen because:
        - Fast and efficient (uses binary descriptors instead of floating-point)
        - Rotation-invariant (handles photos taken from different angles)
        - Scale-invariant (handles different camera distances)

        Args:
            img1_path: Path to first image.
            img2_path: Path to second image.

        Returns:
            Similarity score between 0 (different) and 1 (identical).
        """
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))

            if img1 is None or img2 is None:
                logger.warning("Failed to load images for similarity: %s or %s", img1_path, img2_path)
                return 0.0

            # Convert to grayscale for feature detection
            # ORB works on grayscale images (faster, no color channel overhead)
            grayscale1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            grayscale2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Initialize ORB (Oriented FAST and Rotated BRIEF) detector
            # ORB finds keypoints (distinctive corners/edges) and computes descriptors
            orb = cv2.ORB_create()

            # Detect keypoints and compute 256-bit binary descriptors for each image
            # Returns: list of Keypoint objects and numpy array of descriptors
            kp1, des1 = orb.detectAndCompute(grayscale1, None)
            kp2, des2 = orb.detectAndCompute(grayscale2, None)

            # Need at least 5 descriptors for reliable matching
            # Fewer matches may indicate featureless images or poor quality
            if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
                return 0.0

            # Match features using Brute-Force Matcher with Hamming distance
            # NORM_HAMMING for binary descriptors (XOR + count bits)
            # crossCheck=True ensures matches are symmetric (A↔B only if best in both directions)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # No matches found → completely different images
            if not matches:
                return 0.0

            # Calculate average Hamming distance across all matches
            # Each match.distance is the Hamming distance between descriptor pairs
            # Lower distance = more similar descriptors (fewer differing bits)
            avg_distance = sum(match.distance for match in matches) / len(matches)

            # Convert distance to similarity (0-1)
            # ORB descriptors are 256 bits, so maximum Hamming distance = 256
            # identical images: avg_distance = 0 → similarity = 1.0
            # completely different: avg_distance = 256 → similarity = 0.0
            similarity = 1 - (avg_distance / 256.0)

            return float(similarity)

        except Exception:
            logger.exception("Error computing vision similarity: %s, %s", img1_path, img2_path)
            return 0.0

    def process_batch(
        self,
        batch: list[list[Image]],
        vision_model: str = "qwen2-vl:7b",
    ) -> BatchResult:
        """Process one batch of bottle image groups.

        Args:
            batch: List of image groups to process.
            vision_model: Vision model name for Ollama.

        Returns:
            BatchResult with processed bottles and metadata.
        """
        batch_id = id(batch)
        logger.info("Processing batch with %d image groups", len(batch))

        result = BatchResult(batch_id=batch_id)

        for i, image_group in enumerate(batch, 1):
            # Use first image for bottle ID in logs
            bottle_id = Path(image_group[0].path).stem
            logger.debug("Processing bottle %d/%d: %s", i, len(batch), bottle_id)

            try:
                # Process single bottle with all images in group
                bottle_dict, metadata = self.bottle_analyzer.process_single_bottle(
                    image_group,
                    vision_model=vision_model,
                )

                result.bottle_dicts.append(bottle_dict)
                result.metadata_list.append(metadata)
                logger.info("Successfully processed bottle: %s", bottle_id)

            except Exception:
                logger.exception("Failed to process bottle: %s", bottle_id)
                result.failed_bottles.append(bottle_id)

        logger.info(
            "Batch processing complete: %d successful, %d failed",
            len(result.bottle_dicts),
            len(result.failed_bottles),
        )

        return result

    def learn_from_feedback(
        self,
        corrections: list[Correction],
    ) -> None:
        """Learn from user corrections to improve future grouping.

        Analyzes user corrections to adjust heuristics for subsequent batches.
        Adjusts similarity_threshold based on merge/split correction patterns.

        Args:
            corrections: List of user corrections from feedback.
        """
        if not corrections:
            return

        logger.info("Learning from %d corrections", len(corrections))

        # Analyze correction patterns
        merge_corrections = [
            c for c in corrections if c.field_name == "grouping" and c.corrected_value == "merge"
        ]
        split_corrections = [
            c for c in corrections if c.field_name == "grouping" and c.corrected_value == "split"
        ]

        # Adjust similarity threshold based on correction patterns
        merge_count = len(merge_corrections)
        split_count = len(split_corrections)

        if merge_count > split_count:
            # Too many splits - we're being too aggressive about grouping
            # Tighten threshold to reduce false positives
            adjustment = 0.05 * min(merge_count, 3)  # Cap at 0.15 adjustment
            new_threshold = min(1.0, self.similarity_threshold + adjustment)
            logger.info(
                "Tightening similarity threshold: %.2f -> %.2f (based on %d merge corrections)",
                self.similarity_threshold,
                new_threshold,
                merge_count,
            )
            self.similarity_threshold = new_threshold
        elif split_count > merge_count:
            # Too many merges - we're being too conservative about grouping
            # Loosen threshold to catch more related items
            adjustment = 0.05 * min(split_count, 3)  # Cap at 0.15 adjustment
            new_threshold = max(0.3, self.similarity_threshold - adjustment)
            logger.info(
                "Loosening similarity threshold: %.2f -> %.2f (based on %d split corrections)",
                self.similarity_threshold,
                new_threshold,
                split_count,
            )
            self.similarity_threshold = new_threshold
        else:
            logger.debug(
                "Balanced feedback: %d merges, %d splits - no threshold change",
                merge_count,
                split_count,
            )

    def apply_corrections(
        self,
        bottle_dicts: list[dict],
        corrections: list[Correction],
    ) -> list[dict]:
        """Apply user corrections to bottle dictionaries.

        Args:
            bottle_dicts: List of bottle dictionaries from batch result.
            corrections: List of corrections to apply.

        Returns:
            Updated list of bottle dictionaries with corrections applied.
        """
        if not corrections:
            return bottle_dicts

        logger.info("Applying %d corrections to %d bottles", len(corrections), len(bottle_dicts))

        # Apply corrections to matching bottles
        for correction in corrections:
            # Find bottle by path or name (simplified matching)
            matching_bottle = self._find_matching_bottle(bottle_dicts, correction)

            if matching_bottle:
                # Apply field correction
                original_value = matching_bottle.get(correction.field_name)
                matching_bottle[correction.field_name] = correction.corrected_value

                # Store correction history
                if "corrections" not in matching_bottle:
                    matching_bottle["corrections"] = []

                matching_bottle["corrections"].append(
                    {
                        "field_name": correction.field_name,
                        "original_value": original_value,
                        "corrected_value": correction.corrected_value,
                        "corrected_at": correction.corrected_at.isoformat() if correction.corrected_at else None,
                    }
                )

                logger.info(
                    "Applied correction to bottle: %s = %s (was %s)",
                    correction.field_name,
                    correction.corrected_value,
                    original_value,
                )

        return bottle_dicts

    def _find_matching_bottle(
        self,
        bottle_dicts: list[dict],
        correction: Correction,
    ) -> Optional[dict]:
        """Find matching bottle dictionary for a correction.

        Simple matching based on bottle path or name. In production,
        could use more sophisticated matching.

        Args:
            bottle_dicts: List of bottle dictionaries.
            correction: Correction to match.

        Returns:
            Matching bottle dictionary or None.
        """
        # Try to match by path if available in correction metadata
        if hasattr(correction, "bottle_path"):
            for bottle in bottle_dicts:
                if "images" in bottle and correction.bottle_path in bottle["images"]:
                    return bottle

        # Try to match by name
        if hasattr(correction, "bottle_name"):
            for bottle in bottle_dicts:
                if bottle.get("name") == correction.bottle_name:
                    return bottle

        # If no explicit identifiers, match by field value
        # First find any bottle where the field matches the original_value
        for bottle in bottle_dicts:
            if bottle.get(correction.field_name) == correction.original_value:
                return bottle

        # Also try to match where the field matches the corrected_value
        # This handles sequential corrections on the same field
        for bottle in bottle_dicts:
            if bottle.get(correction.field_name) == correction.corrected_value:
                return bottle

        return None
