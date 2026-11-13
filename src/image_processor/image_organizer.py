"""Organize and group images by bottle identity.

Scans image folders, detects duplicates, groups related images,
and prepares bottle image sets for processing.
"""

import logging
from pathlib import Path
from typing import ClassVar

from lib.duplicates import DuplicateDetector
from models.image import Image

logger = logging.getLogger(__name__)


class ImageOrganizer:
    """Organize bourbon bottle images for processing.

    Scans input folders, detects duplicates, groups related images,
    and prepares image sets for bottle analysis.

    Attributes:
        supported_extensions: Set of supported image file extensions.
        duplicate_detector: DuplicateDetector for hash-based duplicate detection.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".jpeg", ".jpg", ".png", ".tiff", ".heic", ".webp", ".bmp"}

    def __init__(
        self,
        duplicate_detector: DuplicateDetector | None = None,
    ):
        """Initialize image organizer.

        Args:
            duplicate_detector: Optional DuplicateDetector instance.
        """
        self.duplicate_detector = duplicate_detector or DuplicateDetector()

    def scan_images(
        self,
        folder: Path,
        recursive: bool = True,
    ) -> list[Image]:
        """Recursively scan folder for supported image formats.

        Args:
            folder: Path to folder to scan.
            recursive: Whether to scan subdirectories recursively.

        Returns:
            List of Image objects with filepath populated.
        """
        if not folder.exists() or not folder.is_dir():
            logger.warning("Folder does not exist or is not a directory: %s", folder)
            return []

        images: list[Image] = []

        # Recursively or flat scan based on flag
        pattern = "**/*" if recursive else "*"
        for path in folder.glob(pattern):
            # Skip directories
            if path.is_dir():
                continue

            # Check extension
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    # Create Image object
                    img = Image(path=str(path))

                    # Generate hash
                    img.md5_hash = self.duplicate_detector.generate_hash(path)

                    # Get file size
                    img.file_size = path.stat().st_size

                    images.append(img)
                    logger.debug("Found image: %s", path)
                except Exception as e:
                    logger.warning("Failed to process image %s: %s", path, e)

        logger.info("Scanned %d images from %s", len(images), folder)
        return images

    def group_by_bottle(
        self,
        images: list[Image],
        similarity_threshold: float = 0.8,
    ) -> list[list[Image]]:
        """Group images that belong to the same bottle.

        Uses hash-based duplicate detection and filename similarity
        to group related images (multi-angle photos of same bottle).

        Args:
            images: List of Image objects to group.
            similarity_threshold: Minimum similarity threshold for grouping (0.0-1.0).

        Returns:
            List of image groups, where each group is a list of Image objects
                for the same bottle.
        """
        if not images:
            return []

        if len(images) == 1:
            return [images]

        # Detect duplicates first
        self.detect_duplicates(images)

        # Group by hash (exact duplicates first)
        hash_groups: dict[str, list[Image]] = {}
        for img in images:
            if img.md5_hash:
                if img.md5_hash not in hash_groups:
                    hash_groups[img.md5_hash] = []
                hash_groups[img.md5_hash].append(img)
            else:
                # Images without hashes get their own group
                hash_groups[f"no_hash_{img.path}"] = [img]

        # Now group related hashes together based on visual similarity
        # For simplicity, we use filename similarity as a proxy
        groups: list[list[Image]] = []

        for hash_images in hash_groups.values():
            # Use first image from hash group as representative
            representative = hash_images[0]

            # Try to find existing group to join
            assigned = False
            for group in groups:
                # Check if similar to any image in existing group
                for group_img in group:
                    if self._images_similar(representative, group_img, similarity_threshold):
                        group.extend(hash_images)
                        assigned = True
                        break

                if assigned:
                    break

            # Create new group if not assigned
            if not assigned:
                groups.append(hash_images)

        # Log grouping results
        logger.info("Grouped %d images into %d bottle groups", len(images), len(groups))
        for i, group in enumerate(groups, 1):
            logger.debug("Group %d: %d images", i, len(group))

        return groups

    def detect_duplicates(
        self,
        images: list[Image],
    ) -> None:
        """Detect exact duplicate images using file hash comparison.

        Updates Image.is_duplicate and Image.duplicate_of_id fields.

        Args:
            images: List of Image objects to check for duplicates.
        """
        hash_to_images: dict[str, list[Image]] = {}

        # Group by hash
        for img in images:
            if img.md5_hash:
                if img.md5_hash not in hash_to_images:
                    hash_to_images[img.md5_hash] = []
                hash_to_images[img.md5_hash].append(img)

        # Mark duplicates
        duplicate_count = 0
        for hash_images in hash_to_images.values():
            if len(hash_images) > 1:
                # First image is the original, rest are duplicates
                original = hash_images[0]
                for duplicate in hash_images[1:]:
                    duplicate.is_duplicate = True
                    duplicate.duplicate_of_id = original.path
                    duplicate_count += 1

        logger.info("Found %d duplicate images", duplicate_count)

    def is_duplicate_image(
        self,
        img1: Image,
        img2: Image,
    ) -> bool:
        """Check if two images are exact duplicates by comparing file content hash.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            True if images are exact duplicates, False otherwise.
        """
        if not img1.md5_hash or not img2.md5_hash:
            return False

        return img1.md5_hash == img2.md5_hash

    def _images_similar(
        self,
        img1: Image,
        img2: Image,
        threshold: float = 0.6,
    ) -> bool:
        """Check if two images are similar using filename heuristics.

        This is a simplified similarity check based on filename patterns.
        Groups multi-angle shots of the same bottle (e.g., "IMG_1234_front",
        "IMG_1234_back") while avoiding false positives on distinct bottles.

        Args:
            img1: First image.
            img2: Second image.
            threshold: Minimum similarity percentage for grouping (0.0-1.0).

        Returns:
            True if images appear similar, False otherwise.
        """
        # Exact match (duplicate)
        if self.is_duplicate_image(img1, img2):
            return True

        # Check same directory
        path1 = Path(img1.path).parent
        path2 = Path(img2.path).parent
        if path1 != path2:
            return False

        # Extract basename
        name1 = Path(img1.path).stem.lower()
        name2 = Path(img2.path).stem.lower()

        # List of common suffixes to strip for comparison
        # These indicate multi-angle shots of same item
        # Use underscore prefixes to avoid stripping just digits
        suffixes = ["_front", "_back", "_left", "_right", "_top", "_bottom",
                   "_side", "_detail", "_box", "_label", "_cap"]

        # Strip suffixes to get base name
        base1 = name1
        base2 = name2

        # Only strip suffixes starting with underscore to avoid false matches on digits
        for suffix in suffixes:
            if base1.endswith(suffix):
                base1 = base1[:-len(suffix)]
            if base2.endswith(suffix):
                base2 = base2[:-len(suffix)]

        # If bases match exactly after stripping suffixes, likely same bottle
        if base1 == base2 and len(base1) > 0:
            return True

        # Otherwise compare original names with stricter criteria
        # Calculate common prefix length
        common_len = 0
        for c1, c2 in zip(name1, name2, strict=False):
            if c1 == c2:
                common_len += 1
            else:
                break

        # Use max length to avoid matching very short common prefixes
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return False

        # Calculate matching percentage
        match_pct = common_len / max_len

        # For short names (< 8 chars), require near-perfect match to avoid false positives
        # For longer names, use the provided threshold
        required_threshold = 0.9 if max_len < 8 else threshold

        # Also require minimum absolute length to avoid false positives on very short names
        min_match_len = min(5, max_len)  # At least 5 chars for very short names

        if common_len >= min_match_len and match_pct >= required_threshold:
            return True

        return False

    def filter_duplicate_images(
        self,
        images: list[Image],
        keep_original: bool = True,
    ) -> list[Image]:
        """Filter out duplicate images, keeping only one copy.

        Args:
            images: List of Image objects to filter.
            keep_original: If True, keep the first occurrence as original.

        Returns:
            Filtered list of images with duplicates removed.
        """
        seen_hashes: set[str] = set()
        filtered: list[Image] = []

        for img in images:
            if not img.md5_hash:
                # No hash available, keep it
                filtered.append(img)
                continue

            if img.md5_hash not in seen_hashes:
                # First occurrence
                seen_hashes.add(img.md5_hash)
                filtered.append(img)
            elif not keep_original:
                # If we want to keep the "best" version, we could implement
                # logic to keep highest resolution, etc.
                # For now, just skip duplicates
                pass

        removed = len(images) - len(filtered)
        if removed > 0:
            logger.info("Filtered out %d duplicate images", removed)

        return filtered
