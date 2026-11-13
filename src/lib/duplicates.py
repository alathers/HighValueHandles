"""Duplicate image detection using MD5 hashing."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from lib.exceptions import DuplicateImageError

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Detect duplicate images using MD5 hashing.

    Tracks seen hashes and reports duplicates when encountered.

    Attributes:
        seen_hashes: Dict mapping MD5 hashes to image paths.
    """

    def __init__(self):
        """Initialize duplicate detector."""
        self.seen_hashes: Dict[str, Path] = {}

    def generate_hash(self, image_path: Path) -> str:
        """Generate MD5 hash for image file.

        Args:
            image_path: Path to image file.

        Returns:
            Hexadecimal MD5 hash string.

        Raises:
            FileNotFoundError: If image file not found.
            IOError: If file cannot be read.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def check_duplicate(
        self, image_path: Path, raise_on_duplicate: bool = True
    ) -> Tuple[str, bool, Optional[Path]]:
        """Check if image is duplicate of previously seen image.

        Args:
            image_path: Path to image file.
            raise_on_duplicate: Whether to raise DuplicateImageError on duplicate.

        Returns:
            Tuple of (md5_hash, is_duplicate, existing_path).
                - md5_hash: Generated MD5 hash.
                - is_duplicate: True if duplicate found, False otherwise.
                - existing_path: Path of existing duplicate image (None if not duplicate).

        Raises:
            DuplicateImageError: If duplicate detected and raise_on_duplicate=True.
            FileNotFoundError: If image file not found.
            IOError: If file cannot be read.
        """
        md5_hash = self.generate_hash(image_path)

        if md5_hash in self.seen_hashes:
            existing_path = self.seen_hashes[md5_hash]
            if raise_on_duplicate:
                raise DuplicateImageError(str(image_path), str(existing_path))
            return md5_hash, True, existing_path
        else:
            self.seen_hashes[md5_hash] = image_path
            return md5_hash, False, None

    def batch_check_duplicates(
        self, image_paths: list[Path], raise_on_duplicate: bool = True
    ) -> Dict[Path, Tuple[str, bool, Optional[Path]]]:
        """Check multiple images for duplicates.

        Args:
            image_paths: List of image paths to check.
            raise_on_duplicate: Whether to raise DuplicateImageError on first duplicate.

        Returns:
            Dict mapping image paths to tuples of (md5_hash, is_duplicate, existing_path).

        Raises:
            DuplicateImageError: If duplicate detected and raise_on_duplicate=True.
            FileNotFoundError: If any image file not found.
            IOError: If any file cannot be read.
        """
        results = {}
        for image_path in image_paths:
            results[image_path] = self.check_duplicate(image_path, raise_on_duplicate)
        return results

    def get_duplicates(self) -> Dict[str, list[Path]]:
        """Get all duplicates grouped by hash.

        Returns:
            Dict mapping MD5 hashes to lists of duplicate image paths.

        Format: {hash: [path1, path2, path3, ...]}

        Note: This method returns the original image plus all duplicates.
        For a list of only duplicate images (excluding originals), use
        get_duplicate_images() instead.
        """
        duplicates: Dict[str, list[Path]] = {}
        for md5_hash, path in self.seen_hashes.items():
            if md5_hash in duplicates:
                duplicates[md5_hash].append(path)
        return duplicates

    def get_duplicate_images(self, exclude_first: bool = False) -> Set[Path]:
        """Get set of duplicate image paths.

        Args:
            exclude_first: If True, exclude first occurrence (keep only duplicates).
                         If False, include all images that have duplicates.

        Returns:
            Set of image paths that are duplicates.

        Example with images A, B, C sharing hash and D unique:
        - exclude_first=True: {B, C} (only duplicates, exclude A)
        - exclude_first=False: {A, B, C} (all duplicates, include A)
        """
        duplicate_hashes = {h for h, paths in self.get_duplicates().items() if len(paths) > 1}

        if exclude_first:
            # Exclude first occurrence of each hash
            result: Set[Path] = set()
            for md5_hash in duplicate_hashes:
                paths = [p for p in self.seen_hashes if self.generate_hash(p) == md5_hash]
                if len(paths) > 1:
                    result.update(paths[1:])  # All except first
            return result
        else:
            # Include all images with duplicate hashes
            return {
                path
                for md5_hash, path in self.seen_hashes.items()
                if md5_hash in duplicate_hashes
            }

    def clear(self) -> None:
        """Clear all seen hashes."""
        self.seen_hashes.clear()

    def get_hash(self, image_path: Path) -> Optional[str]:
        """Get existing hash for image path.

        Args:
            image_path: Path to image file.

        Returns:
            MD5 hash string if image seen before, None otherwise.
        """
        md5_hash = self.generate_hash(image_path)
        return md5_hash if md5_hash in self.seen_hashes else None

    def is_duplicate(self, image_path: Path) -> bool:
        """Check if image is duplicate without storing.

        Args:
            image_path: Path to image file.

        Returns:
            True if duplicate, False otherwise.

        Raises:
            FileNotFoundError: If image file not found.
            IOError: If file cannot be read.
        """
        md5_hash = self.generate_hash(image_path)
        return md5_hash in self.seen_hashes

    def add_seen(self, md5_hash: str, image_path: Path) -> None:
        """Manually add seen hash and path mapping.

        Args:
            md5_hash: MD5 hash to add.
            image_path: Image path to associate with hash.
        """
        self.seen_hashes[md5_hash] = image_path

    def remove_seen(self, image_path: Path) -> bool:
        """Remove image from seen hashes.

        Args:
            image_path: Path to image file.

        Returns:
            True if removed, False if not found.

        Raises:
            FileNotFoundError: If image file not found.
            IOError: If file cannot be read.
        """
        md5_hash = self.generate_hash(image_path)
        if md5_hash in self.seen_hashes and self.seen_hashes[md5_hash] == image_path:
            del self.seen_hashes[md5_hash]
            return True
        return False

    def get_stats(self) -> Dict[str, int]:
        """Get duplicate detection statistics.

        Returns:
            Dict with stats:
                - total_seen: Total number of unique images seen
                - total_hashes: Total number of hashes tracked
                - duplicate_hashes: Number of hashes with duplicates
                - duplicate_images: Number of duplicate images
        """
        duplicate_hashes = {h for h, paths in self.get_duplicates().items() if len(paths) > 1}
        duplicate_images = self.get_duplicate_images(exclude_first=True)

        return {
            "total_seen": len(self.seen_hashes),
            "total_hashes": len(self.seen_hashes),
            "duplicate_hashes": len(duplicate_hashes),
            "duplicate_images": len(duplicate_images),
        }

    def __len__(self) -> int:
        """Return number of unique hashes tracked."""
        return len(self.seen_hashes)

    def __contains__(self, image_path: Path) -> bool:
        """Check if image path is in seen hashes.

        Args:
            image_path: Path to image file.

        Returns:
            True if image is a duplicate, False otherwise.

        Raises:
            FileNotFoundError: If image file not found.
            IOError: If file cannot be read.
        """
        return self.is_duplicate(image_path)
