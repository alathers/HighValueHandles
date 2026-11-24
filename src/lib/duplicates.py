"""MD5 file hashing for duplicate-photo detection."""

import hashlib
from pathlib import Path


class DuplicateDetector:
    """Hash image files so byte-identical copies can be recognised.

    Only an MD5 helper today: grouping marks exact duplicates and quantity
    counts them. Kept for skipping already-processed photos on re-runs.
    """

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
