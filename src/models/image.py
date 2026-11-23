"""Image model for tracking image file metadata and processing state."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

import json


class ImageCategory(Enum):
    """Categorized image types for processing and organization."""

    ANGLE = auto()
    """Primary bottle angle shot."""

    BOX = auto()
    """Box or packaging image."""

    DETAIL = auto()
    """Close-up detail shot."""

    VENDOR = auto()
    """Vendor reference image."""

    BILL = auto()
    """Receipt or bill image."""

    UNKNOWN = auto()
    """Unclassified image type."""


class ImageStatus(Enum):
    """Processing state tracking for images."""

    PENDING = auto()
    """Not yet processed."""

    IN_PROGRESS = auto()
    """Currently being processed."""

    COMPLETED = auto()
    """Successfully processed."""

    FAILED = auto()
    """Processing failed (error report generated)."""

    SKIPPED = auto()
    """Skipped (e.g., duplicate, not a bottle)."""


@dataclass
class Image:
    """Image file metadata and processing state.

    Attributes:
        path: Path to image file.
        category: Categorized image type.
        status: Processing state.
        bottle_ref: Reference to Bottle object (if assigned).
        md5_hash: MD5 hash for duplicate detection.
        file_size: File size in bytes.
        is_primary: Whether this is the primary image for the bottle.
        is_duplicate: Whether this image is a duplicate of another.
        duplicate_of_id: ID/path of the original image this duplicates.
        capture_time: Capture time in seconds (EXIF, as UTC epoch seconds),
            or None if unknown. Set by grouping.
        capture_time_source: Where capture_time came from ('exif_original',
            'exif_digitized', 'exif_datetime', 'mtime' or 'none').
        near_duplicate_of: Path of an almost identical photo taken just
            before this one (a burst retake), or None.
    """

    path: Path
    category: ImageCategory = ImageCategory.UNKNOWN
    status: ImageStatus = ImageStatus.PENDING
    bottle_ref: Optional[str] = None
    md5_hash: Optional[str] = None
    file_size: int = 0
    is_primary: bool = False
    is_duplicate: bool = False
    duplicate_of_id: Optional[str] = None
    capture_time: Optional[float] = None
    capture_time_source: str = "none"
    near_duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of image.
        """
        return {
            "path": str(self.path),
            "category": self.category.name,
            "status": self.status.name,
            "bottle_ref": self.bottle_ref,
            "md5_hash": self.md5_hash,
            "file_size": self.file_size,
            "is_primary": self.is_primary,
            "is_duplicate": self.is_duplicate,
            "duplicate_of_id": self.duplicate_of_id,
            "capture_time": self.capture_time,
            "capture_time_source": self.capture_time_source,
            "near_duplicate_of": self.near_duplicate_of,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Image":
        """Create Image from dictionary.

        Args:
            data: Dictionary representation of image.

        Returns:
            Image instance.
        """
        return cls(
            path=Path(data["path"]),
            category=ImageCategory[data["category"]],
            status=ImageStatus[data["status"]],
            bottle_ref=data.get("bottle_ref"),
            md5_hash=data.get("md5_hash"),
            file_size=data.get("file_size", 0),
            is_primary=data.get("is_primary", False),
            is_duplicate=data.get("is_duplicate", False),
            duplicate_of_id=data.get("duplicate_of_id"),
            capture_time=data.get("capture_time"),
            capture_time_source=data.get("capture_time_source", "none"),
            near_duplicate_of=data.get("near_duplicate_of"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Image":
        """Create Image from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            Image instance.

        Raises:
            json.JSONDecodeError: If JSON is invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __hash__(self) -> int:
        """Return hash based on path for set/dict operations."""
        return hash(str(self.path))

    def __eq__(self, other: object) -> bool:
        """Check equality based on path."""
        if not isinstance(other, Image):
            return False
        return self.path == other.path


def group_bottle_id(group: "list[Image]") -> str:
    """Bottle ID of an image group: its assigned ``bottle_ref``, else the first photo's stem."""
    if not group:
        return "unknown"
    return group[0].bottle_ref or Path(group[0].path).stem
