"""Image model for tracking image file metadata and grouping state."""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class ImageCategory(Enum):
    """Categorized image types for processing and organization.

    Scanning currently leaves every photo UNKNOWN. VENDOR marks the vendor
    reference photo a ``-v`` run copies to ``images/vendor_ref.<ext>``; BILL
    is kept for receipts (evidence of the price paid, never of MSRP).
    """

    ANGLE = auto()
    """Primary bottle angle shot."""

    BOX = auto()
    """Box or packaging image."""

    DETAIL = auto()
    """Close-up detail shot."""

    VENDOR = auto()
    """Vendor reference image (``images/vendor_ref.<ext>``): a product photo from a
    vendor page that the vision model judged to show the same bottle. It is
    for the reviewer only and is not licensed for publication."""

    BILL = auto()
    """Receipt or bill image."""

    UNKNOWN = auto()
    """Unclassified image type."""


@dataclass
class Image:
    """Image file metadata and grouping state.

    Attributes:
        path: Path to image file.
        category: Categorized image type.
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
    bottle_ref: Optional[str] = None
    md5_hash: Optional[str] = None
    file_size: int = 0
    is_primary: bool = False
    is_duplicate: bool = False
    duplicate_of_id: Optional[str] = None
    capture_time: Optional[float] = None
    capture_time_source: str = "none"
    near_duplicate_of: Optional[str] = None

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
