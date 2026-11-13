"""Processing state models for batch previews and tracking."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional


class BatchItemStatus(Enum):
    """Status of individual batch items."""

    PENDING = auto()
    """Not yet processed."""

    ACCUMULATING = auto()
    """ accumulating images for multi-angle shots."""

    READY = auto()
    """Ready for review and analysis."""

    ANALYZED = auto()
    """Completed LLM analysis."""

    CORRECTED = auto()
    """User corrections applied."""

    FINALIZED = auto()
    """Output files generated."""


@dataclass
class BatchItem:
    """Single item in batch processing queue.

    Represents one bottle (possibly with multiple image angles).

    Attributes:
        id: Unique identifier for this item.
        images: List of image paths for this bottle.
        status: Processing status.
        bottle_ref: Reference to Bottle object (if analyzed).
        created_at: Timestamp when item was added.
        updated_at: Timestamp of last update.
    """

    id: str
    images: List[Path]
    status: BatchItemStatus = BatchItemStatus.PENDING
    bottle_ref: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of batch item.
        """
        return {
            "id": self.id,
            "images": [str(p) for p in self.images],
            "status": self.status.name,
            "bottle_ref": self.bottle_ref,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchItem":
        """Create BatchItem from dictionary.

        Args:
            data: Dictionary representation of batch item.

        Returns:
            BatchItem instance.
        """
        return cls(
            id=data["id"],
            images=[Path(p) for p in data["images"]],
            status=BatchItemStatus[data["status"]],
            bottle_ref=data.get("bottle_ref"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else None,
        )


@dataclass
class BatchPreviewReport:
    """Preview report for user review before final extraction.

    Metadata fields:
    - name: Bottleaname from user/naming tool.
    - distillery: Distillery name.
    - msrp: Recommended MSRP price.
    - specific_release: Specific release information/style.

    Optional metadata fields:
    - proof: Proof statement.
    - year: Age statement.
    - tasting_notes: Tasting notes from review/vendor.

    OCR text:
    - extracted_text: Concatenated OCR text from all images.

    Visual:
    - image_paths: List of image paths included in this batch.

    Processing:
    - status: Pending/NamingTool/Ready/Analyzed/Finalized
    - corrections: List of user corrections applied.

    Attributes:
        id: Unique identifier for this preview.
        metadata: Proposed bottle metadata (may be partial if pre-naming tool).
        extracted_text: OCR text from images.
        image_paths: List of image paths.
        status: Processing status.
        created_at: Timestamp when preview was created.
        updated_at: Timestamp of last update.
        corrections: List of user corrections.
    """

    id: str
    metadata: Dict[str, Any]
    extracted_text: str
    image_paths: List[Path]
    status: BatchItemStatus = BatchItemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    corrections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of preview report.
        """
        return {
            "id": self.id,
            "metadata": self.metadata,
            "extracted_text": self.extracted_text,
            "image_paths": [str(p) for p in self.image_paths],
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "corrections": self.corrections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchPreviewReport":
        """Create BatchPreviewReport from dictionary.

        Args:
            data: Dictionary representation of preview report.

        Returns:
            BatchPreviewReport instance.
        """
        return cls(
            id=data["id"],
            metadata=data["metadata"],
            extracted_text=data["extracted_text"],
            image_paths=[Path(p) for p in data["image_paths"]],
            status=BatchItemStatus[data["status"]],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else None,
            corrections=data.get("corrections", []),
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
    def from_json(cls, json_str: str) -> "BatchPreviewReport":
        """Create BatchPreviewReport from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            BatchPreviewReport instance.

        Raises:
            json.JSONDecodeError: If JSON is invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """Convert to Markdown preview for user review.

        Returns:
            Markdown formatted string with metadata and image list.
        """
        lines = [f"# Preview Report: {self.id}", ""]
        lines.append("## Metadata")

        # Core required fields
        if "name" in self.metadata:
            lines.append(f"**Name:** {self.metadata['name']}")

        if "distillery" in self.metadata:
            lines.append(f"**Distillery:** {self.metadata['distillery']}")

        if "msrp" in self.metadata:
            lines.append(f"**MSRP:** ${self.metadata['msrp']}")

        if "specific_release" in self.metadata:
            lines.append(f"**Release:** {self.metadata['specific_release']}")

        # Optional fields
        if "proof" in self.metadata and self.metadata["proof"]:
            lines.append(f"**Proof:** {self.metadata['proof']}")

        if "age" in self.metadata and self.metadata["age"]:
            lines.append(f"**Age:** {self.metadata['age']}")

        if "tasting_notes" in self.metadata and self.metadata["tasting_notes"]:
            lines.append("")
            lines.append("**Tasting Notes:**")
            lines.append(self.metadata["tasting_notes"])

        # OCR text
        lines.append("")
        lines.append("## OCR Text")
        lines.append("```")
        lines.append(self.extracted_text)
        lines.append("```")
        lines.append("")

        # Image list
        lines.append("## Images")
        for i, path in enumerate(self.image_paths, 1):
            lines.append(f"{i}. {path}")

        # Corrections
        if self.corrections:
            lines.append("")
            lines.append("## Corrections")
            for correction in self.corrections:
                lines.append(f"- **{correction['field_name']}:** `{correction['original_value']}` → `{correction['corrected_value']}`")

        lines.append("")
        return "\n".join(lines)
