"""Bottle and metadata models for bourbon inventory."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from .image import Image


@dataclass
class ExtractedMetadata:
    """OCR-extracted text metadata from bottle images.

    Attributes:
        name: Bottle name extracted from images.
        distillery: Distillery name extracted from images.
        msrp: MSRP price extracted.
        specific_release: Specific release information.
        proof: Proof statement.
        age: Age statement.
        batch_number: Batch/statement number.
        tasting_notes: Tasting notes extracted.
        full_text: Concatenated OCR text from all bottle images.
        selected_text: Text selected for LLM analysis.
        source_images: List of image paths used for extraction.
    """

    name: Optional[str] = None
    distillery: Optional[str] = None
    msrp: Optional[str] = None
    specific_release: Optional[str] = None
    proof: Optional[str] = None
    age: Optional[str] = None
    batch_number: Optional[str] = None
    tasting_notes: Optional[str] = None
    full_text: str = ""
    selected_text: str = ""
    source_images: List[Path] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "name": self.name,
            "distillery": self.distillery,
            "msrp": self.msrp,
            "specific_release": self.specific_release,
            "proof": self.proof,
            "age": self.age,
            "batch_number": self.batch_number,
            "tasting_notes": self.tasting_notes,
            "full_text": self.full_text,
            "selected_text": self.selected_text,
            "source_images": [str(p) for p in self.source_images],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedMetadata":
        """Create ExtractedMetadata from dictionary.

        Args:
            data: Dictionary representation of metadata.

        Returns:
            ExtractedMetadata instance.
        """
        return cls(
            name=data.get("name"),
            distillery=data.get("distillery"),
            msrp=data.get("msrp"),
            specific_release=data.get("specific_release"),
            proof=data.get("proof"),
            age=data.get("age"),
            batch_number=data.get("batch_number"),
            tasting_notes=data.get("tasting_notes"),
            full_text=data.get("full_text", ""),
            selected_text=data.get("selected_text", ""),
            source_images=[Path(p) for p in data.get("source_images", [])],
        )


@dataclass
class ReferenceLink:
    """Reference link for vendor documentation.

    Attributes:
        url: URL to vendor or review page.
        label: Descriptive label for the link.
    """

    url: str
    label: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {"url": self.url, "label": self.label}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ReferenceLink":
        """Create ReferenceLink from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ReferenceLink instance.
        """
        return cls(url=data["url"], label=data["label"])


@dataclass
class Quantity:
    """Quantity and valuation information.

    Scenarios:
    - Single bottle crawled image: quantity=1, price=msrp, source="extracted"
    - User purchased case: quantity=12, price=0, source="manual"

    Attributes:
        count: Number of bottles.
        price: Price per bottle (Decimal for precision).
        source: Source ("extracted" or "manual").
    """

    count: int = 1
    price: Decimal = Decimal("0.00")
    source: str = "extracted"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "count": self.count,
            "price": str(self.price),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Quantity":
        """Create Quantity from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Quantity instance.
        """
        return cls(
            count=data.get("count", 1),
            price=Decimal(data.get("price", "0.00")),
            source=data.get("source", "extracted"),
        )


@dataclass
class Bottle:
    """Bottle model for storing bourbon inventory data.

    Core fields (required):
    - name: Bottle name from user/naming tool.
    - distillery: Distillery name.
    - msrp: Recommended MSRP price.
    - specific_release: Specific release information/style.

    Optional fields:
    - proof: Proof statement.
    -年份: Age statement.
    - tasting_notes: Tasting notes from review/vendor.

    Derived fields:
    - quantity: Quantity and valuation info (extracted or manual).
    - ohr: Estimated cost history value (decaying over time).
    - total_value: quantity * price (if price != 0, else ohr).
    - metadata: OCR-extracted text from images.

    Attributes:
        name: Bottle name (required).
        distillery: Distillery name (required).
        specific_release: Specific release info/style (required).
        msrp: Recommended MSRP price (required).
        proof: Proof statement.
        age: Age statement (e.g., "9 years").
        tasting_notes: Tasting notes.
        metadata: OCR-extracted text.
        quantity: Quantity and valuation info (extracted or manual).
        ohr: Estimated cost history value (decays over time).
        reference_links: List of vendor links (optional, non-blocking).
        corrections: List of user corrections.
        extracted_at: Timestamp of data extraction.
    """

    name: str
    distillery: str
    specific_release: str
    msrp: Decimal
    proof: Optional[str] = None
    age: Optional[str] = None
    tasting_notes: Optional[str] = None
    metadata: Optional[ExtractedMetadata] = None
    quantity: Quantity = field(default_factory=Quantity)
    ohr: Optional[Decimal] = None
    reference_links: List[ReferenceLink] = field(default_factory=list)
    corrections: List["Correction"] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_value(self) -> Decimal:
        """Calculate total bottle value based on quantity/price.

        Returns:
            Total value = quantity * price (or ohr if price = 0).

        Scenario 1: Single bottle from crawled images
        - quantity.count = 1
        - quantity.price = msrp from extracted data
        - Total = 1 * msrp

        Scenario 2: User purchased a case (manual quantity)
        - quantity.count = 12
        - quantity.price = 0 (no extracted price)
        - ohr = decaying estimated value
        - Total = 12 * ohr
        """
        if self.quantity.price > 0:
            return Decimal(self.quantity.count) * self.quantity.price
        elif self.ohr:
            return Decimal(self.quantity.count) * self.ohr
        else:
            return Decimal("0.00")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of bottle.
        """
        data = {
            "name": self.name,
            "distillery": self.distillery,
            "specific_release": self.specific_release,
            "msrp": str(self.msrp),
            "proof": self.proof,
            "age": self.age,
            "tasting_notes": self.tasting_notes,
            "quantity": self.quantity.to_dict(),
            "ohr": str(self.ohr) if self.ohr else None,
            "extracted_at": self.extracted_at.isoformat(),
        }

        if self.metadata:
            data["metadata"] = self.metadata.to_dict()

        if self.reference_links:
            data["reference_links"] = [link.to_dict() for link in self.reference_links]

        if self.corrections:
            data["corrections"] = [correction.to_dict() for correction in self.corrections]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bottle":
        """Create Bottle from dictionary.

        Args:
            data: Dictionary representation of bottle.

        Returns:
            Bottle instance.
        """
        metadata = None
        if "metadata" in data:
            metadata = ExtractedMetadata.from_dict(data["metadata"])

        quantity = Quantity.from_dict(data.get("quantity", {}))

        reference_links = []
        if "reference_links" in data:
            reference_links = [ReferenceLink.from_dict(item) for item in data["reference_links"]]

        corrections = []
        if "corrections" in data:
            corrections = [Correction.from_dict(item) for item in data["corrections"]]

        extracted_at = datetime.utcnow()
        if "extracted_at" in data:
            extracted_at = datetime.fromisoformat(data["extracted_at"])

        return cls(
            name=data["name"],
            distillery=data["distillery"],
            specific_release=data["specific_release"],
            msrp=Decimal(data["msrp"]),
            proof=data.get("proof"),
            age=data.get("age"),
            tasting_notes=data.get("tasting_notes"),
            metadata=metadata,
            quantity=quantity,
            ohr=Decimal(data["ohr"]) if data.get("ohr") else None,
            reference_links=reference_links,
            corrections=corrections,
            extracted_at=extracted_at,
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
    def from_json(cls, json_str: str) -> "Bottle":
        """Create Bottle from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            Bottle instance.

        Raises:
            json.JSONDecodeError: If JSON is invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """Convert to Markdown for website publication.

        Returns:
            Markdown formatted string.
        """
        lines = [f"# {self.name}", ""]
        lines.append(f"**Distillery:** {self.distillery}")
        lines.append("")
        lines.append(f"**MSRP:** ${self.msrp}")

        if self.proof:
            lines.append(f"**Proof:** {self.proof}")

        if self.age:
            lines.append(f"**Age:** {self.age}")

        lines.append("")
        lines.append("## Release Information")
        lines.append(self.specific_release)
        lines.append("")

        if self.tasting_notes:
            lines.append("## Tasting Notes")
            lines.append(self.tasting_notes)
            lines.append("")

        if self.quantity.count > 1 or self.quantity.source == "manual":
            lines.append("## Quantity")
            lines.append(f"**Count:** {self.quantity.count}")
            if self.quantity.price > 0:
                lines.append(f"**Price:** ${self.quantity.price}/bottle")
            lines.append(f"**Total Value:** ${self.total_value}")
            lines.append("")
            lines.append(f"*Source: {self.quantity.source}*")
            lines.append("")

        if self.reference_links:
            lines.append("## References")
            for link in self.reference_links:
                lines.append(f"- [{link.label}]({link.url})")
            lines.append("")

        return "\n".join(lines)


@dataclass
class Correction:
    """User correction for extracted bottle data.

    Used in retry-from-report workflow to incorporate manual corrections.

    Attributes:
        field_name: Name of field to correct.
        original_value: Original extracted value.
        corrected_value: User-provided corrected value.
        corrected_at: Timestamp of correction.
    """

    field_name: str
    original_value: str
    corrected_value: str
    corrected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of correction.
        """
        return {
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "corrected_at": self.corrected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correction":
        """Create Correction from dictionary.

        Args:
            data: Dictionary representation of correction.

        Returns:
            Correction instance.
        """
        corrected_at = datetime.utcnow()
        if "corrected_at" in data:
            corrected_at = datetime.fromisoformat(data["corrected_at"])

        return cls(
            field_name=data["field_name"],
            original_value=data["original_value"],
            corrected_value=data["corrected_value"],
            corrected_at=corrected_at,
        )


@dataclass
class BatchPreviewReport:
    """Preview report for a processed batch of bottles.

    Contains markdown summary, grouping information, and instructions
    for user feedback in interactive processing mode.

    Attributes:
        report_id: Unique identifier for this report.
        batch_id: Identifier for the batch this report summarizes.
        generated_timestamp: When the report was generated.
        markdown_content: Full markdown content of the report.
        report_path: Filesystem path where the report is saved.
    """

    report_id: str
    batch_id: str
    generated_timestamp: datetime = field(default_factory=datetime.utcnow)
    markdown_content: str = ""
    report_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of batch preview report.
        """
        return {
            "report_id": self.report_id,
            "batch_id": self.batch_id,
            "generated_timestamp": self.generated_timestamp.isoformat(),
            "markdown_content": self.markdown_content,
            "report_path": str(self.report_path) if self.report_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchPreviewReport":
        """Create BatchPreviewReport from dictionary.

        Args:
            data: Dictionary representation of batch preview report.

        Returns:
            BatchPreviewReport instance.
        """
        generated_timestamp = datetime.utcnow()
        if "generated_timestamp" in data:
            generated_timestamp = datetime.fromisoformat(data["generated_timestamp"])

        report_path = None
        if "report_path" in data and data["report_path"]:
            report_path = Path(data["report_path"])

        return cls(
            report_id=data["report_id"],
            batch_id=data["batch_id"],
            generated_timestamp=generated_timestamp,
            markdown_content=data.get("markdown_content", ""),
            report_path=report_path,
        )

