"""Bottle and metadata models for bourbon inventory."""

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .image import Image

_MONEY_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_CENTS = Decimal("0.01")


def parse_money(value: Any) -> Optional[Decimal]:
    """Parse a price as produced by OCR or a vision model.

    Accepts numbers and strings such as ``'59.99'``, ``'$1,299.00'`` or
    ``'MSRP $45'``; the first number in a string is used.

    Args:
        value: int, float, Decimal or str (anything else is str()'d).

    Returns:
        The amount quantized to cents, or None for None, booleans, text
        without a number and negative amounts.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float, Decimal)):
            amount = Decimal(str(value))
        else:
            match = _MONEY_PATTERN.search(str(value).replace(",", ""))
            if not match:
                return None
            amount = Decimal(match.group(0))
        if not amount.is_finite() or amount < 0:
            return None
        return amount.quantize(_CENTS)
    except (InvalidOperation, ValueError):
        return None


def format_money(amount: Optional[Decimal]) -> str:
    """'$12.34', or 'Unknown' for a price that was not visible."""
    return "Unknown" if amount is None else f"${amount:.2f}"


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime (isoformat has '+00:00')."""
    return datetime.now(UTC)


def parse_timestamp(value: Any) -> datetime:
    """Parse an ISO timestamp written by any version of this tool.

    Older files hold naive UTC values; they are read as UTC so every loaded
    timestamp is timezone-aware.
    """
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _clean_str(value: Any) -> Optional[str]:
    """Return ``str(value).strip()``, or None for None or blank text."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class ExtractedMetadata:
    """OCR-extracted text metadata from bottle images.

    Attributes:
        name: Bottle name extracted from images.
        distillery: Distillery name extracted from images.
        msrp: MSRP price extracted.
        specific_release: Specific release information.
        proof: Proof statement.
        abv: Alcohol by volume as printed, e.g. '55.2%'.
        age: Age statement.
        batch_number: Batch/statement number.
        tasting_notes: Tasting notes extracted.
        full_text: Concatenated OCR text from all bottle images.
        selected_text: Text selected for LLM analysis.
        source_images: List of image paths used for extraction.
        vision_model: Vision model that produced the field values.
        vision_confidence: Model-reported confidence ('high'/'medium'/'low').
        warnings: Non-fatal problems found during extraction (e.g. the photos
            appear to show different products).
        same_bottle: The vision model's verdict on whether all photos sent
            together show one product (False = mixed); None when unknown or
            when only one photo was sent.
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
    # Declared after the original fields so positional construction keeps working.
    abv: Optional[str] = None
    selected_text: str = ""
    source_images: List[Path] = field(default_factory=list)
    vision_model: Optional[str] = None
    vision_confidence: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    same_bottle: Optional[bool] = None

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
            "abv": self.abv,
            "age": self.age,
            "batch_number": self.batch_number,
            "tasting_notes": self.tasting_notes,
            "full_text": self.full_text,
            "selected_text": self.selected_text,
            "source_images": [str(p) for p in self.source_images],
            "vision_model": self.vision_model,
            "vision_confidence": self.vision_confidence,
            "warnings": list(self.warnings),
            "same_bottle": self.same_bottle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedMetadata":
        """Create ExtractedMetadata from dictionary.

        Keys added later (abv, vision_model, vision_confidence, warnings,
        same_bottle) are optional, so metadata saved by older versions still loads.

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
            abv=data.get("abv"),
            age=data.get("age"),
            batch_number=data.get("batch_number"),
            tasting_notes=data.get("tasting_notes"),
            full_text=data.get("full_text", ""),
            selected_text=data.get("selected_text", ""),
            source_images=[Path(p) for p in data.get("source_images", [])],
            vision_model=data.get("vision_model"),
            vision_confidence=data.get("vision_confidence"),
            warnings=list(data.get("warnings") or []),
            same_bottle=data.get("same_bottle"),
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
    - Price not visible on the label: price=None (unknown, not $0)
    - User purchased case: quantity=12, price=0, source="manual"

    Attributes:
        count: Number of bottles.
        price: Price per bottle (Decimal for precision), or None if unknown.
        source: Source ("extracted" or "manual").
    """

    count: int = 1
    price: Optional[Decimal] = Decimal("0.00")
    source: str = "extracted"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "count": self.count,
            "price": str(self.price) if self.price is not None else None,
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
        price = data.get("price", "0.00")
        return cls(
            count=data.get("count", 1),
            price=Decimal(str(price)) if price not in (None, "") else None,
            source=data.get("source", "extracted"),
        )


@dataclass
class Bottle:
    """Bottle model for storing bourbon inventory data.

    Core fields (required):
    - name: Bottle name from user/naming tool.
    - distillery: Distillery name.
    - msrp: Recommended MSRP price; None when no price is visible (unknown).
    - specific_release: Specific release information/style.

    Optional fields:
    - proof: Proof statement.
    - age: Age statement.
    - tasting_notes: Tasting notes from review/vendor.

    Derived fields:
    - quantity: Quantity and valuation info (extracted or manual).
    - ohr: Estimated cost history value (decaying over time).
    - total_value: quantity * price (if price > 0, else ohr); None if unknown.
    - metadata: OCR-extracted text from images.

    Attributes:
        name: Bottle name (required).
        distillery: Distillery name (required).
        specific_release: Specific release info/style (required).
        msrp: Recommended MSRP price (required argument; None = unknown).
        proof: Proof statement.
        age: Age statement (e.g., "9 years").
        tasting_notes: Tasting notes.
        metadata: OCR-extracted text.
        quantity: Quantity and valuation info (extracted or manual).
        ohr: Estimated cost history value (decays over time).
        reference_links: List of vendor links (optional, non-blocking).
        corrections: List of user corrections.
        extracted_at: Timestamp of data extraction (timezone-aware UTC).
        abv: Alcohol by volume as printed, e.g. '55.2%'.
    """

    name: str
    distillery: str
    specific_release: str
    msrp: Optional[Decimal]
    proof: Optional[str] = None
    age: Optional[str] = None
    tasting_notes: Optional[str] = None
    metadata: Optional[ExtractedMetadata] = None
    quantity: Quantity = field(default_factory=Quantity)
    ohr: Optional[Decimal] = None
    reference_links: List[ReferenceLink] = field(default_factory=list)
    corrections: List["Correction"] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=utc_now)
    abv: Optional[str] = None

    @classmethod
    def from_analysis(
        cls,
        bottle_dict: Dict[str, Any],
        metadata: Optional[ExtractedMetadata] = None,
        quantity: Union[int, "Quantity", None] = None,
        corrections: Optional[List[Union["Correction", Dict[str, Any]]]] = None,
    ) -> "Bottle":
        """Build a Bottle from analyzer output.

        This is the one supported way to turn a ``bottle_dict`` (from
        ``BottleAnalyzer.process_single_bottle`` or a batch JSON) into a
        Bottle. It parses the MSRP, prices the quantity at the MSRP, always
        attaches metadata and lists every source image of the bottle.

        Args:
            bottle_dict: Bottle fields: name, distillery, specific_release,
                msrp (any '$1,299.00' style, number or None; unparseable or
                missing means unknown), proof, abv, age, tasting_notes,
                batch_number, images (all photo paths, primary first) and
                quantity.
            metadata: Extracted metadata; copied, never modified.
            quantity: Bottle count or a ready Quantity (used as-is). Default:
                ``bottle_dict['quantity']``, else 1.
            corrections: Corrections as Correction objects or dicts.

        Returns:
            Bottle with ``metadata.source_images`` = metadata's images followed
            by any other ``bottle_dict['images']`` (order kept, no duplicates).
        """
        msrp = parse_money(bottle_dict.get("msrp"))  # None = unknown, never $0

        if isinstance(quantity, Quantity):
            bottle_quantity = quantity
        else:
            raw_count = quantity if quantity is not None else bottle_dict.get("quantity", 1)
            try:
                count = max(1, int(raw_count))
            except (TypeError, ValueError):
                count = 1
            bottle_quantity = Quantity(count=count, price=msrp, source="extracted")

        bottle_metadata = dataclasses.replace(metadata) if metadata is not None else ExtractedMetadata()
        seen: set = set()
        source_images: List[Path] = []
        for image_path in [*bottle_metadata.source_images, *(bottle_dict.get("images") or [])]:
            if str(image_path) not in seen:
                seen.add(str(image_path))
                source_images.append(Path(image_path))
        bottle_metadata.source_images = source_images
        bottle_metadata.warnings = list(bottle_metadata.warnings)
        if not bottle_metadata.batch_number:
            bottle_metadata.batch_number = _clean_str(bottle_dict.get("batch_number"))

        return cls(
            name=_clean_str(bottle_dict.get("name")) or "Unknown Bottle",
            distillery=_clean_str(bottle_dict.get("distillery")) or "Unknown Distillery",
            specific_release=_clean_str(bottle_dict.get("specific_release")) or "",
            msrp=msrp,
            proof=_clean_str(bottle_dict.get("proof")),
            abv=_clean_str(bottle_dict.get("abv")) or bottle_metadata.abv,
            age=_clean_str(bottle_dict.get("age")),
            tasting_notes=_clean_str(bottle_dict.get("tasting_notes")),
            metadata=bottle_metadata,
            quantity=bottle_quantity,
            corrections=[
                Correction.from_dict(c) if isinstance(c, dict) else c for c in (corrections or [])
            ],
        )

    @property
    def total_value(self) -> Optional[Decimal]:
        """Calculate total bottle value based on quantity/price.

        Returns:
            Total value = quantity * price (or ohr if price = 0), or None
            when the price is unknown and there is no ohr estimate.

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
        price = self.quantity.price
        if price is not None and price > 0:
            return Decimal(self.quantity.count) * price
        if self.ohr:
            return Decimal(self.quantity.count) * self.ohr
        if price is None:
            return None
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
            "msrp": str(self.msrp) if self.msrp is not None else None,
            "proof": self.proof,
            "abv": self.abv,
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

        extracted_at = utc_now()
        if data.get("extracted_at"):
            extracted_at = parse_timestamp(data["extracted_at"])

        return cls(
            name=data["name"],
            distillery=data["distillery"],
            specific_release=data["specific_release"],
            msrp=Decimal(str(data["msrp"])) if data.get("msrp") not in (None, "") else None,
            proof=data.get("proof"),
            abv=data.get("abv"),
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
        lines.append(f"**MSRP:** {format_money(self.msrp)}")

        if self.proof:
            lines.append(f"**Proof:** {self.proof}")

        if self.abv:
            lines.append(f"**ABV:** {self.abv}")

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
            if self.quantity.price is not None and self.quantity.price > 0:
                lines.append(f"**Price:** ${self.quantity.price}/bottle")
            lines.append(f"**Total Value:** {format_money(self.total_value)}")
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
    corrected_at: datetime = field(default_factory=utc_now)

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
        corrected_at = utc_now()
        if data.get("corrected_at"):
            corrected_at = parse_timestamp(data["corrected_at"])

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
    generated_timestamp: datetime = field(default_factory=utc_now)
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
        generated_timestamp = utc_now()
        if data.get("generated_timestamp"):
            generated_timestamp = parse_timestamp(data["generated_timestamp"])

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

