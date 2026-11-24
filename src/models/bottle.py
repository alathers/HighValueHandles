"""Bottle and metadata models for bourbon inventory."""

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from models.evidence import WebResearch
from models.facts import FactSheet

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
        label_price: A price visible in the photos (price tag or printed
            price), as 'NNN.NN'. A shop or retail price, never an MSRP.
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
        vision_attempts: Vision requests made for this reading (1 = first try).
        ocr_by_image: Cleaned OCR text per photo, keyed by file name.
    """

    name: Optional[str] = None
    distillery: Optional[str] = None
    label_price: Optional[str] = None
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
    vision_attempts: int = 1
    ocr_by_image: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "name": self.name,
            "distillery": self.distillery,
            "label_price": self.label_price,
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
            "vision_attempts": self.vision_attempts,
            "ocr_by_image": dict(self.ocr_by_image),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedMetadata":
        """Create ExtractedMetadata from dictionary.

        Keys added later (abv, vision_model, vision_confidence, warnings,
        same_bottle, vision_attempts, ocr_by_image) are optional, so metadata
        saved by older versions still loads. Older files call the visible
        price ``msrp``; it was always a price read from the photos, so it is
        loaded as ``label_price``.

        Args:
            data: Dictionary representation of metadata.

        Returns:
            ExtractedMetadata instance.
        """
        return cls(
            name=data.get("name"),
            distillery=data.get("distillery"),
            label_price=data.get("label_price", data.get("msrp")),
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
            vision_attempts=int(data.get("vision_attempts") or 1),
            ocr_by_image={str(k): str(v) for k, v in (data.get("ocr_by_image") or {}).items()},
        )


@dataclass
class ReferenceLink:
    """A web page consulted for the bottle, as published in ``reference_links``.

    Attributes:
        url: The fetched page URL.
        label: Page title, else its domain.
        source_class: ``official``, ``retailer`` or ``reference`` (None if unknown).
        retrieved_at: ISO-8601 UTC fetch time (None if unknown).
        used_as_evidence: A supported or user fact cites this page.
    """

    url: str
    label: str
    source_class: Optional[str] = None
    retrieved_at: Optional[str] = None
    used_as_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "url": self.url,
            "label": self.label,
            "source_class": self.source_class,
            "retrieved_at": self.retrieved_at,
            "used_as_evidence": self.used_as_evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceLink":
        """Create ReferenceLink from dictionary (the newer keys are optional).

        Args:
            data: Dictionary representation.

        Returns:
            ReferenceLink instance.
        """
        return cls(
            url=data["url"],
            label=data["label"],
            source_class=data.get("source_class"),
            retrieved_at=data.get("retrieved_at"),
            used_as_evidence=bool(data.get("used_as_evidence", False)),
        )


def _cited_web_ids(facts: Optional[FactSheet]) -> set:
    """Web source ids cited by a supported or user field."""
    if facts is None:
        return set()
    out = set()
    for fact in facts.fields.values():
        if fact.status in ("supported", "user") and fact.value is not None:
            out.update(c.source_id for c in fact.citations if c.source_id.startswith("W"))
    return out


def build_reference_links(
    web_research: Optional[WebResearch], facts: Optional[FactSheet]
) -> List[ReferenceLink]:
    """Reference links from the web evidence: pages cited by a supported fact first, then the rest.

    Only fetched pages (the evidence excerpts) are listed; every other
    consulted URL is in ``web_research.pages`` (the research log).
    """
    if web_research is None:
        return []
    cited = _cited_web_ids(facts)
    ordered = [e for e in web_research.evidence if e.source_id in cited] + [
        e for e in web_research.evidence if e.source_id not in cited
    ]
    return [
        ReferenceLink(
            url=e.url,
            label=(e.title or "").strip() or e.domain,
            source_class=e.tier,
            retrieved_at=e.fetched_at,
            used_as_evidence=e.source_id in cited,
        )
        for e in ordered
    ]


def _person_msrp(bottle_dict: Dict[str, Any], corrections: Any) -> Optional[Decimal]:
    """A bottle without facts: its ``msrp`` only if a person's correction set it to that value.

    Anything else in ``bottle_dict['msrp']`` (a label or shop price) is not
    an MSRP and is ignored.
    """
    current = parse_money(bottle_dict.get("msrp"))
    if current is None:
        return None
    items = list(corrections or []) + list(bottle_dict.get("corrections") or [])
    for item in items:
        get = item.get if isinstance(item, dict) else (lambda key, _i=item: getattr(_i, key, None))
        source = str(get("source") or "interactive")
        if get("field_name") == "msrp" and (source == "interactive" or source.startswith(("feedback:", "hint:"))):
            if parse_money(get("corrected_value")) == current:
                return current
    return None


def _facts_from(value: Any) -> Optional[FactSheet]:
    if value is None or isinstance(value, FactSheet):
        return value
    return FactSheet.from_dict(value)


def _research_from(value: Any) -> Optional[WebResearch]:
    if value is None or isinstance(value, WebResearch):
        return value
    return WebResearch.from_dict(value)


@dataclass
class Quantity:
    """Quantity and valuation information.

    Scenarios:
    - Analyzed bottle: price=msrp, which comes only from a verified MSRP
      fact; None (unknown, never $0) until one exists
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
    - name: Bottle name as read from the label.
    - distillery: Distillery name.
    - msrp: Recommended MSRP; None when unknown. A price seen in a photo is a
      shop price (``label_price``), never an MSRP.
    - specific_release: Specific release information/style.

    Schema v2: an analyzed bottle carries its ``facts`` (every value with its
    provenance, conflicts, prices, verification, copy and publish gate), its
    ``web_research`` (every page consulted), an optional ``vendor_image``
    (reference only) and the ``run_id``. The flat fields above are derived
    from the facts (see ``image_processor.enrichment.apply_fact_values``).

    Optional fields:
    - proof: Proof statement.
    - age: Age statement.
    - tasting_notes: Tasting notes from review/vendor.

    Derived fields:
    - quantity: Quantity and valuation info (extracted or manual).
    - total_value: quantity * price; None when the price is unknown.
    - metadata: OCR-extracted text from images.

    Attributes:
        name: Bottle name (required).
        distillery: Distillery name (required).
        specific_release: Specific release info/style (required).
        msrp: Recommended MSRP (required argument; None = unknown).
        proof: Proof statement.
        age: Age statement (e.g., "9 years").
        tasting_notes: Tasting notes.
        metadata: OCR-extracted text.
        quantity: Quantity and valuation info (extracted or manual).
        reference_links: List of vendor links (optional, non-blocking).
        corrections: List of user corrections.
        extracted_at: Timestamp of data extraction (timezone-aware UTC).
        abv: Alcohol by volume as printed, e.g. '55.2%'.
        label_price: A price visible on the label or a price tag (retail,
            never an MSRP); None if none was seen.
        facts: The fact sheet, or None for a bottle that was not fact-checked.
        web_research: The web research record, or None.
        vendor_image: Accepted vendor reference image record (not for
            publication), or None.
        run_id: The ``process`` run that produced the bottle.
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
    reference_links: List[ReferenceLink] = field(default_factory=list)
    corrections: List["Correction"] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=utc_now)
    abv: Optional[str] = None
    label_price: Optional[Decimal] = None
    facts: Optional[FactSheet] = None
    web_research: Optional[WebResearch] = None
    vendor_image: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    @property
    def review_required(self) -> bool:
        """True when a person must look: an unresolved conflict or a stale description."""
        if self.facts is None:
            return False
        if any(not c.resolved for c in self.facts.conflicts):
            return True
        return bool(self.facts.copy is not None and self.facts.copy.stale)

    @property
    def publish_ready(self) -> bool:
        """The fact sheet's publish gate (False without facts)."""
        return bool(self.facts is not None and self.facts.publish.get("ready"))

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
        ``BottleAnalyzer.process_single_bottle`` plus the enrichment phase,
        or a batch JSON) into a Bottle. The MSRP comes only from the facts
        (``facts.msrp`` supported or user-stated); a ``bottle_dict['msrp']``
        without facts behind it is ignored. The quantity is priced at that
        MSRP, metadata is always attached and every source image is listed.

        Args:
            bottle_dict: Bottle fields: name, distillery, specific_release,
                proof, abv, age, tasting_notes, batch_number, label_price,
                images (all photo paths, primary first), quantity, and the
                schema-v2 keys facts, web_research, vendor_image and run_id.
            metadata: Extracted metadata; copied, never modified.
            quantity: Bottle count or a ready Quantity (used as-is). Default:
                ``bottle_dict['quantity']``, else 1.
            corrections: Corrections as Correction objects or dicts.

        Returns:
            Bottle with ``metadata.source_images`` = metadata's images followed
            by any other ``bottle_dict['images']`` (order kept, no duplicates).
        """
        facts = _facts_from(bottle_dict.get("facts"))
        web_research = _research_from(bottle_dict.get("web_research"))
        # None = unknown, never $0; only a verified (or user-stated) MSRP fact counts.
        if facts is not None:
            msrp = parse_money(facts.flat_values().get("msrp"))
        else:
            msrp = _person_msrp(bottle_dict, corrections)

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
        label_price = parse_money(
            bottle_dict.get("label_price") if bottle_dict.get("label_price") not in (None, "")
            else bottle_metadata.label_price
        )
        vendor_image = bottle_dict.get("vendor_image")

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
            reference_links=build_reference_links(web_research, facts),
            label_price=label_price if label_price is not None and label_price > 0 else None,
            facts=facts,
            web_research=web_research,
            vendor_image=dict(vendor_image) if isinstance(vendor_image, dict) else None,
            run_id=_clean_str(bottle_dict.get("run_id")),
        )

    @property
    def total_value(self) -> Optional[Decimal]:
        """Total value: ``quantity.count * quantity.price``.

        Returns:
            The total, or None when the price is unknown (never a guess).
        """
        price = self.quantity.price
        if price is None:
            return None
        return Decimal(self.quantity.count) * price

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
            "label_price": str(self.label_price) if self.label_price is not None else None,
            "quantity": self.quantity.to_dict(),
            "extracted_at": self.extracted_at.isoformat(),
            "run_id": self.run_id,
            "reference_links": [link.to_dict() for link in self.reference_links],
            "corrections": [correction.to_dict() for correction in self.corrections],
            "facts": self.facts.to_dict() if self.facts is not None else None,
            "web_research": self.web_research.to_dict() if self.web_research is not None else None,
            "vendor_image": dict(self.vendor_image) if self.vendor_image is not None else None,
            "review_required": self.review_required,
        }

        if self.metadata:
            data["metadata"] = self.metadata.to_dict()

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

        reference_links = [ReferenceLink.from_dict(item) for item in data.get("reference_links") or []]
        corrections = [Correction.from_dict(item) for item in data.get("corrections") or []]

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
            reference_links=reference_links,
            corrections=corrections,
            extracted_at=extracted_at,
            label_price=parse_money(data.get("label_price")),
            facts=_facts_from(data.get("facts")),
            web_research=_research_from(data.get("web_research")),
            vendor_image=dict(data["vendor_image"]) if isinstance(data.get("vendor_image"), dict) else None,
            run_id=data.get("run_id"),
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
        """The bottle document in Markdown (same as ``PreviewGenerator.generate_bottle_markdown``).

        Returns:
            Markdown formatted string.
        """
        from image_processor.preview_generator import render_bottle_markdown

        return render_bottle_markdown(self)


#: Allowed ``Correction.source`` forms (see :class:`Correction`).
CORRECTION_SOURCE_INTERACTIVE = "interactive"
CORRECTION_SOURCE_PREFIXES = ("feedback:", "hint:")


@dataclass
class Correction:
    """A human correction to one field of one bottle.

    A correction is always aimed at a single bottle by its ``bottle_id`` (the
    run-assigned ID, see ``PreviewGenerator.bottle_id_for``); it is never
    matched to a bottle by field value, so an edit can never land on another
    bottle. ``original_value`` is the value the person saw; a correction is
    applied only while the bottle still holds that value
    (:func:`image_processor.corrections.apply_corrections`).

    Attributes:
        field_name: Name of field to correct.
        original_value: The value the person saw when correcting.
        corrected_value: User-provided corrected value.
        bottle_id: The bottle the correction is for (required).
        corrected_at: Timestamp of correction.
        source: Where the correction came from: ``'interactive'``
            (interactive review), ``'feedback:<file>'`` (a corrections file
            or a retry report) or ``'hint:<file>#<id>: <text>'`` (a hint file).
    """

    field_name: str
    original_value: str
    corrected_value: str
    bottle_id: str
    corrected_at: datetime = field(default_factory=utc_now)
    source: str = CORRECTION_SOURCE_INTERACTIVE

    def __post_init__(self) -> None:
        if not isinstance(self.bottle_id, str) or not self.bottle_id.strip():
            raise ValueError("Correction.bottle_id must be a non-empty string")
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("Correction.field_name must be a non-empty string")
        if self.corrected_value is None:
            # str(None) would otherwise publish the literal text "None".
            raise ValueError(
                f"Correction of {self.field_name!r} for {self.bottle_id} has no corrected_value"
            )
        if not isinstance(self.source, str) or not (
            self.source == CORRECTION_SOURCE_INTERACTIVE
            or (
                self.source.startswith(CORRECTION_SOURCE_PREFIXES)
                and len(self.source.split(":", 1)[1].strip()) > 0
            )
        ):
            raise ValueError(
                "Correction.source must be 'interactive', 'feedback:<file>' or "
                f"'hint:<file>#<id>: <text>', got {self.source!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of correction.
        """
        return {
            "bottle_id": self.bottle_id,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "corrected_at": self.corrected_at.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correction":
        """Create Correction from dictionary.

        Args:
            data: Dictionary representation of correction.

        Returns:
            Correction instance.

        Raises:
            ValueError: If ``bottle_id`` is missing or empty, or ``source``
                is not an allowed form.
        """
        if not data.get("bottle_id"):
            raise ValueError(
                f"Correction for field {data.get('field_name')!r} has no bottle_id; "
                "every correction must name the bottle it is for"
            )
        corrected_at = utc_now()
        if data.get("corrected_at"):
            corrected_at = parse_timestamp(data["corrected_at"])

        return cls(
            field_name=data["field_name"],
            original_value=data["original_value"],
            corrected_value=data["corrected_value"],
            bottle_id=str(data["bottle_id"]),
            corrected_at=corrected_at,
            source=data.get("source") or CORRECTION_SOURCE_INTERACTIVE,
        )
