"""Fact sheet: every published bottle value with its provenance.

The fact-check stage (``image_processor.fact_checker``) builds one
:class:`FactSheet` per bottle. The text model only *proposes* claims; Python
verifies them against the exact source text, reconciles them with the label
reading, OCR text and user facts, assigns confidence by fixed rules and
decides whether the bottle is publish-ready.

Field statuses:

- ``supported``: one agreed value backed by at least one source.
- ``user``: a person stated the value (hint file, interactive edit, feedback).
  It wins over every other candidate; a disagreement is kept as a resolved
  :class:`Conflict`.
- ``conflict``: sources disagree; the value is None and every candidate is kept
  for a human to decide.
- ``unknown``: no source supports a value; ``reason`` says what was tried.

Serialization: every class has ``to_dict``/``from_dict`` producing plain JSON
types. Decimal amounts (prices and the ``msrp`` value) are stored as strings so
``49.99`` never becomes ``49.990000000000002``; proof and ABV are JSON numbers;
the release year is an integer; every other value is a string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from models.evidence import WebEvidence

#: Fields the fact sheet tracks, in display order.
FACT_FIELDS: tuple[str, ...] = (
    "name",
    "distillery",
    "bottler",
    "specific_release",
    "batch_number",
    "proof",
    "abv",
    "age",
    "release_year",
    "mash_bill",
    "tasting_notes",
    "msrp",
)
#: Fields whose value is a float (JSON number).
FLOAT_FIELDS: frozenset[str] = frozenset({"proof", "abv"})
#: Fields whose value is an integer.
INT_FIELDS: frozenset[str] = frozenset({"release_year"})
#: Fields whose value is a Decimal amount (stored as a string).
MONEY_FIELDS: frozenset[str] = frozenset({"msrp"})

SOURCE_KINDS: tuple[str, ...] = ("label", "ocr", "web", "user")
FIELD_STATUSES: tuple[str, ...] = ("supported", "user", "conflict", "unknown")
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
QUOTE_KINDS: tuple[str, ...] = ("verbatim", "reading")
PRICE_KINDS: tuple[str, ...] = ("msrp", "retail", "secondary")
VERIFICATION_STATUSES: tuple[str, ...] = ("verified", "partial", "failed", "skipped")

SourceKind = Literal["label", "ocr", "web", "user"]
FieldStatus = Literal["supported", "user", "conflict", "unknown"]
Confidence = Literal["high", "medium", "low"]
QuoteKind = Literal["verbatim", "reading"]
PriceKind = Literal["msrp", "retail", "secondary"]
VerificationStatus = Literal["verified", "partial", "failed", "skipped"]


def _check(name: str, value: Any, allowed: tuple[str, ...], *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _req(data: Mapping[str, Any], key: str, cls: str) -> Any:
    if key not in data:
        raise ValueError(f"{cls}: missing required key {key!r}")
    return data[key]


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _decimal(value: Any, what: str) -> Decimal:
    if isinstance(value, float):
        raise ValueError(f"{what} must be serialized as a string, not a float")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError(f"{what} is not a number: {value!r}") from e
    if not amount.is_finite():
        raise ValueError(f"{what} is not finite: {value!r}")
    return amount


def encode_value(field_name: str, value: Any) -> Any:
    """JSON form of a field value (Decimals become strings)."""
    if isinstance(value, Decimal):
        return str(value)
    return value


def decode_value(field_name: str, value: Any) -> Any:
    """Inverse of :func:`encode_value` for ``field_name``."""
    if value is None:
        return None
    if field_name in MONEY_FIELDS:
        return _decimal(value, f"{field_name} value")
    return value


@dataclass(frozen=True)
class Source:
    """A source a citation can point to.

    Attributes:
        id: ``L1`` (label reading), ``O1..`` (OCR per photo), ``W1..`` (web
            excerpt) or ``U1..`` (a person).
        kind: ``label``, ``ocr``, ``web`` or ``user``.
        ref: Label/OCR: image file name(s); web: the fetched URL; user:
            ``hint:<file>#<id>: <source>``, ``interactive`` or ``feedback:<file>``.
        url: Fetched URL (web only; never taken from model output).
        title: Page title (web only).
        tier: ``official``, ``retailer`` or ``reference`` (web only).
        fetched_at: ISO-8601 fetch time (web only).
        content_sha256: sha256 of the exact text the citations are checked against.
        evidence_path: Where the pipeline saves the excerpt (``evidence/W1.txt``).
    """

    id: str
    kind: SourceKind
    ref: str
    url: str | None = None
    title: str | None = None
    tier: str | None = None
    fetched_at: str | None = None
    content_sha256: str | None = None
    evidence_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the kind.

        Raises:
            ValueError: If ``kind`` is unknown or ``id`` is empty.
        """
        _check("Source.kind", self.kind, SOURCE_KINDS)
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Source.id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "id": self.id,
            "kind": self.kind,
            "ref": self.ref,
            "url": self.url,
            "title": self.title,
            "tier": self.tier,
            "fetched_at": self.fetched_at,
            "content_sha256": self.content_sha256,
            "evidence_path": self.evidence_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Source:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            id=str(_req(data, "id", name)),
            kind=_req(data, "kind", name),
            ref=str(_req(data, "ref", name)),
            url=_opt_str(data.get("url")),
            title=_opt_str(data.get("title")),
            tier=_opt_str(data.get("tier")),
            fetched_at=_opt_str(data.get("fetched_at")),
            content_sha256=_opt_str(data.get("content_sha256")),
            evidence_path=_opt_str(data.get("evidence_path")),
        )


@dataclass(frozen=True)
class Citation:
    """A pointer from a value to the text that supports it.

    Attributes:
        source_id: Id of a :class:`Source`.
        quote: Verbatim excerpt of that source (``verbatim``), the vision
            model's reading of the label (``reading``), or a user fact's ref.
        quote_kind: ``verbatim`` or ``reading``.
        derived: The value was computed from the quote (e.g. proof from ABV).
    """

    source_id: str
    quote: str
    quote_kind: QuoteKind = "verbatim"
    derived: bool = False

    def __post_init__(self) -> None:
        """Validate the quote kind.

        Raises:
            ValueError: If ``quote_kind`` is unknown.
        """
        _check("Citation.quote_kind", self.quote_kind, QUOTE_KINDS)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "source_id": self.source_id,
            "quote": self.quote,
            "quote_kind": self.quote_kind,
            "derived": self.derived,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Citation:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            source_id=str(_req(data, "source_id", name)),
            quote=str(_req(data, "quote", name)),
            quote_kind=data.get("quote_kind", "verbatim"),
            derived=bool(data.get("derived", False)),
        )


@dataclass
class Candidate:
    """One proposed value for a field with the citations that back it.

    Attributes:
        value: The proposed value (typed as described in the module docstring).
        citations: Supporting citations.
        origin: ``label``, ``ocr``, ``web`` or ``user``.
    """

    value: Any
    citations: list[Citation]
    origin: SourceKind

    def __post_init__(self) -> None:
        """Validate the origin.

        Raises:
            ValueError: If ``origin`` is unknown.
        """
        _check("Candidate.origin", self.origin, SOURCE_KINDS)

    def to_dict(self, field_name: str = "") -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "value": encode_value(field_name, self.value),
            "citations": [c.to_dict() for c in self.citations],
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], field_name: str = "") -> Candidate:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            value=decode_value(field_name, _req(data, "value", name)),
            citations=[Citation.from_dict(c) for c in data.get("citations") or []],
            origin=_req(data, "origin", name),
        )


@dataclass
class FieldFact:
    """The decided value of one field.

    Attributes:
        value: The value when ``status`` is ``supported`` or ``user``; else None.
        status: ``supported``, ``user``, ``conflict`` or ``unknown``.
        confidence: ``high``, ``medium`` or ``low`` (None unless there is a value).
        confidence_basis: Which sources produced the confidence
            (e.g. ``label(high)+ocr``).
        citations: Citations of the chosen value.
        candidates: Every candidate considered (kept for review).
        reason: Why the value is unknown or conflicting.
    """

    value: Any
    status: FieldStatus
    confidence: Confidence | None = None
    confidence_basis: str | None = None
    citations: list[Citation] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate status and confidence.

        Raises:
            ValueError: If either is unknown, or a conflict/unknown field has
                a value.
        """
        _check("FieldFact.status", self.status, FIELD_STATUSES)
        _check("FieldFact.confidence", self.confidence, CONFIDENCE_LEVELS, optional=True)
        if self.status in ("conflict", "unknown") and self.value is not None:
            raise ValueError(f"FieldFact with status {self.status!r} must have value None")

    @classmethod
    def unknown(cls, reason: str, candidates: list[Candidate] | None = None) -> FieldFact:
        """An unknown field with ``reason``."""
        return cls(value=None, status="unknown", reason=reason, candidates=list(candidates or []))

    def to_dict(self, field_name: str = "") -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "value": encode_value(field_name, self.value),
            "status": self.status,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
            "citations": [c.to_dict() for c in self.citations],
            "candidates": [c.to_dict(field_name) for c in self.candidates],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], field_name: str = "") -> FieldFact:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            value=decode_value(field_name, data.get("value")),
            status=_req(data, "status", name),
            confidence=data.get("confidence"),
            confidence_basis=_opt_str(data.get("confidence_basis")),
            citations=[Citation.from_dict(c) for c in data.get("citations") or []],
            candidates=[Candidate.from_dict(c, field_name) for c in data.get("candidates") or []],
            reason=_opt_str(data.get("reason")),
        )


@dataclass
class Conflict:
    """Sources that disagree about a field, for a human to settle.

    Attributes:
        id: Stable id within the sheet (``proof-1``).
        field: Field name (``proof``; ``proof/abv`` for the consistency check).
        kind: ``value_mismatch`` or ``proof_abv_inconsistent``.
        candidates: The disagreeing candidates.
        note: Human-readable summary.
        resolved_by: The user citation that settled it, if any.
    """

    id: str
    field: str
    kind: str
    candidates: list[Candidate]
    note: str
    resolved_by: Citation | None = None

    @property
    def resolved(self) -> bool:
        """Whether a person settled this conflict."""
        return self.resolved_by is not None

    def _value_field(self) -> str:
        return self.field if self.field in FACT_FIELDS else ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "id": self.id,
            "field": self.field,
            "kind": self.kind,
            "candidates": [c.to_dict(self._value_field()) for c in self.candidates],
            "note": self.note,
            "resolved_by": self.resolved_by.to_dict() if self.resolved_by else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Conflict:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        fname = str(_req(data, "field", name))
        vfield = fname if fname in FACT_FIELDS else ""
        resolved = data.get("resolved_by")
        return cls(
            id=str(_req(data, "id", name)),
            field=fname,
            kind=str(_req(data, "kind", name)),
            candidates=[Candidate.from_dict(c, vfield) for c in data.get("candidates") or []],
            note=str(data.get("note") or ""),
            resolved_by=Citation.from_dict(resolved) if resolved else None,
        )


@dataclass
class PriceObservation:
    """A price seen somewhere, with its kind.

    Attributes:
        kind: ``msrp`` (a stated suggested retail price), ``retail`` (a shop's
            price, a price tag on the bottle, JSON-LD offers) or ``secondary``
            (auction or resale).
        amount: Positive Decimal.
        currency: ISO currency code.
        citation: Where the price was seen.
        confidence: Confidence of an MSRP observation (None for other kinds).
    """

    kind: PriceKind
    amount: Decimal
    currency: str
    citation: Citation
    confidence: str | None = None

    def __post_init__(self) -> None:
        """Validate kind and amount.

        Raises:
            ValueError: If the kind is unknown or the amount is not a positive Decimal.
        """
        _check("PriceObservation.kind", self.kind, PRICE_KINDS)
        if isinstance(self.amount, bool) or not isinstance(self.amount, Decimal):
            raise ValueError("PriceObservation.amount must be a Decimal")
        if not self.amount.is_finite() or self.amount <= 0:
            raise ValueError(f"PriceObservation.amount must be positive, got {self.amount}")

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (amount as a string)."""
        return {
            "kind": self.kind,
            "amount": str(self.amount),
            "currency": self.currency,
            "citation": self.citation.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceObservation:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            kind=_req(data, "kind", name),
            amount=_decimal(_req(data, "amount", name), "PriceObservation.amount"),
            currency=str(_req(data, "currency", name)),
            citation=Citation.from_dict(_req(data, "citation", name)),
            confidence=_opt_str(data.get("confidence")),
        )


@dataclass
class Verification:
    """How the text-model pass went.

    Attributes:
        status: ``verified`` (EXTRACT ran and parsed), ``partial`` (COMPOSE
            failed), ``failed`` or ``skipped`` (no text model).
        reason: Why the status is not ``verified``.
        text_model: Model name used, if any.
        think: Think level used for EXTRACT.
        prompt_version: Prompt version (``facts-1``).
        duration_s: Wall time of the text pass.
        attempts: Model calls made.
        degraded_think: A length cutoff forced a lower think level.
        rejected: Every model proposal Python rejected, with the reason code.
    """

    status: VerificationStatus
    reason: str | None
    text_model: str | None
    think: str | None
    prompt_version: str
    duration_s: float = 0.0
    attempts: int = 0
    degraded_think: bool = False
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the status.

        Raises:
            ValueError: If ``status`` is unknown.
        """
        _check("Verification.status", self.status, VERIFICATION_STATUSES)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "status": self.status,
            "reason": self.reason,
            "text_model": self.text_model,
            "think": self.think,
            "prompt_version": self.prompt_version,
            "duration_s": self.duration_s,
            "attempts": self.attempts,
            "degraded_think": self.degraded_think,
            "rejected": [dict(r) for r in self.rejected],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Verification:
        """Inverse of :meth:`to_dict`."""
        name = cls.__name__
        return cls(
            status=_req(data, "status", name),
            reason=_opt_str(data.get("reason")),
            text_model=_opt_str(data.get("text_model")),
            think=_opt_str(data.get("think")),
            prompt_version=str(_req(data, "prompt_version", name)),
            duration_s=float(data.get("duration_s") or 0.0),
            attempts=int(data.get("attempts") or 0),
            degraded_think=bool(data.get("degraded_think", False)),
            rejected=[dict(r) for r in data.get("rejected") or []],
        )


@dataclass
class PublishCopy:
    """Verified description text.

    Attributes:
        sentences: ``{text, citations: [int], fields: [str]}`` per sentence.
        description_md: Sentences rendered with inline ``[n]`` markers.
        stale: A field the copy cites changed after it was written.
        reason: Why the copy is stale.
    """

    sentences: list[dict[str, Any]]
    description_md: str
    stale: bool = False
    reason: str | None = None

    def cited_fields(self) -> set[str]:
        """Field names any sentence uses."""
        return {str(f) for s in self.sentences for f in s.get("fields") or []}

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "sentences": [dict(s) for s in self.sentences],
            "description_md": self.description_md,
            "stale": self.stale,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PublishCopy:
        """Inverse of :meth:`to_dict`."""
        return cls(
            sentences=[dict(s) for s in data.get("sentences") or []],
            description_md=str(data.get("description_md") or ""),
            stale=bool(data.get("stale", False)),
            reason=_opt_str(data.get("reason")),
        )


@dataclass
class UserFact:
    """A value a person stated for a field.

    Attributes:
        field: Field name (one of :data:`FACT_FIELDS`).
        value: The stated value.
        ref: Where it came from: ``hint:<file>#<id>: <source>``,
            ``interactive`` or ``feedback:<file>``.
    """

    field: str
    value: Any
    ref: str

    def __post_init__(self) -> None:
        """Validate the field.

        Raises:
            ValueError: If the field is not a fact field, the value is None or
                the ref is empty.
        """
        if self.field not in FACT_FIELDS:
            raise ValueError(f"UserFact.field must be one of {FACT_FIELDS}, got {self.field!r}")
        if self.value is None:
            raise ValueError("UserFact.value must not be None")
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError("UserFact.ref must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {"field": self.field, "value": encode_value(self.field, self.value), "ref": self.ref}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UserFact:
        """Inverse of :meth:`to_dict`."""
        fname = str(_req(data, "field", cls.__name__))
        return cls(
            field=fname,
            value=decode_value(fname, _req(data, "value", cls.__name__)),
            ref=str(_req(data, "ref", cls.__name__)),
        )


@dataclass
class FactSheet:
    """Everything known about one bottle, with provenance.

    Attributes:
        fields: One :class:`FieldFact` per name in :data:`FACT_FIELDS`.
        sources: Every source a citation may point to.
        conflicts: Disagreements (resolved ones keep ``resolved_by``).
        prices: Every price observation (MSRP, retail, secondary).
        verification: The text-model pass record.
        copy: Verified description, or None.
        publish: ``{ready: bool, blockers: [str]}`` (plus ``group_warnings``,
            the split/mixed-group warnings the gate was given, so a later
            re-reconcile keeps them).
        review: ``{approved: bool, by: str|None, at: str|None}``.
    """

    fields: dict[str, FieldFact]
    sources: list[Source]
    conflicts: list[Conflict]
    prices: list[PriceObservation]
    verification: Verification
    copy: PublishCopy | None
    publish: dict[str, Any] = field(default_factory=lambda: {"ready": False, "blockers": []})
    review: dict[str, Any] = field(default_factory=lambda: {"approved": False, "by": None, "at": None})

    def flat_values(self) -> dict[str, Any]:
        """Each field's value when supported or user-stated, else None."""
        out: dict[str, Any] = {}
        for name in FACT_FIELDS:
            fact = self.fields.get(name)
            out[name] = fact.value if fact is not None and fact.status in ("supported", "user") else None
        return out

    def source(self, source_id: str) -> Source | None:
        """The source with ``source_id``, or None."""
        return next((s for s in self.sources if s.id == source_id), None)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "fields": {name: fact.to_dict(name) for name, fact in self.fields.items()},
            "sources": [s.to_dict() for s in self.sources],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "prices": [p.to_dict() for p in self.prices],
            "verification": self.verification.to_dict(),
            "copy": self.copy.to_dict() if self.copy is not None else None,
            "publish": {
                **self.publish,
                "blockers": list(self.publish.get("blockers") or []),
            },
            "review": dict(self.review),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactSheet:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: If a required key is missing or a value is invalid.
        """
        name = cls.__name__
        copy = data.get("copy")
        publish = dict(data.get("publish") or {"ready": False, "blockers": []})
        publish["ready"] = bool(publish.get("ready", False))
        publish["blockers"] = [str(b) for b in publish.get("blockers") or []]
        review = dict(data.get("review") or {})
        review = {
            "approved": bool(review.get("approved", False)),
            "by": _opt_str(review.get("by")),
            "at": _opt_str(review.get("at")),
        }
        return cls(
            fields={
                str(k): FieldFact.from_dict(v, str(k)) for k, v in (_req(data, "fields", name) or {}).items()
            },
            sources=[Source.from_dict(s) for s in data.get("sources") or []],
            conflicts=[Conflict.from_dict(c) for c in data.get("conflicts") or []],
            prices=[PriceObservation.from_dict(p) for p in data.get("prices") or []],
            verification=Verification.from_dict(_req(data, "verification", name)),
            copy=PublishCopy.from_dict(copy) if copy else None,
            publish=publish,
            review=review,
        )


@dataclass(frozen=True)
class FactInput:
    """Everything the fact check needs for one bottle.

    Attributes:
        bottle_id: The bottle's id.
        label: Vision reading: name, distillery, specific_release, proof, abv,
            age, batch_number, tasting_notes, label_price (missing or None =
            not read).
        vision_confidence: Vision's overall confidence (high|medium|low).
        images: Photo FILE NAMES (no directories), primary first.
        ocr_by_image: File name -> OCR text.
        user_facts: Values a person stated.
        web: Web excerpts in rank order (``WebEvidence``).
        group_warnings: Split or mixed-group warnings (each blocks publishing).
    """

    bottle_id: str
    label: Mapping[str, Any]
    vision_confidence: str | None
    images: Sequence[str]
    ocr_by_image: Mapping[str, str]
    user_facts: Sequence[UserFact] = ()
    web: Sequence[WebEvidence] = ()
    group_warnings: Sequence[str] = ()
