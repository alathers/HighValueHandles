"""Web-evidence contract shared by the web-lookup layer and the fact check.

The web layer (producer) searches, fetches, cleans and excerpts pages; it
never asks a model anything. It returns one :class:`WebResearch` per bottle.
The fact-check stage and the output writers (consumers) read it:

- :class:`WebEvidence` is one page excerpt, *exactly* as the text model sees
  it, with a sha256 of that text so every quote can be traced back to the
  bytes that were shown. Its ``source_id`` (``"W1"``, ``"W2"``, ...) is what
  citations refer to.
- :class:`StructuredPrice` is a price read deterministically from a page's
  JSON-LD ``offers.price``. It is only ever a *retail* observation, never an
  MSRP.
- :class:`PageRecord` logs every URL considered for a bottle (cited, fetched
  but not cited, or skipped, with the reason), so each bottle document can
  show the full research trail.
- :class:`VendorImage` is a reference photo from a vendor page; it is not
  licensed for publication.

All classes serialize with ``to_dict``/``from_dict`` to plain JSON types:
times are ISO-8601 strings and Decimals are strings (never floats, which
would change ``49.99`` into ``49.990000000000002``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Optional

#: Source classes of a web page: the brand's own site, a shop, anything else.
SOURCE_CLASSES: tuple[str, ...] = ("official", "retailer", "reference")
#: What happened to a page considered for a bottle.
PAGE_OUTCOMES: tuple[str, ...] = ("cited", "fetched_not_cited", "skipped")
#: Overall result of a bottle's web research.
RESEARCH_STATUSES: tuple[str, ...] = ("ok", "no_results", "skipped", "failed")
#: Search providers that can appear in :attr:`WebResearch.provider`
#: (``auto`` and ``none`` are configuration values, never recorded here).
SEARCH_PROVIDERS: tuple[str, ...] = ("duckduckgo", "brave", "ollama", "searxng")
#: Longest excerpt shown to the text model per page.
MAX_EVIDENCE_CHARS = 10_000
#: Usage note stored with every vendor image.
VENDOR_IMAGE_USAGE = "reference only - not licensed for publication"

SourceClass = Literal["official", "retailer", "reference"]
PageOutcome = Literal["cited", "fetched_not_cited", "skipped"]
ResearchStatus = Literal["ok", "no_results", "skipped", "failed"]


def sha256_text(text: str) -> str:
    """Hex sha256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_choice(name: str, value: Any, allowed: tuple[str, ...], *, optional: bool = False) -> None:  # noqa: ANN401
    if optional and value is None:
        return
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _require(data: Mapping[str, Any], key: str, cls: str) -> Any:  # noqa: ANN401
    if key not in data:
        raise ValueError(f"{cls}: missing required key {key!r}")
    return data[key]


def _opt_str(value: Any) -> Optional[str]:  # noqa: ANN401
    return None if value is None else str(value)


@dataclass(frozen=True)
class StructuredPrice:
    """A price read from a page's JSON-LD ``offers.price`` (no model involved).

    Attributes:
        amount: Price, greater than zero.
        currency: ISO currency code as the page states it (e.g. ``'USD'``).
        kind: Always ``'retail'``: a shop's offer price is never an MSRP.
        origin: Where on the page the price came from.
    """

    amount: Decimal
    currency: str
    kind: Literal["retail"] = "retail"
    origin: str = "jsonld:offers.price"

    def __post_init__(self) -> None:
        """Validate the price.

        Raises:
            ValueError: If the amount is not a positive finite Decimal, the
                currency is empty or the kind is not ``'retail'``.
        """
        if isinstance(self.amount, bool) or not isinstance(self.amount, Decimal):
            raise ValueError(f"StructuredPrice.amount must be a Decimal, got {type(self.amount).__name__}")
        if not self.amount.is_finite() or self.amount <= 0:
            raise ValueError(f"StructuredPrice.amount must be positive, got {self.amount}")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("StructuredPrice.currency must be a non-empty string")
        if self.kind != "retail":
            raise ValueError(f"StructuredPrice.kind must be 'retail', got {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (the amount as a string)."""
        return {
            "amount": str(self.amount),
            "currency": self.currency,
            "kind": self.kind,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StructuredPrice:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: If a key is missing or the amount is not a number.
        """
        raw = _require(data, "amount", cls.__name__)
        if isinstance(raw, float):
            raise ValueError("StructuredPrice.amount must be serialized as a string, not a float")
        try:
            amount = Decimal(str(raw))
        except InvalidOperation as e:
            raise ValueError(f"StructuredPrice.amount is not a number: {raw!r}") from e
        return cls(
            amount=amount,
            currency=str(_require(data, "currency", cls.__name__)),
            kind=data.get("kind", "retail"),
            origin=str(data.get("origin", "jsonld:offers.price")),
        )


@dataclass(frozen=True)
class WebEvidence:
    """One page excerpt exactly as shown to the text model.

    Attributes:
        source_id: Citation id, ``"W1"``, ``"W2"``, ... in rank order.
        url: Final URL after redirects, as fetched.
        domain: Registrable domain (e.g. ``buffalotracedistillery.com``).
        tier: Source class: ``official``, ``retailer`` or ``reference``.
        title: Page title, if any.
        text: The excerpt shown to the text model (at most
            :data:`MAX_EVIDENCE_CHARS` characters).
        sha256: Hex sha256 of ``text`` (UTF-8); checked on construction.
        fetched_at: ISO-8601 UTC time of the fetch.
        http_status: HTTP status of the fetch.
        query: Search query that found the page.
        from_cache: Whether the page came from the local web cache.
        truncated: Whether the page text was cut to fit the excerpt.
        structured_prices: JSON-LD offer prices (retail observations only).
    """

    source_id: str
    url: str
    domain: str
    tier: SourceClass
    title: Optional[str]
    text: str
    sha256: str
    fetched_at: str
    http_status: int
    query: str
    from_cache: bool
    truncated: bool
    structured_prices: tuple[StructuredPrice, ...] = ()

    def __post_init__(self) -> None:
        """Validate the excerpt and its hash.

        Raises:
            ValueError: If ``sha256`` does not match ``text``, the text is
                too long, the tier is unknown or the URL is not http(s).
        """
        if not isinstance(self.text, str):
            raise ValueError("WebEvidence.text must be a string")
        if len(self.text) > MAX_EVIDENCE_CHARS:
            raise ValueError(
                f"WebEvidence.text is {len(self.text)} characters; the limit is {MAX_EVIDENCE_CHARS}"
            )
        expected = sha256_text(self.text)
        if str(self.sha256).lower() != expected:
            raise ValueError(
                f"WebEvidence {self.source_id}: sha256 {self.sha256!r} does not match its text ({expected})"
            )
        _check_choice("WebEvidence.tier", self.tier, SOURCE_CLASSES)
        if not str(self.url).lower().startswith(("http://", "https://")):
            raise ValueError(f"WebEvidence.url must be http(s), got {self.url!r}")
        if not isinstance(self.structured_prices, tuple):
            object.__setattr__(self, "structured_prices", tuple(self.structured_prices))

    @classmethod
    def create(cls, *, text: str, **fields: Any) -> WebEvidence:  # noqa: ANN401
        """Build an excerpt, computing ``sha256`` from ``text``."""
        return cls(text=text, sha256=sha256_text(text), **fields)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "source_id": self.source_id,
            "url": self.url,
            "domain": self.domain,
            "tier": self.tier,
            "title": self.title,
            "text": self.text,
            "sha256": self.sha256,
            "fetched_at": self.fetched_at,
            "http_status": self.http_status,
            "query": self.query,
            "from_cache": self.from_cache,
            "truncated": self.truncated,
            "structured_prices": [p.to_dict() for p in self.structured_prices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WebEvidence:
        """Inverse of :meth:`to_dict` (re-checks the sha256).

        Raises:
            ValueError: If a key is missing or the data is inconsistent.
        """
        name = cls.__name__
        return cls(
            source_id=str(_require(data, "source_id", name)),
            url=str(_require(data, "url", name)),
            domain=str(_require(data, "domain", name)),
            tier=_require(data, "tier", name),
            title=_opt_str(data.get("title")),
            text=_require(data, "text", name),
            sha256=str(_require(data, "sha256", name)),
            fetched_at=str(_require(data, "fetched_at", name)),
            http_status=int(_require(data, "http_status", name)),
            query=str(data.get("query", "")),
            from_cache=bool(data.get("from_cache", False)),
            truncated=bool(data.get("truncated", False)),
            structured_prices=tuple(
                StructuredPrice.from_dict(p) for p in data.get("structured_prices") or ()
            ),
        )


@dataclass(frozen=True)
class PageRecord:
    """One URL considered during a bottle's web research.

    Every search hit that was looked at gets a record, so the bottle
    document can list every website consulted and why it was or was not
    used.

    Attributes:
        url: The URL (final URL if it was fetched).
        title: Page or search-result title, if known.
        domain: Registrable domain.
        source_class: ``official``, ``retailer`` or ``reference`` (None if
            it could not be classified).
        retrieved_at: ISO-8601 UTC time of the fetch (None if never fetched).
        outcome: ``cited`` (a supported fact cites it), ``fetched_not_cited``
            or ``skipped`` (never used as evidence).
        cited_for: Field names this page supports (only for ``cited``).
        reason: Why it was skipped or not cited (robots.txt, deny-listed,
            size/type cap, fetch error, not about this product, no
            verifiable quote, ...); None for a cited page.
        from_cache: Whether the page came from the local web cache.
    """

    url: str
    title: Optional[str]
    domain: str
    source_class: Optional[SourceClass]
    retrieved_at: Optional[str]
    outcome: PageOutcome
    cited_for: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    from_cache: bool = False

    def __post_init__(self) -> None:
        """Validate the outcome and the source class.

        Raises:
            ValueError: If ``outcome`` or ``source_class`` is unknown, or a
                page that is not cited lists fields it supports.
        """
        _check_choice("PageRecord.outcome", self.outcome, PAGE_OUTCOMES)
        _check_choice("PageRecord.source_class", self.source_class, SOURCE_CLASSES, optional=True)
        if not isinstance(self.cited_for, list):
            object.__setattr__(self, "cited_for", [str(f) for f in self.cited_for])
        if self.cited_for and self.outcome != "cited":
            raise ValueError(
                f"PageRecord {self.url}: cited_for {self.cited_for} requires outcome 'cited', "
                f"got {self.outcome!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "source_class": self.source_class,
            "retrieved_at": self.retrieved_at,
            "outcome": self.outcome,
            "cited_for": list(self.cited_for),
            "reason": self.reason,
            "from_cache": self.from_cache,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PageRecord:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: If a key is missing or a value is not allowed.
        """
        name = cls.__name__
        return cls(
            url=str(_require(data, "url", name)),
            title=_opt_str(data.get("title")),
            domain=str(_require(data, "domain", name)),
            source_class=data.get("source_class"),
            retrieved_at=_opt_str(data.get("retrieved_at")),
            outcome=_require(data, "outcome", name),
            cited_for=[str(f) for f in data.get("cited_for") or []],
            reason=_opt_str(data.get("reason")),
            from_cache=bool(data.get("from_cache", False)),
        )


@dataclass(frozen=True)
class VendorImage:
    """A vendor's product photo, kept as a reference for the reviewer.

    Attributes:
        cache_path: Local path of the downloaded image.
        image_url: URL the image was downloaded from.
        page_url: Page that referenced the image.
        retrieved_at: ISO-8601 UTC download time.
        verdict: The vision model's same-bottle comparison with the primary
            photo (as returned by the verifier).
        usage: Licensing note; vendor images are never published.
    """

    cache_path: str
    image_url: str
    page_url: str
    retrieved_at: str
    verdict: dict[str, Any]
    usage: str = VENDOR_IMAGE_USAGE

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "cache_path": self.cache_path,
            "image_url": self.image_url,
            "page_url": self.page_url,
            "retrieved_at": self.retrieved_at,
            "verdict": dict(self.verdict),
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VendorImage:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: If a key is missing.
        """
        name = cls.__name__
        return cls(
            cache_path=str(_require(data, "cache_path", name)),
            image_url=str(_require(data, "image_url", name)),
            page_url=str(_require(data, "page_url", name)),
            retrieved_at=str(_require(data, "retrieved_at", name)),
            verdict=dict(data.get("verdict") or {}),
            usage=str(data.get("usage") or VENDOR_IMAGE_USAGE),
        )


@dataclass
class WebResearch:
    """Everything the web layer did for one bottle.

    Attributes:
        status: ``ok`` (at least one evidence page), ``no_results``,
            ``skipped`` (web lookups off, no provider, or the identity was
            too weak to search) or ``failed`` (with ``reason``).
        reason: Why the status is not ``ok``.
        provider: Search provider used: one of :data:`SEARCH_PROVIDERS`
            (``duckduckgo``, ``brave``, ``ollama``, ``searxng``), or None
            if none ran.
        queries: Every query sent, in order.
        searched_at: ISO-8601 UTC time of the first search.
        pages: Every URL considered (see :class:`PageRecord`).
        evidence: Excerpts passed to the fact check, in rank order.
        vendor_image: Reference image (``-v``), if one was accepted.
    """

    status: ResearchStatus
    reason: Optional[str] = None
    provider: Optional[str] = None
    queries: list[str] = field(default_factory=list)
    searched_at: Optional[str] = None
    pages: list[PageRecord] = field(default_factory=list)
    evidence: list[WebEvidence] = field(default_factory=list)
    vendor_image: Optional[VendorImage] = None

    def __post_init__(self) -> None:
        """Validate the status.

        Raises:
            ValueError: If ``status`` or ``provider`` is unknown, two
                evidence excerpts share a ``source_id`` (a citation would be
                ambiguous), or the status is ``ok`` without any evidence.
        """
        _check_choice("WebResearch.status", self.status, RESEARCH_STATUSES)
        _check_choice("WebResearch.provider", self.provider, SEARCH_PROVIDERS, optional=True)
        seen: set[str] = set()
        for item in self.evidence:
            if item.source_id in seen:
                raise ValueError(f"WebResearch: duplicate evidence source_id {item.source_id!r}")
            seen.add(item.source_id)
        if self.status == "ok" and not self.evidence:
            raise ValueError("WebResearch: status 'ok' requires at least one evidence excerpt")

    @property
    def pages_consulted(self) -> int:
        """Number of URLs considered."""
        return len(self.pages)

    @property
    def pages_cited(self) -> int:
        """Number of URLs cited by a supported fact."""
        return sum(1 for p in self.pages if p.outcome == "cited")

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return {
            "status": self.status,
            "reason": self.reason,
            "provider": self.provider,
            "queries": list(self.queries),
            "searched_at": self.searched_at,
            "pages": [p.to_dict() for p in self.pages],
            "evidence": [e.to_dict() for e in self.evidence],
            "vendor_image": self.vendor_image.to_dict() if self.vendor_image is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WebResearch:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: If a key is missing or a value is not allowed.
        """
        vendor = data.get("vendor_image")
        return cls(
            status=_require(data, "status", cls.__name__),
            reason=_opt_str(data.get("reason")),
            provider=_opt_str(data.get("provider")),
            queries=[str(q) for q in data.get("queries") or []],
            searched_at=_opt_str(data.get("searched_at")),
            pages=[PageRecord.from_dict(p) for p in data.get("pages") or []],
            evidence=[WebEvidence.from_dict(e) for e in data.get("evidence") or []],
            vendor_image=VendorImage.from_dict(vendor) if vendor else None,
        )
