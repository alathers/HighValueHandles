"""Plain text, JSON-LD and link metadata from an HTML page (stdlib only).

The web layer shows the fact check *exact* excerpts of the text a page
displays, so this module only removes markup; it never rewrites words.

- ``script`` (except ``application/ld+json``, which is parsed), ``style``,
  ``noscript``, ``template`` and ``svg`` contents are skipped.
- Block-level tags become line breaks; entities are unescaped; runs of
  whitespace within a line collapse to one space; empty lines are dropped.
- The text is capped at :data:`MAX_TEXT_CHARS` characters.
- Malformed JSON-LD is ignored.

JSON-LD helpers (:func:`jsonld_products`, :func:`offer_prices`,
:func:`product_images`) read structured product data deterministically. A
JSON-LD offer price is a shop's *retail* price, never an MSRP.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any, Optional

from models.evidence import StructuredPrice

#: Longest page text kept, in characters.
MAX_TEXT_CHARS = 200_000

#: Elements whose content is never page text.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})

#: Elements that start a new line of text.
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "details",
        "dialog", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1",
        "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html", "li", "main", "nav", "ol",
        "option", "p", "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th",
        "thead", "title", "tr", "ul",
    }
)  # fmt: skip

_WS = re.compile(r"\s+")
_PRICE_TEXT = re.compile(r"^\s*(?:[A-Z]{3}\s*)?[$€£]?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*$")


@dataclass(frozen=True)
class ExtractedHtml:
    """What a page shows, without markup.

    Attributes:
        title: Text of the first ``<title>``, if any.
        text: Visible text, one block per line (at most
            :data:`MAX_TEXT_CHARS` characters).
        jsonld: Parsed ``application/ld+json`` objects (top-level lists are
            flattened).
        og_image: ``og:image`` meta content, as written on the page.
        canonical: ``<link rel="canonical">`` href, as written on the page.
        meta_description: ``<meta name="description">`` content.
        text_truncated: Whether the text was cut at :data:`MAX_TEXT_CHARS`.
    """

    title: Optional[str]
    text: str
    jsonld: list[dict[str, Any]] = field(default_factory=list)
    og_image: Optional[str] = None
    canonical: Optional[str] = None
    meta_description: Optional[str] = None
    text_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (used by the page cache)."""
        return {
            "title": self.title,
            "text": self.text,
            "jsonld": list(self.jsonld),
            "og_image": self.og_image,
            "canonical": self.canonical,
            "meta_description": self.meta_description,
            "text_truncated": self.text_truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedHtml:
        """Inverse of :meth:`to_dict`."""
        return cls(
            title=data.get("title"),
            text=str(data.get("text") or ""),
            jsonld=[d for d in data.get("jsonld") or [] if isinstance(d, dict)],
            og_image=data.get("og_image"),
            canonical=data.get("canonical"),
            meta_description=data.get("meta_description"),
            text_truncated=bool(data.get("text_truncated", False)),
        )


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.in_jsonld = False
        self.jsonld_buf: list[str] = []
        self.jsonld_raw: list[str] = []
        self.in_title = False
        self.pre_depth = 0
        self.title_parts: list[str] = []
        self.title_done = False
        self.og_image: Optional[str] = None
        self.canonical: Optional[str] = None
        self.meta_description: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        media = a.get("type", "").split(";", 1)[0].strip().lower()
        if tag == "script" and media == "application/ld+json":
            self.in_jsonld = True
            self.jsonld_buf = []
            return
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag == "meta":
            self._meta(a)
        elif tag == "link":
            rels = a.get("rel", "").lower().split()
            if "canonical" in rels and self.canonical is None and a.get("href"):
                self.canonical = a["href"].strip()
        elif tag == "title" and not self.title_done and self.skip_depth == 0:
            self.in_title = True
        if tag == "pre":
            self.pre_depth += 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        # Self-closing: never opens a skipped region.
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            self._meta(a)
        elif tag == "link":
            self.handle_starttag(tag, attrs)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def _meta(self, a: dict[str, str]) -> None:
        key = (a.get("property") or a.get("name") or "").strip().lower()
        content = a.get("content", "").strip()
        if not content:
            return
        if key in ("og:image", "og:image:url", "og:image:secure_url") and self.og_image is None:
            self.og_image = content
        elif key == "description" and self.meta_description is None:
            self.meta_description = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.in_jsonld:
            self.in_jsonld = False
            self.jsonld_raw.append("".join(self.jsonld_buf))
            return
        if tag in _SKIP_TAGS:
            if self.skip_depth > 0:
                self.skip_depth -= 1
            return
        if tag == "title" and self.in_title:
            self.in_title = False
            self.title_done = True
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_jsonld:
            self.jsonld_buf.append(data)
            return
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
            return
        if not self.pre_depth:
            # A raw newline in HTML source is just whitespace.
            data = data.replace("\r", " ").replace("\n", " ")
        self.parts.append(data)


def _clean_lines(raw: str) -> str:
    lines = (_WS.sub(" ", line).strip() for line in raw.split("\n"))
    return "\n".join(line for line in lines if line)


def drop_partial_tail(text: str) -> str:
    """``text`` without its last word, for text that was cut mid-stream.

    A cut can split a word or a number (``$1,2`` of ``$1,299.99``); the
    fragment would read as a different value, so it is dropped. Text without
    any whitespace is returned unchanged.
    """
    idx = max(text.rfind(" "), text.rfind("\n"))
    if idx <= 0:
        return text
    return text[:idx].rstrip()


def _parse_jsonld(raw: str) -> list[dict[str, Any]]:
    s = raw.strip()
    if not s:
        return []
    # Some sites wrap JSON-LD in HTML comments or CDATA markers.
    for prefix, suffix in (("<!--", "-->"), ("<![CDATA[", "]]>"), ("//<![CDATA[", "//]]>")):
        if s.startswith(prefix) and s.endswith(suffix):
            s = s[len(prefix) : -len(suffix)].strip()
    try:
        data = json.loads(s)
    except (ValueError, RecursionError):
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def extract(html: str) -> ExtractedHtml:
    """Visible text, title, JSON-LD, ``og:image`` and canonical link of a page.

    Never raises on malformed markup; returns what could be read.
    """
    parser = _TextParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001 - html.parser can raise on pathological input
        pass
    text = _clean_lines("".join(parser.parts))
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = drop_partial_tail(text[:MAX_TEXT_CHARS])
    title = _WS.sub(" ", "".join(parser.title_parts)).strip() or None
    jsonld: list[dict[str, Any]] = []
    for raw in parser.jsonld_raw:
        jsonld.extend(_parse_jsonld(raw))
    return ExtractedHtml(
        title=title,
        text=text,
        jsonld=jsonld,
        og_image=parser.og_image,
        canonical=parser.canonical,
        meta_description=parser.meta_description,
        text_truncated=truncated,
    )


# ---------------------------------------------------------------------------
# JSON-LD helpers
# ---------------------------------------------------------------------------


def _types(node: dict[str, Any]) -> set[str]:
    t = node.get("@type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    return set()


def _walk(nodes: Iterable[Any], depth: int = 0) -> Iterator[dict[str, Any]]:
    if depth > 8:
        return
    for node in nodes:
        if isinstance(node, list):
            yield from _walk(node, depth + 1)
        elif isinstance(node, dict):
            yield node
            for key in ("@graph", "mainEntity", "mainEntityOfPage", "itemListElement", "item"):
                child = node.get(key)
                if isinstance(child, (list, dict)):
                    yield from _walk([child], depth + 1)


def jsonld_products(jsonld: Iterable[Any]) -> list[dict[str, Any]]:
    """Every JSON-LD node whose ``@type`` is (or includes) ``Product``.

    Walks ``@graph``, lists, ``mainEntity`` and ``itemListElement``.
    """
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for node in _walk(list(jsonld)):
        types = {t.split("/")[-1] for t in _types(node)}
        if "Product" in types and id(node) not in seen:
            seen.add(id(node))
            out.append(node)
    return out


def _to_amount(value: Any) -> Optional[Decimal]:  # noqa: ANN401
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        m = _PRICE_TEXT.match(value)
        if not m:
            return None
        text = m.group(1).replace(",", "") + (m.group(2) or "")
    else:
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    return amount


def _offer_nodes(product: dict[str, Any]) -> list[dict[str, Any]]:
    offers = product.get("offers")
    nodes: list[dict[str, Any]] = []
    stack = [offers]
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[0:0] = item
        elif isinstance(item, dict):
            nodes.append(item)
            inner = item.get("offers")
            if isinstance(inner, (list, dict)):
                stack.append(inner)
    return nodes


def offer_prices(product: dict[str, Any]) -> list[StructuredPrice]:
    """Prices in a Product's JSON-LD ``offers`` (any currency, amount > 0).

    Reads ``price``, ``lowPrice`` and ``priceSpecification.price``. An offer
    without a ``priceCurrency`` is skipped (the currency is never assumed).
    Every result is a *retail* observation.
    """
    out: list[StructuredPrice] = []
    seen: set[tuple[Decimal, str]] = set()
    for offer in _offer_nodes(product):
        candidates: list[tuple[Any, Any, str]] = []
        currency = offer.get("priceCurrency")
        for key in ("price", "lowPrice"):
            if key in offer:
                candidates.append((offer.get(key), currency, f"jsonld:offers.{key}"))
        spec = offer.get("priceSpecification")
        for s in spec if isinstance(spec, list) else [spec]:
            if isinstance(s, dict) and "price" in s:
                candidates.append(
                    (s.get("price"), s.get("priceCurrency") or currency,
                     "jsonld:offers.priceSpecification.price")
                )  # fmt: skip
        for raw, cur, origin in candidates:
            amount = _to_amount(raw)
            if amount is None or not isinstance(cur, str) or not cur.strip():
                continue
            key = (amount, cur.strip().upper())
            if key in seen:
                continue
            seen.add(key)
            out.append(StructuredPrice(amount=amount, currency=key[1], origin=origin))
    return out


def product_images(product: dict[str, Any]) -> list[str]:
    """Image URLs of a JSON-LD Product, as written (may be relative)."""
    out: list[str] = []

    def add(item: Any) -> None:  # noqa: ANN401
        if isinstance(item, str):
            url = item.strip()
            if url and url not in out:
                out.append(url)
        elif isinstance(item, dict):
            add(item.get("url") or item.get("contentUrl"))
        elif isinstance(item, list):
            for x in item:
                add(x)

    add(product.get("image"))
    return out
