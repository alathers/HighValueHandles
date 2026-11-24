"""Strict, deterministic checks of everything the text model proposes (pure).

Nothing the model says is trusted. A claim survives only if its quote is
found verbatim (after :func:`normalize`) in the source it cites, the value is
literally supported by that quote, role and page-match rules hold, and no URL
was invented. There is no fuzzy matching: a rejected true fact is cheap (it
becomes Unknown), an accepted false one is published.

Rejection reason codes:

- source: ``unknown_source``
- quote: ``quote_too_short``, ``quote_too_long``, ``elided_quote``,
  ``quote_not_found``, ``quote_in_other_source``
- value: ``number_mismatch``, ``unit_cue_missing``, ``value_not_in_quote``
- role: ``role_cue_missing``
- page: ``page_not_matched``, ``release_not_near_quote``
- price: ``amount_mismatch``, ``amount_out_of_range``, ``msrp_cue_missing``,
  ``secondary_cue_missing``
- any: ``url_in_output``
- copy: ``no_citation``, ``bad_citation``, ``no_fields``, ``unverified_field``,
  ``number_not_cited``, ``unverifiable_claim``, ``unsupported_name``,
  ``url_in_output``
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Union

from image_processor.fact_prompt import (
    QUOTE_MAX_CHARS,
    QUOTE_MIN_CHARS,
    Claim,
    EvidencePack,
    NumberedField,
    PageMatch,
    PriceClaim,
)
from models.bottle import parse_money
from models.facts import Citation

# ---------------------------------------------------------------------------
# Normalization and tokens
# ---------------------------------------------------------------------------

_CHAR_MAP = {
    # quotes and primes -> ASCII
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "‵": "'",
    "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "‶": '"',
    "«": '"', "»": '"',
    # dashes and minus -> hyphen
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "﹘": "-", "﹣": "-", "－": "-",
    # spaces
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    # invisible characters and soft hyphen -> nothing
    "\u200b": "", "‌": "", "‍": "", "⁠": "", "﻿": "", "­": "",
}
_TRANSLATE = str.maketrans(_CHAR_MAP)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
_DOLLAR_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?![\d])")
_URL_RE = re.compile(r"https?:|www\.|\]\(", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<![\w.])(19\d{2}|20\d{2})(?![\w])")

MSRP_CUE = r"\bmsrp\b|\bsrp\b|suggested (retail )?price|suggested retail|recommended retail|retail price of|will retail for"
_MSRP_CUE_RE = re.compile(MSRP_CUE)
_SECONDARY_CUE_RE = re.compile(r"auction|sold for|hammer|resale|secondary")
_DISTILL_CUE_RE = re.compile(r"distilled by|distillery|\bdsp\b")
_BOTTLER_CUE_RE = re.compile(r"bottled by|produced by")
_BANNED_RE = re.compile(
    r"\b(best|smoothest|rare|legendary|award\w*|sought-after|allocated|unicorn|iconic|limited)\b"
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "and", "or", "by", "in", "on", "at", "to", "for", "with", "from",
        "is", "it", "its", "s", "co", "company", "inc", "llc", "ltd",
    }
)
_BATCH_WORDS: frozenset[str] = frozenset({"batch", "no", "number", "lot", "num"})

#: Fields whose value is tied to one release (page must match exactly).
BATCH_SPECIFIC_FIELDS: frozenset[str] = frozenset({"proof", "abv", "age", "batch_number", "release_year"})
#: Characters around a quote searched for a label release token.
RELEASE_WINDOW = 600
#: Characters around a price quote searched for an MSRP or resale cue.
PRICE_CUE_WINDOW = 200
#: Proof read from an ABV statement may differ from 2*ABV by this much.
PROOF_FROM_ABV_TOLERANCE = Decimal("0.1")
_CENT = Decimal("0.01")


def normalize(text: Any) -> str:
    """NFKC, casefold, ASCII quotes/dashes, drop invisibles, collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.casefold()
    s = s.translate(_TRANSLATE)
    return _WS_RE.sub(" ", s).strip()


def tokens(text: Any) -> list[str]:
    """Alphanumeric tokens of the normalized text."""
    return _TOKEN_RE.findall(normalize(text))


def significant_tokens(text: Any) -> list[str]:
    """Tokens that are not stopwords and not single letters."""
    return [t for t in tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def batch_tokens(text: Any) -> list[str]:
    """Identifier tokens of a batch number ('Batch No. C923' -> ['c923'])."""
    return [t for t in tokens(text) if t not in _BATCH_WORDS]


def numbers_in(text: Any) -> list[Decimal]:
    """Numbers in the normalized text (thousands separators removed)."""
    out = []
    for m in _NUMBER_RE.finditer(normalize(text)):
        whole = m.group(1).replace(",", "")
        out.append(Decimal(f"{whole}.{m.group(2)}" if m.group(2) else whole))
    return out


def dollar_amounts(text: Any) -> list[Decimal]:
    """``$`` amounts in the normalized text."""
    out = []
    for m in _DOLLAR_RE.finditer(normalize(text)):
        whole = m.group(1).replace(",", "")
        out.append(Decimal(f"{whole}.{m.group(2)}" if m.group(2) else whole))
    return out


def has_url(text: Any) -> bool:
    """Whether the text contains ``http``/``https``/``www.`` or a markdown link."""
    return bool(_URL_RE.search(normalize(text)))


def first_number(value: Any) -> Decimal | None:
    """The first number in ``value`` (numbers are taken as they are)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            d = Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
        except InvalidOperation:
            return None
        return d if d.is_finite() else None
    nums = numbers_in(value)
    return nums[0] if nums else None


_AGE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(?:years?|yrs?)\b(?:\s*(?:,|and)?\s*(\d+)\s*-?\s*months?\b)?"
)
_MONTHS_ONLY_RE = re.compile(r"(?<![\w.])(\d+)\s*-?\s*months?\b")


def age_months_in(text: Any) -> list[int]:
    """Ages stated in ``text`` ('12 years', '6 yr', '10-year', '4 years 3 months'), in months."""
    s = normalize(text)
    out: list[int] = []
    spans: list[tuple[int, int]] = []
    for m in _AGE_RE.finditer(s):
        months = Decimal(m.group(1)) * 12 + (Decimal(m.group(2)) if m.group(2) else 0)
        out.append(int(months))
        spans.append(m.span())
    for m in _MONTHS_ONLY_RE.finditer(s):
        if not any(a <= m.start() < b for a, b in spans):
            out.append(int(m.group(1)))
    return out


def age_value_months(value: Any) -> int | None:
    """Months of an age value ('12 years'; a bare number means years)."""
    found = age_months_in(value)
    if found:
        return found[0]
    n = first_number(value)
    if n is not None and normalize(value).replace(".", "", 1).isdigit():
        return int(n * 12)
    return None


_NULL_VALUES = frozenset(
    {"", "null", "none", "n/a", "na", "unknown", "not visible", "not shown", "-",
     "unknown bottle", "unknown distillery", "not stated"}
)


def canonical_value(field_name: str, raw: Any) -> Any:
    """The typed value of a field, or None for null-like input.

    proof/abv -> float, release_year -> int, msrp -> Decimal, everything
    else a whitespace-collapsed string. A number field without a number is
    None.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str) and normalize(raw) in _NULL_VALUES:
        return None
    if field_name in ("proof", "abv"):
        n = first_number(raw)
        return float(n) if n is not None and n > 0 else None
    if field_name == "release_year":
        m = _YEAR_RE.search(normalize(raw)) if not isinstance(raw, int) else None
        if isinstance(raw, int):
            return raw if 1800 <= raw <= 2100 else None
        return int(m.group(1)) if m else None
    if field_name == "msrp":
        amount = parse_money(raw)
        return amount if amount is not None and amount > 0 else None
    text = _WS_RE.sub(" ", str(raw)).strip()
    return text or None


def label_release_tokens(label: Mapping[str, Any]) -> list[str]:
    """Normalized release identifiers of the bottle in hand.

    Four-digit years in the name or specific_release, plus the batch-number
    tokens.
    """
    out: list[str] = []
    for key in ("name", "specific_release"):
        for m in _YEAR_RE.finditer(normalize(label.get(key) or "")):
            if m.group(1) not in out:
                out.append(m.group(1))
    batch = canonical_value("batch_number", label.get("batch_number"))
    if batch:
        for tok in batch_tokens(batch):
            if tok not in out:
                out.append(tok)
    return out


def _token_in(token: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


# A unit must touch its number ("115 proof", "proof: 115", "57.5% ABV",
# "ABV 57.5", "alc. 57.5% by vol."): a number elsewhere in a quote that merely
# contains the unit word ("133.2 proof, aged 12 years" -> 12) is not a proof.
_PROOF_AFTER_RE = re.compile(r"\s*-?\s*(?:°\s*)?(?:us\s+)?proof\b")
_PROOF_BEFORE_RE = re.compile(r"\bproof\s*(?::|=|of|is|at|-|\()?\s*$")
_ABV_AFTER_RE = re.compile(r"\s*(?:%|percent\b|abv\b|alc\b)")
_ABV_BEFORE_RE = re.compile(
    r"(?:\babv|\balc\.?(?:\s*/\s*vol\.?)?|\balcohol(?:\s+by\s+volume)?)\s*(?::|=|of|is|at|-|\()?\s*$"
)
#: A number followed by another unit ("Barrel Proof 12 Year") is not the proof.
_OTHER_UNIT_AFTER_RE = re.compile(r"\s*-?\s*(?:years?|yrs?|months?|year-old|%|ml|l\b|liters?|cases?|barrels?)")
#: Words that make a year a founding date, not a release year.
_FOUNDED_BEFORE_RE = re.compile(r"(?:\best\.?|\bestablished|\bfounded|\bsince|\bcirca|\bopened|\bbuilt)\s*(?:in\s+)?$")


def number_with_unit(text: str, value: Decimal, field_name: str) -> bool:
    """Whether ``value`` occurs in normalized ``text`` directly tied to its unit.

    proof: ``115 proof``, ``115-proof``, ``proof: 115``. abv: ``57.5%``,
    ``57.5 ABV``, ``ABV 57.5``, ``alc. 57.5``. Other fields: plain presence.
    """
    for m in _NUMBER_RE.finditer(text):
        whole = m.group(1).replace(",", "")
        n = Decimal(f"{whole}.{m.group(2)}" if m.group(2) else whole)
        if n != value:
            continue
        after = text[m.end() : m.end() + 16]
        before = text[max(0, m.start() - 28) : m.start()]
        other_unit = _OTHER_UNIT_AFTER_RE.match(after) is not None
        if field_name == "proof":
            if _PROOF_AFTER_RE.match(after) or (_PROOF_BEFORE_RE.search(before) and not other_unit):
                return True
        elif field_name == "abv":
            if _ABV_AFTER_RE.match(after) or (_ABV_BEFORE_RE.search(before) and not other_unit):
                return True
        else:
            return True
    return False


def year_stated_as_release(text: str, year: int) -> bool:
    """Whether ``year`` occurs in normalized ``text`` other than as a founding date."""
    for m in _YEAR_RE.finditer(text):
        if int(m.group(1)) != year:
            continue
        if not _FOUNDED_BEFORE_RE.search(text[max(0, m.start() - 20) : m.start()]):
            return True
    return False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Accepted:
    """A claim that passed every check."""

    claim: Claim
    value: Any
    citation: Citation


@dataclass(frozen=True)
class AcceptedPrice:
    """A price that passed every check (``kind`` may have been downgraded)."""

    price: PriceClaim
    kind: str
    citation: Citation
    downgraded: bool = False


@dataclass(frozen=True)
class Rejected:
    """A proposal Python refused, with a reason code."""

    item: Any
    reason: str
    detail: str = ""

    def to_record(self, stage: str = "extract") -> dict[str, Any]:
        """Plain-JSON record for ``Verification.rejected``."""
        rec: dict[str, Any] = {"stage": stage, "reason": self.reason, "detail": self.detail}
        it = self.item
        if isinstance(it, Claim):
            rec.update(kind="claim", field=it.field, value=it.value, source_id=it.source_id, quote=it.quote)
        elif isinstance(it, PriceClaim):
            rec.update(
                kind="price",
                field="msrp" if it.kind == "msrp" else f"price:{it.kind}",
                value=str(it.amount),
                currency=it.currency,
                source_id=it.source_id,
                quote=it.quote,
            )
        elif isinstance(it, PageMatch):
            rec.update(kind="page_match", source_id=it.source_id, value=it.verdict, quote=it.quote)
        else:
            rec.update(kind="other", value=str(it)[:300])
        return rec


ClaimResult = Union[Accepted, Rejected]
PriceResult = Union[AcceptedPrice, Rejected]


# ---------------------------------------------------------------------------
# Quote and page checks
# ---------------------------------------------------------------------------


def _quote_problem(quote: str | None, source_text: str) -> str | None:
    """Reason code if ``quote`` is not a valid verbatim quote of ``source_text``."""
    if not isinstance(quote, str):
        return "quote_too_short"
    stripped = quote.strip()
    if "..." in stripped or "…" in stripped:
        return "elided_quote"
    if len(stripped) < QUOTE_MIN_CHARS:
        return "quote_too_short"
    if len(stripped) > QUOTE_MAX_CHARS:
        return "quote_too_long"
    nq = normalize(stripped)
    if len(nq) < QUOTE_MIN_CHARS:
        return "quote_too_short"
    if nq not in normalize(source_text):
        return "quote_not_found"
    return None


def _check_quote(source_id: str, quote: str, pack: EvidencePack) -> tuple[str | None, str]:
    """(reason, detail) for the source and quote of a claim or price; (None, '') if fine."""
    src = pack.get(source_id)
    if src is None:
        return "unknown_source", f"{source_id!r} is not a citable source of this bottle"
    problem = _quote_problem(quote, src.text)
    if problem == "quote_not_found":
        nq = normalize(quote)
        others = [s.id for s in pack.sources.values() if s.id != source_id and nq in normalize(s.text)]
        if others:
            return "quote_in_other_source", f"quote is in {', '.join(others)}, not in {source_id}"
        return "quote_not_found", f"quote is not in {source_id}"
    if problem:
        return problem, ""
    return None, ""


def page_matches(source_id: str, pack: EvidencePack, page_match: Mapping[str, PageMatch]) -> tuple[bool, str]:
    """Whether web page ``source_id`` was judged ``exact`` with a valid quote."""
    pm = page_match.get(source_id)
    if pm is None:
        return False, f"no page_match verdict for {source_id}"
    if pm.verdict != "exact":
        return False, f"{source_id} page_match={pm.verdict}"
    src = pack.get(source_id)
    problem = _quote_problem(pm.quote, src.text if src else "")
    if problem:
        return False, f"{source_id} page_match quote failed: {problem}"
    return True, ""


def _release_near(
    source_text: str, quote: str, release_tokens: Sequence[str], window: int = RELEASE_WINDOW
) -> bool:
    if not release_tokens:
        return True
    ns = normalize(source_text)
    nq = normalize(quote)
    pos = ns.find(nq)
    if pos < 0:
        return False
    region = ns[max(0, pos - window) : pos + len(nq) + window]
    return any(_token_in(tok, region) for tok in release_tokens)


def _cue_near(source_text: str, quote: str, cue: re.Pattern[str], window: int = PRICE_CUE_WINDOW) -> bool:
    ns = normalize(source_text)
    nq = normalize(quote)
    if cue.search(nq):
        return True
    pos = ns.find(nq)
    if pos < 0:
        return False
    return cue.search(ns[max(0, pos - window) : pos + len(nq) + window]) is not None


def _msrp_cue_binds(source_text: str, quote: str, amount: Decimal, window: int = PRICE_CUE_WINDOW) -> bool:
    """Whether an MSRP cue near the quote refers to ``amount``.

    The cue must be in the quote or within ``window`` characters of it, and
    the ``$`` amount nearest to that cue must be this amount: on
    "MSRP $59.99. Our price $64.99." the cue binds to $59.99, so $64.99 is
    not an MSRP.
    """
    ns = normalize(source_text)
    nq = normalize(quote)
    pos = ns.find(nq)
    if pos < 0:
        return False
    lo, hi = max(0, pos - window), pos + len(nq) + window
    amounts = []
    for m in _DOLLAR_RE.finditer(ns):
        whole = m.group(1).replace(",", "")
        amounts.append((m.start(), m.end(), Decimal(f"{whole}.{m.group(2)}" if m.group(2) else whole)))
    if not amounts:
        return False
    target = amount.quantize(_CENT)
    for cue in _MSRP_CUE_RE.finditer(ns, lo, hi):

        def dist(a: tuple[int, int, Decimal], c: re.Match[str] = cue) -> tuple[int, int]:
            if a[0] >= c.end():
                return (a[0] - c.end(), 0)  # amount after the cue wins a tie
            return (max(0, c.start() - a[1]), 1)

        nearest = min(amounts, key=dist)
        if nearest[2].quantize(_CENT) == target:
            return True
    return False


def as_page_match_map(page_match: Mapping[str, PageMatch] | Iterable[PageMatch] | None) -> dict[str, PageMatch]:
    """``source_id -> PageMatch`` (the first verdict per page wins)."""
    if page_match is None:
        return {}
    if isinstance(page_match, Mapping):
        return dict(page_match)
    out: dict[str, PageMatch] = {}
    for pm in page_match:
        out.setdefault(pm.source_id, pm)
    return out


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def _value_check(claim: Claim) -> tuple[str | None, str, Any, bool]:
    """(reason, detail, canonical value, derived) for the value-vs-quote rules."""
    f = claim.field
    nq = normalize(claim.quote)
    if f in ("proof", "abv", "release_year"):
        value = canonical_value(f, claim.value)
        v = first_number(claim.value)
        if value is None or v is None:
            return "number_mismatch", f"no usable number in value {claim.value!r}", None, False
        if f == "release_year":
            if int(v) != v or not _YEAR_RE.fullmatch(str(int(v))):
                return "number_mismatch", f"{claim.value!r} is not a four-digit year", None, False
            if Decimal(int(v)) not in numbers_in(claim.quote):
                return "number_mismatch", f"{int(v)} is not in the quote", None, False
            if not year_stated_as_release(nq, int(v)):
                return "number_mismatch", f"{int(v)} is stated as a founding date, not a release", None, False
            return None, "", value, False
        quote_numbers = numbers_in(claim.quote)
        if v in quote_numbers:
            if not number_with_unit(nq, v, f):
                unit = "'proof'" if f == "proof" else "'%', 'ABV' or 'alc'"
                return "unit_cue_missing", f"{v} is not stated next to {unit} in the quote", None, False
            return None, "", value, False
        if f == "proof":
            for pct in _PERCENT_RE.findall(nq):
                if abs(v - 2 * Decimal(pct)) <= PROOF_FROM_ABV_TOLERANCE:
                    return None, f"derived from {pct}% ABV", value, True
        return "number_mismatch", f"{v} is not a number in the quote", None, False
    if f == "age":
        months = age_value_months(claim.value)
        if months is None:
            return "number_mismatch", f"no age in value {claim.value!r}", None, False
        if months in age_months_in(claim.quote):
            return None, "", canonical_value(f, claim.value), False
        return "number_mismatch", f"age {claim.value!r} is not stated in the quote", None, False
    value = canonical_value(f, claim.value)
    if value is None:
        return "value_not_in_quote", "empty value", None, False
    quote_tokens = set(tokens(claim.quote))
    if f == "batch_number":
        need = batch_tokens(value)
    else:
        need = significant_tokens(value)
    if not need:
        return "value_not_in_quote", "value has no identifying tokens", None, False
    missing = [t for t in need if t not in quote_tokens]
    if missing:
        return "value_not_in_quote", f"not in quote: {', '.join(missing)}", None, False
    return None, "", value, False


def verify_claim(
    claim: Claim,
    pack: EvidencePack,
    page_match: Mapping[str, PageMatch] | Iterable[PageMatch] | None,
    label_release: Sequence[str] | None = None,
) -> ClaimResult:
    """Check one claim against the pack.

    Args:
        claim: The model's claim.
        pack: The evidence pack the model saw.
        page_match: The model's page verdicts (map or list).
        label_release: Release tokens of the bottle in hand (default: from
            ``pack.label``).

    Returns:
        :class:`Accepted` with the typed value and citation, or :class:`Rejected`.
    """
    pm = as_page_match_map(page_match)
    if has_url(claim.value) or has_url(claim.quote):
        return Rejected(claim, "url_in_output", "a claim may not contain a URL")
    reason, detail = _check_quote(claim.source_id, claim.quote, pack)
    if reason:
        return Rejected(claim, reason, detail)
    reason, detail, value, derived = _value_check(claim)
    if reason:
        return Rejected(claim, reason, detail)
    nq = normalize(claim.quote)
    if claim.field == "distillery" and not _DISTILL_CUE_RE.search(nq):
        return Rejected(claim, "role_cue_missing", "distillery needs 'distilled by', 'distillery' or 'DSP'")
    if claim.field == "bottler" and not _BOTTLER_CUE_RE.search(nq):
        return Rejected(claim, "role_cue_missing", "bottler needs 'bottled by' or 'produced by'")
    src = pack.get(claim.source_id)
    if src is not None and src.kind == "web":
        verdict = pm[claim.source_id].verdict if claim.source_id in pm else None
        if verdict == "different_product":
            return Rejected(claim, "page_not_matched", f"{claim.source_id} page_match=different_product")
    if src is not None and src.kind == "web" and claim.field in BATCH_SPECIFIC_FIELDS:
        ok, why = page_matches(claim.source_id, pack, pm)
        if not ok:
            return Rejected(claim, "page_not_matched", why)
        tokens_ = list(label_release) if label_release is not None else label_release_tokens(pack.label)
        if not _release_near(src.text, claim.quote, tokens_):
            return Rejected(
                claim,
                "release_not_near_quote",
                f"none of {', '.join(tokens_)} within {RELEASE_WINDOW} characters of the quote",
            )
    citation = Citation(claim.source_id, claim.quote.strip(), "verbatim", derived)
    return Accepted(claim, value, citation)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def verify_price(
    price: PriceClaim,
    pack: EvidencePack,
    page_match: Mapping[str, PageMatch] | Iterable[PageMatch] | None,
    label_release: Sequence[str] | None = None,
) -> PriceResult:
    """Check one price.

    - The amount must be a ``$`` amount in the quote, to the cent, and
      ``0 < amount < 100000``.
    - ``msrp`` needs an MSRP cue in or within 200 characters of the quote;
      otherwise it is downgraded to ``retail`` on a retailer page and
      rejected elsewhere. A web MSRP also needs an exact page match and a
      label release token near the quote.
    - ``secondary`` needs an auction or resale cue.
    - A price on a page judged ``different_product`` is rejected.
    """
    pm = as_page_match_map(page_match)
    if has_url(price.quote):
        return Rejected(price, "url_in_output", "a price quote may not contain a URL")
    reason, detail = _check_quote(price.source_id, price.quote, pack)
    if reason:
        return Rejected(price, reason, detail)
    amount = price.amount
    if not (Decimal(0) < amount < Decimal(100000)):
        return Rejected(price, "amount_out_of_range", f"{amount} is outside 0..100000")
    if amount != amount.quantize(_CENT) or not any(
        amount.quantize(_CENT) == a.quantize(_CENT) for a in dollar_amounts(price.quote)
    ):
        return Rejected(price, "amount_mismatch", f"${amount} is not a $ amount in the quote")
    src = pack.get(price.source_id)
    assert src is not None  # _check_quote guarantees it
    citation = Citation(price.source_id, price.quote.strip(), "verbatim", False)
    is_web = src.kind == "web"
    if is_web:
        verdict = pm.get(price.source_id).verdict if price.source_id in pm else None
        if verdict == "different_product":
            return Rejected(price, "page_not_matched", f"{price.source_id} page_match=different_product")

    if price.kind == "msrp":
        if not _msrp_cue_binds(src.text, price.quote, amount):
            if src.tier == "retailer":
                return AcceptedPrice(price, "retail", citation, downgraded=True)
            return Rejected(price, "msrp_cue_missing", "no MSRP/suggested-retail wording near the quote")
        if is_web:
            ok, why = page_matches(price.source_id, pack, pm)
            if not ok:
                return Rejected(price, "page_not_matched", why)
            tokens_ = list(label_release) if label_release is not None else label_release_tokens(pack.label)
            if not _release_near(src.text, price.quote, tokens_):
                return Rejected(
                    price,
                    "release_not_near_quote",
                    f"none of {', '.join(tokens_)} within {RELEASE_WINDOW} characters of the quote",
                )
        return AcceptedPrice(price, "msrp", citation)
    if price.kind == "secondary" and not _cue_near(src.text, price.quote, _SECONDARY_CUE_RE):
        return Rejected(price, "secondary_cue_missing", "no auction/resale wording near the quote")
    return AcceptedPrice(price, price.kind, citation)


# ---------------------------------------------------------------------------
# OCR backing
# ---------------------------------------------------------------------------

_FIELD_LINE_CUES: dict[str, re.Pattern[str]] = {
    "proof": re.compile(r"proof"),
    "abv": re.compile(r"%|abv|alc"),
}


def ocr_backing(label_value: Any, field_name: str, ocr_texts: Mapping[str, str]) -> Citation | None:
    """A verbatim OCR line that backs a label value, or None.

    Every number of the value (compared numerically) and every significant
    token must appear on one normalized line of one OCR text; proof and ABV
    lines also need their unit ('proof'; '%'/'ABV'/'alc'), and an age line
    must state the same age. The citation quote is that OCR line as printed.

    Args:
        label_value: The vision reading of the field.
        field_name: Field name.
        ocr_texts: OCR source id (``O1``, ...) -> OCR text.
    """
    value = canonical_value(field_name, label_value)
    if value is None:
        return None
    if field_name in ("proof", "abv", "release_year"):
        need_numbers = [first_number(value)]
        need_tokens: list[str] = []
    elif field_name == "batch_number":
        need_numbers = []
        need_tokens = batch_tokens(value)
    elif field_name == "age":
        need_numbers = []
        need_tokens = []
    elif field_name == "msrp":
        return None
    else:
        need_tokens = significant_tokens(value)
        need_numbers = []
    if not need_numbers and not need_tokens and field_name != "age":
        return None
    months = age_value_months(value) if field_name == "age" else None
    if field_name == "age" and months is None:
        return None
    for source_id, text in ocr_texts.items():
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            nline = normalize(line)
            if not nline:
                continue
            cue = _FIELD_LINE_CUES.get(field_name)
            if cue is not None and not cue.search(nline):
                continue
            if field_name == "age" and months not in age_months_in(nline):
                continue
            line_numbers = numbers_in(nline)
            if any(n is None or n not in line_numbers for n in need_numbers):
                continue
            if field_name in ("proof", "abv") and not all(
                n is not None and number_with_unit(nline, n, field_name) for n in need_numbers
            ):
                continue
            line_tokens = set(_TOKEN_RE.findall(nline))
            if any(t not in line_tokens for t in need_tokens):
                continue
            return Citation(source_id, line, "verbatim", False)
    return None


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


@dataclass
class CopyCheck:
    """Outcome of :func:`verify_copy` (all or nothing)."""

    ok: bool
    sentences: list[dict[str, Any]]
    failures: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def _allowed_numbers(nf: NumberedField) -> set[Decimal]:
    allowed: set[Decimal] = set(numbers_in(str(nf.value)))
    v = first_number(nf.value)
    if v is not None:
        allowed.add(v)
        if nf.field == "proof":
            allowed.add(v / 2)
        if nf.field == "abv":
            allowed.add(v * 2)
        if nf.field == "msrp":
            allowed.add(v.quantize(_CENT))
    return {a.normalize() for a in allowed}


def _sentence_failures(sentence: Mapping[str, Any], by_index: Mapping[int, NumberedField]) -> list[tuple[str, str]]:
    text = str(sentence.get("text") or "")
    cits = list(sentence.get("citations") or [])
    fields_ = list(sentence.get("fields") or [])
    out: list[tuple[str, str]] = []
    if has_url(text):
        out.append(("url_in_output", "sentence contains a URL or link"))
    if not cits:
        out.append(("no_citation", "sentence cites nothing"))
    bad = [c for c in cits if not isinstance(c, int) or isinstance(c, bool) or c not in by_index]
    if bad:
        out.append(("bad_citation", f"citation(s) {bad} do not exist"))
    cited = [by_index[c] for c in cits if c in by_index and not isinstance(c, bool)]
    cited_fields = {nf.field for nf in cited}
    if not fields_:
        out.append(("no_fields", "sentence lists no fields"))
    extra = [f for f in fields_ if f not in cited_fields]
    if extra:
        out.append(("unverified_field", f"field(s) {extra} not among the cited supported fields"))
    allowed: set[Decimal] = set()
    for nf in cited:
        allowed |= _allowed_numbers(nf)
    stray = [str(n) for n in numbers_in(text) if n.normalize() not in allowed]
    if stray:
        out.append(("number_not_cited", f"number(s) {', '.join(stray)} not in a cited field"))
    names = _unsupported_names(text, cited)
    if names:
        out.append(("unsupported_name", f"name(s) not in a cited field: {'; '.join(names)}"))
    cited_values = " ".join(normalize(nf.value) for nf in cited)
    banned = sorted({m.group(1) for m in _BANNED_RE.finditer(normalize(text))} - set())
    unsupported = [w for w in banned if not _token_in_text(w, cited_values)]
    if unsupported:
        out.append(("unverifiable_claim", f"unverifiable word(s): {', '.join(unsupported)}"))
    return out


_CAP_WORD_RE = re.compile(r"^[\"'(\[]*[A-Z][\w'’.&-]*[\"'),;:.!?\]]*$")


def _unsupported_names(text: str, cited: Sequence[NumberedField]) -> list[str]:
    """Capitalized multi-word phrases (not at sentence start) absent from the cited facts.

    Conservative heuristic: every significant token of a run of two or more
    capitalized words after the first word must appear in a cited field's
    value or provenance quote ("Kentucky Straight Bourbon" is rejected unless a
    cited fact says it).
    """
    allowed = set(tokens(" ".join(f"{nf.value} {nf.provenance}" for nf in cited)))
    words = text.split()
    runs: list[list[str]] = []
    run: list[str] = []
    for i, w in enumerate(words):
        if i > 0 and _CAP_WORD_RE.match(w):
            run.append(w)
            if w[-1] in ".!?;:,)":
                runs.append(run)
                run = []
            continue
        if run:
            runs.append(run)
        run = []
    if run:
        runs.append(run)
    out = []
    for r in runs:
        if len(r) < 2:
            continue
        missing = [t for t in significant_tokens(" ".join(r)) if t not in allowed]
        if missing:
            out.append(" ".join(r).strip("\"'()[].,;:!?"))
    return out


def _token_in_text(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def verify_copy(sentences: Sequence[Mapping[str, Any]], numbered_fields: Sequence[NumberedField]) -> CopyCheck:
    """Check composed sentences against the numbered fields, all or nothing.

    Every sentence needs valid citations, may only use cited supported
    fields, may only contain numbers from cited fields (proof/ABV
    conversions allowed), no URL and no unverifiable marketing word unless
    it is inside a cited value. One failing sentence fails the whole copy.
    """
    by_index = {nf.index: nf for nf in numbered_fields}
    clean = [dict(s) for s in sentences]
    if not clean:
        return CopyCheck(False, [], [], "no sentences")
    failures: list[dict[str, Any]] = []
    for i, sentence in enumerate(clean, start=1):
        for code, detail in _sentence_failures(sentence, by_index):
            failures.append(
                {"stage": "compose", "kind": "copy", "sentence": i, "text": str(sentence.get("text") or ""),
                 "reason": code, "detail": detail}
            )
    if failures:
        summary = "; ".join(f"sentence {f['sentence']}: {f['reason']}" for f in failures)
        return CopyCheck(False, clean, failures, f"copy rejected: {summary}")
    return CopyCheck(True, clean, [], None)


def render_description(sentences: Sequence[Mapping[str, Any]]) -> str:
    """Sentences with inline ``[n]`` citation markers (Python, never the model)."""
    parts = []
    for s in sentences:
        marks = "".join(f"[{c}]" for c in s.get("citations") or [])
        parts.append(f"{str(s.get('text') or '').strip()} {marks}".strip())
    return " ".join(parts)
