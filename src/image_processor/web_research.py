"""Web evidence builder: search, fetch, classify and excerpt pages (no LLM).

For one bottle identity (label fields only) :meth:`WebResearcher.research`:

0. skips when there is no provider, the provider was disabled earlier in
   the run, or the identity is too uncertain to search (placeholder name or
   ``vision_confidence == 'low'``);
1. builds at most ``max_queries`` deterministic queries from the label
   fields (:func:`build_queries`);
2. searches, de-duplicates, drops deny-listed (and, with an allow list,
   non-allowed) domains, and ranks the brand's official site first;
3. fetches at most ``max_pages`` pages through the polite fetcher and
   classifies each as ``official``, ``retailer`` or ``reference``;
4. drops pages that do not mention every distinctive name token;
5. cuts exact excerpts (windows around the name and price/release
   keywords; never paraphrased, never with inserted ellipses) and returns
   them as :class:`~models.evidence.WebEvidence` ``W1``, ``W2``, ...;
6. optionally finds a vendor reference image, accepted only on a
   high-confidence same-product verdict from the vision verifier.

Every URL considered is logged as a :class:`~models.evidence.PageRecord`.
Pages handed to the fact check are recorded as ``fetched_not_cited``; the
fact check later marks the ones it cites.

Nothing here invents or infers facts; the fact check (gpt-oss plus a
deterministic verifier) decides what the excerpts support. Public calls
never raise: any failure becomes ``WebResearch(status='failed')``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
import unicodedata
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from lib.html_text import jsonld_products, offer_prices, product_images
from lib.search_providers import (
    SearchHit,
    SearchProvider,
    SearchProviderAuthError,
    SearchProviderDisabledError,
    SearchProviderError,
)
from models.evidence import (
    MAX_EVIDENCE_CHARS,
    SEARCH_PROVIDERS,
    PageRecord,
    StructuredPrice,
    VendorImage,
    WebEvidence,
    WebResearch,
)

logger = logging.getLogger(__name__)

#: Domains never used as evidence: social, video, forums, Q&A, auctions and
#: resale. Matches the domain and its sub-domains. Editable.
DENY_DOMAINS: tuple[str, ...] = (
    # social / video / forums / Q&A
    "reddit.com", "facebook.com", "fb.com", "instagram.com", "x.com", "twitter.com",
    "tiktok.com", "youtube.com", "youtu.be", "pinterest.com", "quora.com", "threads.net",
    "linkedin.com", "tumblr.com", "snapchat.com", "discord.com", "vimeo.com",
    "straightbourbon.com",
    # auctions / resale / classifieds
    "ebay.com", "unicornauctions.com", "whiskyauctioneer.com", "scotchwhiskyauctions.com",
    "catawiki.com", "liveauctioneers.com", "invaluable.com", "bidspotter.com",
    "hibid.com", "proxibid.com", "craigslist.org", "mercari.com", "offerup.com",
    "poshmark.com", "etsy.com", "whiskyhammer.com", "justwhisky.com",
    "bourbonauctions.com", "unicorn-auctions.com", "sothebys.com", "christies.com",
    "bonhams.com", "worthpoint.com",
)  # fmt: skip

#: Words that never make a name token distinctive.
NAME_STOPWORDS = frozenset(
    {"bourbon", "whiskey", "whisky", "straight", "kentucky", "the", "small", "batch",
     "single", "barrel", "proof"}
)  # fmt: skip

#: Words that never make a domain "official" (brand tokens exclude these).
BRAND_STOPWORDS = NAME_STOPWORDS | frozenset(
    {"distillery", "distilling", "distillers", "company", "co", "inc", "llc", "and",
     "old", "reserve", "special", "private", "select", "selection", "release", "edition",
     "limited", "original", "rare", "family", "estate", "bottled", "bond", "year", "years",
     "aged", "cask", "strength", "wheat", "wheated", "american", "tennessee", "black",
     "label", "gold", "blue", "white", "green", "full", "store", "pick", "spirits",
     "brands", "brand", "whiskeys", "bourbons", "high", "rye"}
)  # fmt: skip

#: Second-level public suffixes kept whole by :func:`registrable_domain`.
SECOND_LEVEL_SUFFIXES = frozenset(
    {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "org.au", "co.nz", "co.jp", "com.br",
     "com.mx", "co.za", "com.sg", "com.cn"}
)  # fmt: skip

#: Query-string parameters stripped when de-duplicating URLs.
TRACKING_PARAMS = frozenset(
    {"gclid", "fbclid", "msclkid", "yclid", "mc_cid", "mc_eid", "srsltid", "_ga", "ref",
     "ref_src", "igshid", "spm"}
)  # fmt: skip

#: Keywords around which excerpt windows are cut (besides name tokens).
EXCERPT_KEYWORDS: tuple[str, ...] = (
    "msrp", "suggested", "retail", "price", "batch", "release", "proof", "year", "aged",
)  # fmt: skip
EXCERPT_WINDOW = 800
MAX_QUERY_CHARS = 200
MAX_VENDOR_IMAGE_TRIES = 2

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_BATCH_RE = re.compile(r"\bbatch\s*(?:no\.?|number|#)?\s*([a-z0-9][a-z0-9-]*)", re.I)
_PRIORITY_KEYWORDS = ("msrp", "suggested", "retail")


@dataclass(frozen=True)
class WebOptions:
    """Limits of one research run (field names match WP3's ``WebSettings``)."""

    max_queries: int = 2
    max_results: int = 8
    max_pages: int = 5
    max_llm_pages: int = 4
    deny_domains: tuple[str, ...] = ()
    allow_domains: tuple[str, ...] = ()
    vendor_images: bool = False
    excerpt_chars: int = 10_000


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def fold(text: Any) -> str:  # noqa: ANN401
    """ASCII-folded, case-folded text (accents removed)."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    return s.encode("ascii", "ignore").decode("ascii").casefold()


def _fold_char(c: str) -> str:
    """One-character fold, so folded text keeps the original offsets."""
    if c.isascii():
        return c.lower()
    base = unicodedata.normalize("NFKD", c)[:1]
    if base.isascii() and base:
        return base.lower()
    low = c.lower()
    return low[0] if low else c


def _fold_aligned(text: str) -> str:
    return "".join(_fold_char(c) for c in text)


def _word_tokens(text: Any) -> list[str]:  # noqa: ANN401
    s = re.sub(r"['’]", "", fold(text))
    return re.findall(r"[a-z0-9]+", s)


def name_tokens(name: Any) -> list[str]:  # noqa: ANN401
    """Distinctive tokens of a bottle name (stopwords and 1-letter words dropped)."""
    out: list[str] = []
    for t in _word_tokens(name):
        if t in NAME_STOPWORDS or len(t) < 2 or t in out:
            continue
        out.append(t)
    return out


def brand_tokens(identity: Mapping[str, Any]) -> list[str]:
    """Alphanumeric tokens (at least 4 chars, with a letter) of distillery and name."""
    out: list[str] = []
    for value in (identity.get("distillery"), identity.get("name")):
        for t in _word_tokens(value):
            if len(t) >= 4 and re.search(r"[a-z]", t) and t not in BRAND_STOPWORDS and t not in out:
                out.append(t)
    return out


def release_tokens(identity: Mapping[str, Any]) -> list[str]:
    """4-digit years and the batch token from ``specific_release``/``batch_number``."""
    out: list[str] = []
    release = str(identity.get("specific_release") or "")
    batch = str(identity.get("batch_number") or "")
    for text in (release, batch):
        for y in _YEAR_RE.findall(text):
            if y not in out:
                out.append(y)
    token = ""
    if batch.strip():
        m = _BATCH_RE.search(batch)
        token = m.group(1) if m else batch.strip()
    else:
        m = _BATCH_RE.search(release)
        token = m.group(1) if m else ""
    token = re.sub(r"\s+", " ", fold(token).replace('"', "")).strip()
    if token and token not in out and not _YEAR_RE.fullmatch(token):
        out.append(token)
    return out


def _clean_field(value: Any) -> str:  # noqa: ANN401
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace('"', " ")
    return re.sub(r"\s+", " ", s).strip()


def build_queries(identity: Mapping[str, Any], max_queries: int = 2) -> list[str]:
    """Deterministic search queries from label fields only.

    1. ``"<name>" <distillery> <release tokens> MSRP``
    2. ``"<name>" <distillery> <specific_release>``

    ASCII-folded, capped at 200 characters, de-duplicated.
    """
    name = _clean_field(identity.get("name"))
    if not name:
        return []
    distillery = _clean_field(identity.get("distillery"))
    if distillery and fold(distillery) in fold(name):
        distillery = ""
    release = _clean_field(identity.get("specific_release"))
    tokens = [_clean_field(t) for t in release_tokens(identity)]
    candidates = [
        " ".join(p for p in (f'"{name}"', distillery, *tokens, "MSRP") if p),
        " ".join(p for p in (f'"{name}"', distillery, release) if p),
    ]
    out: list[str] = []
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip()[:MAX_QUERY_CHARS].strip()
        if q and q not in out:
            out.append(q)
    return out[: max(0, int(max_queries))]


# ---------------------------------------------------------------------------
# URL / domain helpers
# ---------------------------------------------------------------------------


def registrable_domain(host: str) -> str:
    """Last two labels of ``host`` (three for ``co.uk``-style suffixes)."""
    h = (host or "").strip().lower().rstrip(".")
    labels = [x for x in h.split(".") if x]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in SECOND_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def canonical_url(url: str) -> str:
    """URL without fragment and tracking parameters (for de-duplication)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(query), ""))


def _domain_matches(domain: str, host: str, patterns: tuple[str, ...]) -> bool:
    for p in patterns:
        p = p.strip().lower().lstrip(".")
        if p and (host == p or host.endswith("." + p) or domain == p):
            return True
    return False


def is_official_domain(domain: str, identity: Mapping[str, Any]) -> bool:
    """Whether the domain's first label contains a distillery/brand token."""
    first = (domain or "").split(".", 1)[0].replace("-", "")
    return any(t in first for t in brand_tokens(identity))


def classify(domain: str, identity: Mapping[str, Any], extracted: Any) -> str:  # noqa: ANN401
    """``official``, ``retailer`` (JSON-LD Product with offers) or ``reference``."""
    if is_official_domain(domain, identity):
        return "official"
    jsonld = getattr(extracted, "jsonld", None) or []
    if any(p.get("offers") for p in jsonld_products(jsonld)):
        return "retailer"
    return "reference"


# ---------------------------------------------------------------------------
# Prefilter and excerpt
# ---------------------------------------------------------------------------


def missing_name_tokens(tokens: list[str], title: Optional[str], text: str) -> list[str]:
    """Name tokens that do not appear (as words) in title plus text."""
    present = set(_word_tokens(f"{title or ''}\n{text}"))
    return [t for t in tokens if t not in present]


def _token_pattern(token: str) -> re.Pattern[str]:
    body = "['’]?".join(re.escape(c) for c in token)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _keyword_patterns(tokens: list[str]) -> list[tuple[re.Pattern[str], int]]:
    pats: list[tuple[re.Pattern[str], int]] = [(_token_pattern(t), 2) for t in tokens]
    for k in EXCERPT_KEYWORDS:
        pats.append(
            (re.compile(rf"(?<![a-z]){re.escape(k)}"), 1 if k not in _PRIORITY_KEYWORDS else 3)
        )
    pats.append((re.compile(r"\$"), 1))
    pats.append((_YEAR_RE, 1))
    return pats


def build_excerpt(text: str, tokens: list[str], cap: int) -> tuple[str, bool]:
    """Exact excerpt of ``text`` around name tokens and keywords.

    Windows of +-800 characters around each match are merged and kept in
    page order, joined by a blank line. When they exceed ``cap``, windows
    with the name or MSRP/retail cues are preferred; the output is still in
    page order and is cut at ``cap``. Every segment starts and ends on a word
    boundary (see :func:`_snap_inward`), so a number is never cut in half. No
    ellipses are ever inserted. Page text from :mod:`lib.html_text` has no
    empty lines, so ``"\n\n"`` in an excerpt always marks a segment boundary:
    a quote must not span it (the passages are not adjacent on the page).

    Returns:
        ``(excerpt, cut)``: ``cut`` is True when text was dropped to fit.
    """
    if not text or cap <= 0:
        return "", bool(text)
    folded = _fold_aligned(text)
    hits: list[tuple[int, int, int, int]] = []  # (start, end, weight, match position)
    for pat, weight in _keyword_patterns(tokens):
        for m in pat.finditer(folded):
            start = max(0, m.start() - EXCERPT_WINDOW)
            end = min(len(text), m.end() + EXCERPT_WINDOW)
            hits.append((start, end, weight, m.start()))
    if not hits:
        return "", False
    hits.sort()
    segments: list[_Segment] = []
    for start, end, weight, pos in hits:
        if segments and start <= segments[-1].end:
            segments[-1].add(end, weight, pos)
        else:
            seg = _Segment(start, end)
            seg.add(end, weight, pos)
            segments.append(seg)
    for seg in segments:
        seg.start, seg.end = _snap_inward(text, seg.start, seg.end)
    segments = [s for s in segments if s.end > s.start]
    if not segments:
        return "", False
    sep = "\n\n"
    total = sum(s.end - s.start for s in segments) + len(sep) * (len(segments) - 1)
    if total <= cap:
        return sep.join(text[s.start : s.end] for s in segments), False
    # Too long: keep the most relevant segments, centring a cut segment on
    # its strongest match. Output stays in page order.
    order = sorted(segments, key=lambda s: (-(s.name_hits > 0), -s.best_weight, -s.score, s.start))
    chosen: list[tuple[int, int]] = []
    budget = cap
    for seg in order:
        joiner = len(sep) if chosen else 0
        room = budget - joiner
        if room <= 0:
            break
        if seg.end - seg.start <= room:
            chosen.append((seg.start, seg.end))
            budget -= (seg.end - seg.start) + joiner
            continue
        if chosen and room < 400:
            continue
        start = max(seg.start, seg.focus - room // 2)
        end = min(seg.end, start + room)
        start = max(seg.start, end - room)
        start, end = _snap_inward(text, start, end)
        if end <= start:
            continue
        chosen.append((start, end))
        budget -= (end - start) + joiner
    chosen.sort()
    excerpt = sep.join(text[a:b] for a, b in chosen)
    if len(excerpt) > cap:  # pragma: no cover - the budget above keeps it within cap
        a, b = _snap_inward(excerpt, 0, cap)
        excerpt = excerpt[a:b]
    return excerpt, True


#: How far a window edge may move to reach a word boundary.
_SNAP_LIMIT = 80


def _snap_inward(text: str, start: int, end: int) -> tuple[int, int]:
    """Move ``start``/``end`` inward to word boundaries.

    A window edge that falls inside a word (or a number such as ``$1,299.99``)
    would hand the fact check a fragment like ``299.99``, which reads as a
    different value. Edges therefore move inward past the partial word, up to
    :data:`_SNAP_LIMIT` characters (beyond that the text has no whitespace
    worth respecting and the edge stays).
    """
    n = len(text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    if 0 < start < end and not text[start - 1].isspace() and not text[start].isspace():
        limit = min(end, start + _SNAP_LIMIT)
        i = start
        while i < limit and not text[i].isspace():
            i += 1
        if i < limit:
            start = i
    while start < end and text[start].isspace():
        start += 1
    if start < end < n and not text[end].isspace() and not text[end - 1].isspace():
        limit = max(start, end - _SNAP_LIMIT)
        j = end
        while j > limit and not text[j - 1].isspace():
            j -= 1
        if j > limit:
            end = j
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class _Segment:
    """A run of merged excerpt windows."""

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.name_hits = 0
        self.score = 0
        self.best_weight = 0
        self.focus = start

    def add(self, end: int, weight: int, pos: int) -> None:
        self.end = max(self.end, end)
        self.score += weight
        if weight == 2:
            self.name_hits += 1
        if weight > self.best_weight:
            self.best_weight = weight
            self.focus = pos


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class _Candidate:
    url: str
    hit: SearchHit
    query: str
    order: int
    domain: str
    official: bool
    record_title: Optional[str] = None


@dataclass
class _Fetched:
    cand: _Candidate
    page: Any
    source_class: str
    domain: str
    slot: int = -1


def _placeholder(name: Any) -> bool:  # noqa: ANN401
    try:
        from image_processor.preview_generator import _is_placeholder_name  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - fall back to a local check
        n = str(name or "").strip().lower()
        return not n or n.startswith("unknown")
    return _is_placeholder_name(None if name is None else str(name))


class WebResearcher:
    """Builds :class:`~models.evidence.WebResearch` for bottle identities.

    Args:
        provider: Search provider, or None (every bottle is ``skipped``).
        fetcher: :class:`lib.web_fetch.HttpFetcher` (or a fake with
            ``fetch_page``/``fetch_image``).
        options: Limits and domain lists.
        provider_reason: Why there is no provider (used when it is None).
        vendor_verifier: ``(primary_photo, candidate) -> verdict`` (see
            :mod:`image_processor.vendor_image`); needed for vendor images.
        image_cache_dir: Where accepted/candidate vendor images are stored
            (default: the fetcher cache's ``images/`` directory).
    """

    def __init__(
        self,
        *,
        provider: Optional[SearchProvider],
        fetcher: Any,  # noqa: ANN401
        options: WebOptions,
        provider_reason: str = "",
        vendor_verifier: Optional[Callable[[Path, Path], dict]] = None,
        image_cache_dir: Optional[Path] = None,
    ) -> None:
        self.provider = provider
        self.fetcher = fetcher
        self.options = options
        self.provider_reason = provider_reason
        self.vendor_verifier = vendor_verifier
        self.image_cache_dir = image_cache_dir
        self._disabled_reason: Optional[str] = None
        self._lock = threading.Lock()

    # -- public ----------------------------------------------------------------

    def research(
        self, identity: Mapping[str, Any], *, primary_photo: Optional[Path] = None
    ) -> WebResearch:
        """Research one bottle. Never raises."""
        try:
            return self._research(identity, primary_photo)
        except Exception as e:  # noqa: BLE001 - FR-035: never fail the bottle
            logger.exception("web research failed")
            return self._result("failed", reason=f"web research error: {self._safe(e)}")

    def research_many(
        self, items: list[tuple[Mapping[str, Any], Optional[Path]]], *, workers: int = 1
    ) -> list[WebResearch]:
        """Research several bottles; results are in input order. Never raises."""
        items = list(items)
        if not items:
            return []

        def one(item: tuple[Mapping[str, Any], Optional[Path]]) -> WebResearch:
            try:
                identity, photo = item
            except Exception as e:  # noqa: BLE001
                return self._result("failed", reason=f"bad research input: {type(e).__name__}")
            return self.research(identity, primary_photo=photo)

        n = max(1, min(int(workers or 1), len(items)))
        if n == 1:
            return [one(it) for it in items]
        try:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="web") as pool:
                return list(pool.map(one, items))
        except Exception as e:  # noqa: BLE001
            logger.exception("web research pool failed")
            reason = f"web research error: {self._safe(e)}"
            return [self._result("failed", reason=reason) for _ in items]

    # -- internals ---------------------------------------------------------------

    def _provider_name(self) -> Optional[str]:
        name = getattr(self.provider, "name", None)
        return name if name in SEARCH_PROVIDERS else None

    @staticmethod
    def _safe(e: BaseException) -> str:
        return f"{type(e).__name__}: {e}"[:300]

    def _result(self, status: str, **kw: Any) -> WebResearch:  # noqa: ANN401
        kw.setdefault("provider", self._provider_name())
        return WebResearch(status=status, **kw)  # type: ignore[arg-type]

    def _disable(self, reason: str, *, error: bool) -> None:
        with self._lock:
            if self._disabled_reason is not None:
                return
            self._disabled_reason = reason
        if error:
            logger.error("web search disabled for the rest of this run: %s", reason)
        else:
            logger.info("web search disabled for the rest of this run: %s", reason)

    def _research(self, identity: Mapping[str, Any], primary_photo: Optional[Path]) -> WebResearch:
        opts = self.options
        if self.provider is None:
            return WebResearch(
                status="skipped", reason=self.provider_reason or "no web search provider configured"
            )
        with self._lock:
            disabled = self._disabled_reason
        if disabled is not None:
            return self._result("skipped", reason=f"search provider disabled: {disabled}")
        name = identity.get("name")
        conf = str(identity.get("vision_confidence") or "").strip().lower()
        tokens = name_tokens(name)
        if _placeholder(name) or conf == "low" or not tokens:
            return self._result("skipped", reason="identity too uncertain to search")

        queries = build_queries(identity, opts.max_queries)
        if not queries:
            return self._result("skipped", reason="identity too uncertain to search")

        # 2. search
        searched_at = _now_iso()
        hits: list[tuple[SearchHit, str]] = []
        errors: list[str] = []
        sent: list[str] = []
        for q in queries:
            with self._lock:
                if self._disabled_reason is not None:
                    break
            sent.append(q)
            try:
                for h in self.provider.search(q, max_results=opts.max_results):
                    hits.append((h, q))
            except SearchProviderAuthError as e:
                self._disable(str(e), error=True)
                errors.append(str(e))
                break
            except SearchProviderDisabledError as e:
                self._disable(str(e), error=False)
                errors.append(str(e))
                break
            except SearchProviderError as e:
                logger.warning("web search failed for one query: %s", e)
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001 - provider bug: treat like an error
                logger.warning("web search raised %s", type(e).__name__)
                errors.append(self._safe(e))
        base = {"queries": sent, "searched_at": searched_at}
        if not hits and errors:
            return self._result("failed", reason=f"search failed: {errors[-1]}"[:300], **base)
        if not hits:
            return self._result("no_results", reason="search returned no results", **base)

        # One slot per candidate (rank order), then the pre-filter skips.
        pre_skipped: list[PageRecord] = []
        candidates = self._candidates(hits, identity, pre_skipped)
        slots: list[Optional[PageRecord]] = [None] * len(candidates)

        # 3. fetch, classify, prefilter
        fetched_ok: list[_Fetched] = []
        n_fetched = 0
        n_mismatch = 0
        for i, cand in enumerate(candidates):
            if i >= max(0, opts.max_pages):
                reason = f"page limit reached (max_pages={opts.max_pages})"
                slots[i] = self._record(cand, None, "skipped", reason)
                continue
            try:
                page = self.fetcher.fetch_page(cand.url)
            except Exception as e:  # noqa: BLE001 - FetchRefused or anything else
                slots[i] = self._record(cand, None, "skipped", _refusal_reason(e))
                continue
            n_fetched += 1
            final_url = getattr(page, "final_url", None) or cand.url
            domain = registrable_domain(_host(final_url)) or cand.domain
            host = _host(final_url)
            extracted = page.html_text
            if _domain_matches(domain, host, DENY_DOMAINS + tuple(opts.deny_domains)):
                slots[i] = self._page_record(page, cand, domain, None, "skipped",
                                             "deny-listed (after redirect)")  # fmt: skip
                continue
            if (
                opts.allow_domains
                and not is_official_domain(domain, identity)
                and not _domain_matches(domain, host, tuple(opts.allow_domains))
            ):
                slots[i] = self._page_record(page, cand, domain, None, "skipped",
                                             "not in allow_domains (after redirect)")  # fmt: skip
                continue
            source_class = classify(domain, identity, extracted)
            missing = missing_name_tokens(tokens, extracted.title, extracted.text)
            if missing:
                n_mismatch += 1
                slots[i] = self._page_record(
                    page, cand, domain, source_class, "skipped",
                    f"name mismatch: page does not mention {', '.join(missing)}",
                )  # fmt: skip
                continue
            fetched_ok.append(_Fetched(cand, page, source_class, domain, slot=i))

        # 5. excerpts
        evidence: list[WebEvidence] = []
        cap = max(0, min(int(opts.excerpt_chars), MAX_EVIDENCE_CHARS))
        passed: list[_Fetched] = []
        for f in fetched_ok:
            page = f.page
            if len(evidence) >= max(0, opts.max_llm_pages):
                slots[f.slot] = self._page_record(
                    page, f.cand, f.domain, f.source_class, "fetched_not_cited",
                    f"evidence limit reached (max_llm_pages={opts.max_llm_pages})",
                )  # fmt: skip
                passed.append(f)
                continue
            excerpt, cut = build_excerpt(page.html_text.text, tokens, cap)
            if not excerpt.strip():
                slots[f.slot] = self._page_record(
                    page, f.cand, f.domain, f.source_class, "fetched_not_cited",
                    "no relevant passage in page text",
                )  # fmt: skip
                passed.append(f)
                continue
            source_id = f"W{len(evidence) + 1}"
            prices = self._prices(page, tokens) if f.source_class == "retailer" else ()
            evidence.append(
                WebEvidence.create(
                    text=excerpt,
                    source_id=source_id,
                    url=getattr(page, "final_url", None) or f.cand.url,
                    domain=f.domain,
                    tier=f.source_class,
                    title=page.html_text.title or f.cand.hit.title or None,
                    fetched_at=str(page.fetched_at),
                    http_status=int(page.status),
                    query=f.cand.query,
                    from_cache=bool(page.from_cache),
                    truncated=bool(cut or getattr(page, "truncated", False)),
                    structured_prices=tuple(prices),
                )
            )
            slots[f.slot] = self._page_record(
                page, f.cand, f.domain, f.source_class, "fetched_not_cited",
                f"sent to fact check as {source_id}",
            )  # fmt: skip
            passed.append(f)

        # 6. vendor image
        vendor: Optional[VendorImage] = None
        if (opts.vendor_images and self.vendor_verifier is not None and primary_photo is not None
                and evidence):  # fmt: skip
            vendor = self._vendor_image(passed, tokens, Path(primary_photo))

        records = [r for r in slots if r is not None] + pre_skipped
        for r in records:
            logger.info("web: %s -> %s%s", r.url, r.outcome, f" ({r.reason})" if r.reason else "")

        if evidence:
            return self._result("ok", pages=records, evidence=evidence, vendor_image=vendor, **base)
        if n_fetched and n_mismatch == n_fetched:
            reason = f"{n_fetched} pages fetched; none matched the bottle name"
        elif n_fetched:
            reason = f"{n_fetched} pages fetched; none had a usable passage"
        else:
            reason = f"{len(records)} results considered; no page could be fetched"
        return self._result("no_results", reason=reason, pages=records, **base)

    def _candidates(
        self,
        hits: list[tuple[SearchHit, str]],
        identity: Mapping[str, Any],
        records: list[PageRecord],
    ) -> list[_Candidate]:
        opts = self.options
        deny = DENY_DOMAINS + tuple(opts.deny_domains)
        seen: set[str] = set()
        cands: list[_Candidate] = []
        for order, (hit, query) in enumerate(hits):
            url = str(hit.url).strip()
            key = canonical_url(url)
            if key in seen:
                continue
            seen.add(key)
            host = _host(url)
            domain = registrable_domain(host)
            cand = _Candidate(url=url, hit=hit, query=query, order=order, domain=domain,
                              official=False, record_title=hit.title or None)  # fmt: skip
            if not url.lower().startswith(("http://", "https://")) or not host:
                records.append(self._record(cand, None, "skipped", "not an http(s) URL"))
                continue
            if _domain_matches(domain, host, deny):
                records.append(self._record(cand, None, "skipped", "deny-listed"))
                continue
            cand.official = is_official_domain(domain, identity)
            if (
                opts.allow_domains
                and not cand.official
                and not _domain_matches(domain, host, tuple(opts.allow_domains))
            ):
                records.append(self._record(cand, None, "skipped", "not in allow_domains"))
                continue
            cands.append(cand)
        cands.sort(key=lambda c: (not c.official, c.order))
        return cands

    @staticmethod
    def _record(
        cand: _Candidate, source_class: Optional[str], outcome: str, reason: str
    ) -> PageRecord:
        return PageRecord(
            url=cand.url,
            title=cand.record_title,
            domain=cand.domain,
            source_class=source_class,  # type: ignore[arg-type]
            retrieved_at=None,
            outcome=outcome,  # type: ignore[arg-type]
            reason=reason,
        )

    @staticmethod
    def _page_record(
        page: Any,  # noqa: ANN401
        cand: _Candidate,
        domain: str,
        source_class: Optional[str],
        outcome: str,
        reason: str,
    ) -> PageRecord:
        return PageRecord(
            url=getattr(page, "final_url", None) or cand.url,
            title=page.html_text.title or cand.record_title,
            domain=domain,
            source_class=source_class,  # type: ignore[arg-type]
            retrieved_at=str(page.fetched_at),
            outcome=outcome,  # type: ignore[arg-type]
            reason=reason,
            from_cache=bool(getattr(page, "from_cache", False)),
        )

    @staticmethod
    def _prices(page: Any, tokens: list[str]) -> list[StructuredPrice]:  # noqa: ANN401
        """JSON-LD offer prices of Products whose name matches the bottle."""
        out: list[StructuredPrice] = []
        for product in jsonld_products(page.html_text.jsonld):
            pname = product.get("name")
            if not isinstance(pname, str) or missing_name_tokens(tokens, None, pname):
                continue
            for p in offer_prices(product):
                if p not in out:
                    out.append(p)
        return out

    # -- vendor image ------------------------------------------------------------

    def _image_dir(self) -> Optional[Path]:
        if self.image_cache_dir is not None:
            return Path(self.image_cache_dir)
        cache = getattr(self.fetcher, "cache", None)
        root = getattr(cache, "root", None)
        return Path(root) / "images" if root is not None else None

    def _vendor_image(
        self, passed: list[_Fetched], tokens: list[str], primary_photo: Path
    ) -> Optional[VendorImage]:
        candidates: list[tuple[str, str]] = []
        for f in passed:
            if f.source_class not in ("official", "retailer"):
                continue
            page = f.page
            base = getattr(page, "final_url", None) or f.cand.url
            urls: list[str] = []
            if page.html_text.og_image:
                urls.append(page.html_text.og_image)
            for product in jsonld_products(page.html_text.jsonld):
                pname = product.get("name")
                if isinstance(pname, str) and missing_name_tokens(tokens, None, pname):
                    continue
                urls.extend(product_images(product))
            for u in urls:
                absolute = urljoin(base, u)
                if absolute.lower().startswith(("http://", "https://")) and absolute not in (
                    c[0] for c in candidates
                ):
                    candidates.append((absolute, base))
        directory = self._image_dir()
        if not candidates or directory is None:
            return None
        for image_url, page_url in candidates[:MAX_VENDOR_IMAGE_TRIES]:
            try:
                data, ext = self.fetcher.fetch_image(image_url)
            except Exception as e:  # noqa: BLE001
                logger.info("vendor image %s not used: %s", image_url, _refusal_reason(e))
                continue
            try:
                path = _store_image(directory, image_url, data, ext)
            except OSError as e:
                logger.info("vendor image could not be stored: %s", e)
                return None
            retrieved_at = _now_iso()
            try:
                verdict = self.vendor_verifier(primary_photo, path)  # type: ignore[misc]
            except Exception as e:  # noqa: BLE001 - vendor image is optional
                logger.warning("vendor image skipped: verifier failed (%s)", type(e).__name__)
                return None
            if not isinstance(verdict, dict):
                continue
            if (verdict.get("same_product") is True and verdict.get("confidence") == "high"
                    and verdict.get("same_release") != "no"):  # fmt: skip
                return VendorImage(
                    cache_path=str(path),
                    image_url=image_url,
                    page_url=page_url,
                    retrieved_at=retrieved_at,
                    verdict=dict(verdict),
                )
            logger.info("vendor image %s rejected by the verifier", image_url)
        return None


def _store_image(directory: Path, image_url: str, data: bytes, ext: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ext = re.sub(r"[^a-z0-9]", "", str(ext).lower())[:5] or "img"
    path = directory / f"{hashlib.sha256(image_url.encode('utf-8')).hexdigest()}.{ext}"
    if path.is_file():
        try:
            if path.read_bytes() == data:
                return path
        except OSError:
            pass
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _refusal_reason(e: BaseException) -> str:
    """PageRecord reason for a fetch refusal or error."""
    reason = getattr(e, "reason", None)
    if not isinstance(reason, str) or not reason:
        return f"fetch error: {type(e).__name__}"
    label = {
        "robots": "robots.txt",
        "unsafe host": "unsafe host",
        "unsafe scheme": "unsafe host",
        "too big": "too big",
        "content-type": "content-type",
        "budget": "request budget used",
    }.get(reason, "fetch error")
    detail = str(getattr(e, "detail", "") or "")
    return f"{label}: {detail}" if detail and reason != "budget" else label
