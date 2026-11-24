"""Web search providers for label research.

Providers (all over an injected ``session`` with ``.request(...)``):

- :class:`DuckDuckGoSearchProvider`: keyless default. Posts to the HTML
  endpoint (falling back to the lite endpoint) and parses the result list
  with ``html.parser``. At least 2 s between queries; throttling (HTTP
  202/403/429 or an anomaly/captcha page) is retried once, then the query
  fails; after 3 throttled queries in a row the provider disables itself.
- :class:`BraveSearchProvider`: Brave Search API (``BRAVE_API_KEY``).
- :class:`OllamaSearchProvider`: Ollama ``web_search`` (``OLLAMA_API_KEY``).
- :class:`SearxngSearchProvider`: a self-hosted SearXNG instance (JSON
  output must be enabled in its ``settings.yml``).

Common behaviour: a rate limiter per provider (injected clock and sleep),
one retry on 429 honouring ``Retry-After`` up to 30 s, 401/403 raise
:class:`SearchProviderAuthError`, and results are cached for 24 h in the
web cache's ``search/`` namespace when a cache is given.

API keys: held in a private attribute, shown as ``key=***`` by ``repr``,
sent only in a header (never in a URL), never logged. Exception text is
passed through :func:`redact` before it is logged or raised.

Queries contain only label fields; nothing else about the user is sent.
"""

from __future__ import annotations

import html
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Optional, Protocol
from urllib.parse import parse_qs, urljoin, urlsplit

from lib.web_cache import WebCache
from lib.web_fetch import USER_AGENT, parse_retry_after

logger = logging.getLogger(__name__)

#: Accepted ``web.provider`` values.
PROVIDER_NAMES: tuple[str, ...] = ("auto", "duckduckgo", "brave", "ollama", "searxng", "none")

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
OLLAMA_ENDPOINT = "https://ollama.com/api/web_search"
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

SEARCH_CACHE_TTL_SECONDS = 24 * 3600
MAX_RETRY_AFTER_SECONDS = 30.0
DDG_MIN_INTERVAL_SECONDS = 2.0
DDG_THROTTLE_BACKOFF_SECONDS = 10.0
DDG_MAX_CONSECUTIVE_THROTTLES = 3
_SNIPPET_CHARS = 500
_TIMEOUT = (5, 15)


@dataclass(frozen=True)
class SearchHit:
    """One search result.

    Attributes:
        url: Result URL (http/https).
        title: Result title (plain text).
        snippet: Result snippet (plain text).
        rank: 1-based position in this provider's result list.
        provider: Provider name.
    """

    url: str
    title: str
    snippet: str
    rank: int
    provider: str


class SearchProviderError(Exception):
    """A search failed (transient); the bottle continues without it."""


class SearchProviderAuthError(SearchProviderError):
    """The provider rejected the credentials (401/403); disable it for the run."""


class SearchProviderDisabledError(SearchProviderError):
    """The provider disabled itself for the rest of the run (e.g. throttled)."""


class SearchProvider(Protocol):
    """A web search backend."""

    name: str

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        """Up to ``max_results`` hits for ``query``."""
        ...


def redact(text: Any, secrets: Iterable[Optional[str]]) -> str:  # noqa: ANN401
    """``text`` with every non-empty secret replaced by ``***``."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "***")
    return out


class _RateLimiter:
    """At least ``interval`` seconds between calls (thread-safe)."""

    def __init__(self, interval: float, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._last: Optional[float] = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._last is not None:
                delay = self._last + self.interval - self._clock()
                if delay > 0:
                    self._sleep(delay)
            self._last = self._clock()


def _strip_tags(text: Any) -> str:  # noqa: ANN401
    s = re.sub(r"<[^>]+>", "", str(text or ""))
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _is_http_url(url: Any) -> bool:  # noqa: ANN401
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(parts.hostname)


class _BaseProvider:
    name = "base"
    min_interval = 1.0

    def __init__(
        self,
        session: Any,  # noqa: ANN401
        *,
        cache: Optional[WebCache] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        cache_ttl_seconds: float = SEARCH_CACHE_TTL_SECONDS,
    ) -> None:
        self._session = session
        self._cache = cache
        self._sleep = sleep
        self._limiter = _RateLimiter(self.min_interval, clock, sleep)
        self._cache_ttl = cache_ttl_seconds
        self.requests_made = 0

    def _secrets(self) -> tuple[str, ...]:
        return ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        """Up to ``max_results`` hits for ``query`` (cached for 24 h).

        Raises:
            SearchProviderError: On failure (auth errors as
                :class:`SearchProviderAuthError`).
        """
        max_results = max(1, int(max_results))
        key = f"{self.name}\n{max_results}\n{query}"
        if self._cache is not None:
            cached = self._cache.get("search", key, self._cache_ttl)
            hits = _hits_from_cache(cached)
            if hits is not None:
                return hits
        hits = self._search(query, max_results)[:max_results]
        if self._cache is not None:
            self._cache.put("search", key, [asdict(h) for h in hits])
        return hits

    def _search(self, query: str, max_results: int) -> list[SearchHit]:
        raise NotImplementedError

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Rate-limited request with one 429 retry; maps errors."""
        for attempt in (1, 2):
            self._limiter.wait()
            self.requests_made += 1
            try:
                resp = self._session.request(method, url, timeout=_TIMEOUT, **kwargs)
            except Exception as e:  # noqa: BLE001 - any client error
                msg = redact(f"{self.name} search request failed: {type(e).__name__}: {e}",
                             self._secrets())  # fmt: skip
                raise SearchProviderError(msg) from None
            status = int(getattr(resp, "status_code", 0) or 0)
            if status == 429 and attempt == 1:
                delay = parse_retry_after((getattr(resp, "headers", None) or {}).get("Retry-After"))
                delay = 1.0 if delay is None else delay
                if delay > MAX_RETRY_AFTER_SECONDS:
                    raise SearchProviderError(
                        f"{self.name} search rate limited (Retry-After {delay:.0f}s)"
                    )
                self._sleep(delay)
                continue
            if status == 429:
                raise SearchProviderError(f"{self.name} search rate limited (HTTP 429)")
            self._check_status(status)
            return resp
        raise SearchProviderError(f"{self.name} search failed")  # pragma: no cover

    def _check_status(self, status: int) -> None:
        if status in (401, 403):
            raise SearchProviderAuthError(
                f"{self.name} search rejected the credentials (HTTP {status})"
            )
        if not 200 <= status < 300:
            raise SearchProviderError(f"{self.name} search failed (HTTP {status})")

    def _json(self, resp: Any) -> Any:  # noqa: ANN401
        try:
            return resp.json()
        except Exception as e:  # noqa: BLE001
            raise SearchProviderError(f"{self.name} search returned invalid JSON") from e

    def _hits(self, rows: Iterable[tuple[Any, Any, Any]], max_results: int) -> list[SearchHit]:
        out: list[SearchHit] = []
        for url, title, snippet in rows:
            if not _is_http_url(url):
                continue
            out.append(
                SearchHit(
                    url=str(url).strip(),
                    title=_strip_tags(title),
                    snippet=_strip_tags(snippet)[:_SNIPPET_CHARS],
                    rank=len(out) + 1,
                    provider=self.name,
                )
            )
            if len(out) >= max_results:
                break
        return out


def _hits_from_cache(cached: Any) -> Optional[list[SearchHit]]:  # noqa: ANN401
    if not isinstance(cached, list):
        return None
    try:
        return [SearchHit(**{k: row[k] for k in ("url", "title", "snippet", "rank", "provider")})
                for row in cached]  # fmt: skip
    except (KeyError, TypeError):
        return None


class _KeyedProvider(_BaseProvider):
    def __init__(self, key: str, session: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if not key or not str(key).strip():
            raise ValueError(f"{type(self).__name__} needs an API key")
        super().__init__(session, **kwargs)
        self.__key = str(key).strip()

    def _key(self) -> str:
        return self.__key

    def _secrets(self) -> tuple[str, ...]:
        return (self.__key,)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key=***)"


class BraveSearchProvider(_KeyedProvider):
    """Brave Search API (``X-Subscription-Token`` header)."""

    name = "brave"

    def _search(self, query: str, max_results: int) -> list[SearchHit]:
        resp = self._request(
            "GET",
            BRAVE_ENDPOINT,
            params={"q": query, "count": min(max_results, 20)},
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "X-Subscription-Token": self._key(),
            },
        )
        data = self._json(resp)
        results = ((data or {}).get("web") or {}).get("results") or []
        return self._hits(
            ((r.get("url"), r.get("title"), r.get("description")) for r in results
             if isinstance(r, dict)),
            max_results,
        )  # fmt: skip


class OllamaSearchProvider(_KeyedProvider):
    """Ollama ``web_search`` API (``Authorization: Bearer`` header)."""

    name = "ollama"

    def _search(self, query: str, max_results: int) -> list[SearchHit]:
        resp = self._request(
            "POST",
            OLLAMA_ENDPOINT,
            json={"query": query, "max_results": min(max_results, 10)},
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {self._key()}",
            },
        )
        data = self._json(resp)
        results = (data or {}).get("results") or []
        return self._hits(
            ((r.get("url"), r.get("title"), r.get("content")) for r in results
             if isinstance(r, dict)),
            max_results,
        )  # fmt: skip


class SearxngSearchProvider(_BaseProvider):
    """A self-hosted SearXNG instance (``format=json`` must be enabled)."""

    name = "searxng"

    def __init__(self, base_url: str, session: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if not _is_http_url(base_url):
            raise ValueError(f"SearXNG URL must be http(s), got {base_url!r}")
        super().__init__(session, **kwargs)
        self.base_url = str(base_url).rstrip("/")

    def __repr__(self) -> str:
        return f"SearxngSearchProvider(base_url={self.base_url!r})"

    def _check_status(self, status: int) -> None:
        if status == 403:
            raise SearchProviderError("enable json in searxng settings.yml search.formats")
        super()._check_status(status)

    def _search(self, query: str, max_results: int) -> list[SearchHit]:
        resp = self._request(
            "GET",
            f"{self.base_url}/search",
            params={"q": query, "format": "json"},
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        data = self._json(resp)
        results = (data or {}).get("results") or []
        return self._hits(
            ((r.get("url"), r.get("title"), r.get("content")) for r in results
             if isinstance(r, dict)),
            max_results,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------


class _DdgParser(HTMLParser):
    """Result links and snippets of the DuckDuckGo HTML and lite pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._mode: Optional[str] = None  # "title" | "snippet"
        self._tag: Optional[str] = None
        self._depth = 0
        self._buf: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._mode is not None:
            if tag == self._tag:
                self._depth += 1
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        classes = set(a.get("class", "").split())
        if tag == "a" and classes & {"result__a", "result-link"}:
            self._start("title", tag, a.get("href", ""))
        elif classes & {"result__snippet", "result-snippet"}:
            self._start("snippet", tag, "")

    def _start(self, mode: str, tag: str, href: str) -> None:
        self._mode, self._tag, self._depth, self._buf, self._href = mode, tag, 0, [], href

    def handle_endtag(self, tag: str) -> None:
        if self._mode is None or tag != self._tag:
            return
        if self._depth:
            self._depth -= 1
            return
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if self._mode == "title":
            self.results.append({"href": self._href, "title": text, "snippet": ""})
        elif self.results and not self.results[-1]["snippet"]:
            self.results[-1]["snippet"] = text
        self._mode = self._tag = None

    def handle_data(self, data: str) -> None:
        if self._mode is not None:
            self._buf.append(data)


def _unwrap_ddg_href(href: str) -> Optional[str]:
    """Target URL of a DuckDuckGo result link; None for ads and junk."""
    href = (href or "").strip()
    if not href or "/y.js?" in href or "ad_provider" in href:
        return None
    absolute = urljoin("https://duckduckgo.com/", href)
    try:
        parts = urlsplit(absolute)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host.endswith("duckduckgo.com"):
        if parts.path.rstrip("/") != "/l":
            return None
        target = (parse_qs(parts.query).get("uddg") or [""])[0]
        if not target or "/y.js?" in target or "ad_provider" in target:
            return None
        absolute = target
    if not _is_http_url(absolute):
        return None
    if (urlsplit(absolute).hostname or "").lower().endswith("duckduckgo.com"):
        return None
    return absolute


def parse_ddg_results(body: str) -> list[tuple[str, str, str]]:
    """``(url, title, snippet)`` rows from a DuckDuckGo HTML or lite page."""
    parser = _DdgParser()
    try:
        parser.feed(body or "")
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    rows: list[tuple[str, str, str]] = []
    for r in parser.results:
        url = _unwrap_ddg_href(r["href"])
        if url:
            rows.append((url, r["title"], r["snippet"]))
    return rows


class DuckDuckGoSearchProvider(_BaseProvider):
    """Keyless DuckDuckGo search via the HTML endpoint (lite fallback).

    Not an official API: politeness is enforced (at least 2 s between
    queries), and throttling disables the provider after
    :data:`DDG_MAX_CONSECUTIVE_THROTTLES` throttled queries in a row.
    """

    name = "duckduckgo"
    min_interval = DDG_MIN_INTERVAL_SECONDS

    def __init__(
        self,
        session: Any,  # noqa: ANN401
        *,
        throttle_backoff: float = DDG_THROTTLE_BACKOFF_SECONDS,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(session, **kwargs)
        self.throttle_backoff = throttle_backoff
        self.consecutive_throttles = 0
        self.disabled = False
        self._state_lock = threading.Lock()

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:  # noqa: D102
        if self.disabled:
            raise SearchProviderDisabledError("duckduckgo disabled for this run: throttled")
        return super().search(query, max_results=max_results)

    def _post(self, url: str, query: str) -> tuple[int, str]:
        self._limiter.wait()
        self.requests_made += 1
        try:
            resp = self._session.request(
                "POST",
                url,
                data={"q": query},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html",
                    "Accept-Language": "en-US",
                },
                timeout=_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001
            raise SearchProviderError(
                f"duckduckgo search request failed: {type(e).__name__}"
            ) from None
        status = int(getattr(resp, "status_code", 0) or 0)
        try:
            body = str(getattr(resp, "text", "") or "")
        except Exception:  # noqa: BLE001
            body = ""
        return status, body

    def _attempt(self, query: str) -> Optional[list[tuple[str, str, str]]]:
        """Rows, or None when throttled."""
        for endpoint in (DDG_HTML_ENDPOINT, DDG_LITE_ENDPOINT):
            status, body = self._post(endpoint, query)
            if status in (202, 403, 429):
                return None
            if not 200 <= status < 300:
                raise SearchProviderError(f"duckduckgo search failed (HTTP {status})")
            rows = parse_ddg_results(body)
            if rows:
                return rows
            lowered = body.lower()
            if "anomaly" in lowered or "captcha" in lowered:
                return None
        return []

    def _search(self, query: str, max_results: int) -> list[SearchHit]:
        rows = self._attempt(query)
        if rows is None:
            self._sleep(self.throttle_backoff)
            rows = self._attempt(query)
        if rows is None:
            with self._state_lock:
                self.consecutive_throttles += 1
                disable = (
                    not self.disabled
                    and self.consecutive_throttles >= DDG_MAX_CONSECUTIVE_THROTTLES
                )
                if disable:
                    self.disabled = True
            if disable:
                logger.warning(
                    "duckduckgo throttled %d queries in a row; web search disabled for the rest "
                    "of this run",
                    self.consecutive_throttles,
                )
                raise SearchProviderDisabledError("duckduckgo throttled; disabled for this run")
            raise SearchProviderError("duckduckgo throttled")
        with self._state_lock:
            self.consecutive_throttles = 0
        return self._hits(rows, max_results)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _env(environ: Mapping[str, str], *names: str) -> str:
    for n in names:
        v = (environ.get(n) or "").strip()
        if v:
            return v
    return ""


def resolve_provider_name(
    name: str, environ: Mapping[str, str], *, searxng_url: str = ""
) -> tuple[Optional[str], str]:
    """Concrete provider for a ``web.provider`` value, plus a reason.

    ``auto`` picks Brave (``BRAVE_API_KEY``/``BOURBON_BRAVE_API_KEY``), then
    Ollama (``OLLAMA_API_KEY``), then SearXNG (``searxng_url``), then
    DuckDuckGo (no key needed).

    Returns:
        ``(name, reason)``; ``name`` is None when no provider can be used
        (``none``, or an explicit provider missing its key/URL), and
        ``reason`` says why.

    Raises:
        ValueError: If ``name`` is not one of :data:`PROVIDER_NAMES`.
    """
    n = (name or "auto").strip().lower()
    if n not in PROVIDER_NAMES:
        raise ValueError(f"unknown web provider {name!r}; expected one of {PROVIDER_NAMES}")
    brave = _env(environ, "BRAVE_API_KEY", "BOURBON_BRAVE_API_KEY")
    ollama = _env(environ, "OLLAMA_API_KEY")
    if n == "auto":
        if brave:
            return "brave", "auto: BRAVE_API_KEY is set"
        if ollama:
            return "ollama", "auto: OLLAMA_API_KEY is set"
        if (searxng_url or "").strip():
            return "searxng", "auto: web.searxng_url is set"
        return "duckduckgo", "auto: no API key or SearXNG URL; using keyless DuckDuckGo"
    if n == "none":
        return None, "web search provider is 'none'"
    if n == "brave" and not brave:
        return None, "web provider 'brave' needs BRAVE_API_KEY (or BOURBON_BRAVE_API_KEY)"
    if n == "ollama" and not ollama:
        return None, "web provider 'ollama' needs OLLAMA_API_KEY"
    if n == "searxng" and not (searxng_url or "").strip():
        return None, "web provider 'searxng' needs web.searxng_url"
    return n, f"web provider {n!r} selected"


def build_provider(
    name: str,
    environ: Mapping[str, str],
    session: Any,  # noqa: ANN401
    *,
    searxng_url: str = "",
    cache: Optional[WebCache] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cache_ttl_hours: float = 24,
) -> Optional[SearchProvider]:
    """Build the provider for ``name`` (see :func:`resolve_provider_name`).

    Returns:
        The provider, or None for ``none`` or a missing key/URL.

    Raises:
        ValueError: For an unknown name or an invalid SearXNG URL.
    """
    resolved, reason = resolve_provider_name(name, environ, searxng_url=searxng_url)
    if resolved is None:
        logger.debug("no web search provider: %s", reason)
        return None
    kw: dict[str, Any] = {
        "cache": cache,
        "clock": clock,
        "sleep": sleep,
        "cache_ttl_seconds": float(cache_ttl_hours) * 3600,
    }
    if resolved == "brave":
        return BraveSearchProvider(
            _env(environ, "BRAVE_API_KEY", "BOURBON_BRAVE_API_KEY"), session, **kw
        )
    if resolved == "ollama":
        return OllamaSearchProvider(_env(environ, "OLLAMA_API_KEY"), session, **kw)
    if resolved == "searxng":
        return SearxngSearchProvider(searxng_url.strip(), session, **kw)
    return DuckDuckGoSearchProvider(session, **kw)
