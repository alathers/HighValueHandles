"""Polite, safe HTTP fetching of public web pages and images.

Every request goes through an injected ``session`` (anything with
``.request(method, url, **kw)`` and ``.cookies``; a ``requests.Session`` in
production, a fake in tests). Rules:

- **Schemes:** ``http`` and ``https`` only.
- **SSRF guard on every hop:** redirects are followed manually
  (``allow_redirects=False``, at most :data:`MAX_REDIRECTS` hops) and each
  hop is checked. IP literals, ``localhost``, ``*.localhost``, ``*.local``,
  ``*.internal``, single-label hosts, and hosts with any resolved address
  that is not globally routable are refused. (The check resolves the name
  itself; the HTTP client resolves again, so a DNS-rebinding race is not
  fully excluded. The tool only fetches URLs returned by a search engine.)
- **robots.txt:** fetched through the same session, cached for 24 h and
  parsed with ``RobotFileParser.parse`` (never ``.read()``, which would use
  urllib and bypass the session). 404/410 means allowed; any other error,
  5xx or timeout means disallowed. ``Crawl-delay`` is honoured up to 30 s; a
  longer delay skips the host.
- **Rate limits:** one request in flight per host, at least
  ``max(per_host_interval, crawl_delay)`` seconds between requests to a
  host, and a per-run request budget (``max_requests``).
- **Bodies** are streamed with ``iter_content`` up to a byte cap. A page
  over the cap is cut (``truncated=True``); an image over the cap is refused.
- **Types:** pages must be ``text/html``, ``application/xhtml+xml`` or
  ``text/plain``; images ``image/jpeg``, ``image/png`` or ``image/webp``,
  verified with Pillow, at least 300 px on the short edge.
- **Retries:** one retry on 429 or 503, honouring ``Retry-After`` up to
  30 s. Nothing else is retried.
- **Privacy:** a generic :data:`USER_AGENT`, cookies cleared before every
  request, no ``Referer``, ``Accept-Language: en-US``.
- **Cache:** pages for ``cache_ttl_hours``; refusals (robots, type, unsafe
  host, HTTP 4xx) for 24 h.
"""

from __future__ import annotations

import dataclasses
import io
import ipaddress
import logging
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from lib.html_text import ExtractedHtml, drop_partial_tail, extract
from lib.web_cache import WebCache

logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return version("bourbon-image-processor")
        except PackageNotFoundError:
            return "dev"
    except Exception:  # noqa: BLE001
        return "dev"


#: Generic User-Agent for every web request. Contains no personal data.
USER_AGENT = f"bourbon-process/{_package_version()} (label-research; respects robots.txt)"

#: Headers sent with every request (no Referer, no cookies).
BASE_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US",
}

MAX_REDIRECTS = 5
MAX_RETRY_AFTER_SECONDS = 30.0
MAX_CRAWL_DELAY_SECONDS = 30.0
ROBOTS_TTL_SECONDS = 24 * 3600
NEGATIVE_TTL_SECONDS = 24 * 3600
MAX_ROBOTS_BYTES = 512_000
MIN_IMAGE_EDGE = 300

PAGE_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
IMAGE_CONTENT_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
_PIL_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_RETRY_CODES = frozenset({429, 503})
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)

#: Refusal reasons (``FetchRefused.reason``) cached as negative results.
_CACHEABLE_REFUSALS = frozenset({"robots", "unsafe host", "content-type", "http error"})


class FetchRefused(Exception):
    """A URL was not fetched, or its response was not usable.

    Attributes:
        reason: Short category: ``robots``, ``unsafe host``, ``unsafe scheme``,
            ``too big``, ``content-type``, ``http error``, ``fetch error``,
            ``budget``, ``too many redirects`` or ``bad image``.
        detail: Human-readable detail (no secrets).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class FetchedPage:
    """A fetched page, reduced to its text and metadata.

    Attributes:
        url: URL requested.
        final_url: URL after redirects.
        status: HTTP status of the final response.
        fetched_at: ISO-8601 UTC time of the fetch.
        content_type: Media type of the response (no parameters).
        html_text: Extracted text, title, JSON-LD and links.
        truncated: Whether the body hit the byte cap (or the text cap).
        from_cache: Whether it came from the web cache.
    """

    url: str
    final_url: str
    status: int
    fetched_at: str
    content_type: str
    html_text: ExtractedHtml
    truncated: bool
    from_cache: bool

    @property
    def title(self) -> Optional[str]:
        """Page title, if any."""
        return self.html_text.title

    @property
    def text(self) -> str:
        """Visible page text."""
        return self.html_text.text


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_retry_after(value: Any, *, now: Optional[datetime] = None) -> Optional[float]:  # noqa: ANN401
    """Seconds from a ``Retry-After`` header (delta-seconds or HTTP date)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (when - current).total_seconds())


def _media_type(content_type: Any) -> str:  # noqa: ANN401
    return str(content_type or "").split(";", 1)[0].strip().lower()


def _decode(body: bytes, content_type: str) -> str:
    charset = None
    m = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", content_type or "", re.I)
    if m:
        charset = m.group(1)
    else:
        mm = _META_CHARSET.search(body[:4096])
        if mm:
            charset = mm.group(1).decode("ascii", "ignore")
    for enc in (charset, "utf-8"):
        if not enc:
            continue
        try:
            return body.decode(enc, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")


def _host_is_unsafe_name(host: str) -> Optional[str]:
    h = host.strip().lower().rstrip(".")
    if not h:
        return "empty host"
    try:
        ipaddress.ip_address(h.strip("[]").split("%", 1)[0])
    except ValueError:
        pass
    else:
        return "IP literal"
    if h == "localhost" or h.endswith((".localhost", ".local", ".internal")):
        return "local name"
    if "." not in h:
        return "single-label host"
    return None


class HttpFetcher:
    """Fetches pages and images under the rules in the module docstring.

    Thread-safe: one fetcher is shared by all workers of a run, so the
    per-host locks and the request budget apply across bottles.

    Args:
        session: Object with ``.request(method, url, **kw)`` and ``.cookies``.
        cache: Web cache (or None for no caching).
        resolver: ``getaddrinfo``-compatible callable; defaults to
            ``socket.getaddrinfo`` looked up at call time.
        clock: Monotonic seconds, for rate limiting.
        sleep: Sleep function (fake in tests).
        timeout: ``(connect, read)`` timeout passed to the session.
        max_page_bytes: Byte cap for a page body.
        max_image_bytes: Byte cap for an image.
        per_host_interval: Minimum seconds between requests to one host.
        max_requests: Request budget for the run (robots.txt included).
        refresh: Skip cache reads (still writes).
        cache_ttl_hours: How long fetched pages stay fresh.
    """

    def __init__(
        self,
        *,
        session: Any,  # noqa: ANN401
        cache: Optional[WebCache],
        resolver: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: tuple[float, float] = (5, 15),
        max_page_bytes: int = 2_000_000,
        max_image_bytes: int = 5_000_000,
        per_host_interval: float = 2.0,
        max_requests: int = 500,
        refresh: bool = False,
        cache_ttl_hours: float = 168,
    ) -> None:
        self.session = session
        self.cache = cache
        self._resolver = resolver
        self._clock = clock
        self._sleep = sleep
        self.timeout = timeout
        self.max_page_bytes = int(max_page_bytes)
        self.max_image_bytes = int(max_image_bytes)
        self.per_host_interval = float(per_host_interval)
        self.max_requests = int(max_requests)
        self.refresh = refresh
        self.cache_ttl_seconds = float(cache_ttl_hours) * 3600
        self.requests_made = 0
        self._budget_lock = threading.Lock()
        self._hosts_lock = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._host_last: dict[str, float] = {}
        self._host_delay: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_locks: dict[str, threading.Lock] = {}
        # Requests share one session; serialise cookie clearing + request start.
        self._session_lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"HttpFetcher(requests_made={self.requests_made}, max_requests={self.max_requests}, "
            f"per_host_interval={self.per_host_interval})"
        )

    # -- safety ----------------------------------------------------------------

    def check_url(self, url: str) -> tuple[str, str]:
        """Validate scheme and host of ``url``; returns ``(scheme, netloc)``.

        Raises:
            FetchRefused: ``unsafe scheme`` or ``unsafe host``.
        """
        try:
            parts = urlsplit(url)
            host = parts.hostname or ""
            port = parts.port
        except ValueError as e:
            raise FetchRefused("unsafe host", f"unparseable URL ({e})") from e
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise FetchRefused("unsafe scheme", f"scheme {scheme or '(none)'!r} not allowed")
        problem = _host_is_unsafe_name(host)
        if problem:
            raise FetchRefused("unsafe host", f"{host or '(none)'}: {problem}")
        if parts.username or parts.password:
            raise FetchRefused("unsafe host", "credentials in URL")
        resolver = self._resolver or socket.getaddrinfo
        try:
            infos = resolver(
                host, port or (443 if scheme == "https" else 80), 0, socket.SOCK_STREAM
            )
        except (OSError, UnicodeError) as e:
            raise FetchRefused("fetch error", f"{host}: name resolution failed") from e
        addresses = {info[4][0] for info in infos if info and len(info) > 4 and info[4]}
        if not addresses:
            raise FetchRefused("fetch error", f"{host}: no addresses")
        for addr in addresses:
            try:
                ip = ipaddress.ip_address(str(addr).split("%", 1)[0])
            except ValueError as e:
                raise FetchRefused("unsafe host", f"{host}: bad address") from e
            if not ip.is_global:
                raise FetchRefused("unsafe host", f"{host} resolves to a non-public address")
        return scheme, parts.netloc.lower()

    # -- rate limiting -----------------------------------------------------------

    def _take_budget(self) -> None:
        with self._budget_lock:
            if self.requests_made >= self.max_requests:
                raise FetchRefused("budget", f"run request budget of {self.max_requests} used")
            self.requests_made += 1

    @contextmanager
    def _host_slot(self, netloc: str) -> Iterator[None]:
        with self._hosts_lock:
            lock = self._host_locks.setdefault(netloc, threading.Lock())
        with lock:
            interval = max(self.per_host_interval, self._host_delay.get(netloc, 0.0))
            last = self._host_last.get(netloc)
            if last is not None:
                wait = last + interval - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._host_last[netloc] = self._clock()
            yield

    # -- HTTP ------------------------------------------------------------------

    def _send(self, url: str, netloc: str, accept: str) -> Any:  # noqa: ANN401
        """One GET with the host slot, budget and the 429/503 retry."""
        for attempt in (1, 2):
            with self._host_slot(netloc):
                self._take_budget()
                headers = dict(BASE_HEADERS)
                headers["Accept"] = accept
                try:
                    with self._session_lock:
                        cookies = getattr(self.session, "cookies", None)
                        if cookies is not None and hasattr(cookies, "clear"):
                            cookies.clear()
                    resp = self.session.request(
                        "GET",
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                        stream=True,
                    )
                except Exception as e:  # noqa: BLE001 - any client error is a fetch error
                    raise FetchRefused("fetch error", type(e).__name__) from e
            status = int(getattr(resp, "status_code", 0) or 0)
            if status in _RETRY_CODES and attempt == 1:
                delay = parse_retry_after(_header(resp, "Retry-After"))
                delay = 1.0 if delay is None else delay
                _close(resp)
                if delay > MAX_RETRY_AFTER_SECONDS:
                    # Transient: a "fetch error", so it is not negative-cached.
                    raise FetchRefused("fetch error", f"HTTP {status}; Retry-After {delay:.0f}s")
                self._sleep(delay)
                continue
            return resp
        raise FetchRefused("fetch error", "retry loop exhausted")  # pragma: no cover

    def _get(self, url: str, *, accept: str, check_robots: bool = True) -> tuple[Any, str]:
        """GET following redirects manually; every hop is checked.

        Returns:
            ``(response, final_url)`` of a non-redirect response.
        """
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            scheme, netloc = self.check_url(current)
            if check_robots:
                self._check_robots(scheme, netloc, current)
            resp = self._send(current, netloc, accept)
            status = int(getattr(resp, "status_code", 0) or 0)
            if status in _REDIRECT_CODES:
                location = _header(resp, "Location")
                _close(resp)
                if not location:
                    raise FetchRefused("http error", f"HTTP {status} without Location")
                current = urljoin(current, str(location).strip())
                continue
            return resp, current
        raise FetchRefused("too many redirects", f"more than {MAX_REDIRECTS} redirects")

    @staticmethod
    def _read(resp: Any, cap: int) -> tuple[bytes, bool]:  # noqa: ANN401
        """Body up to ``cap`` bytes; returns ``(body, hit_cap)``."""
        chunks: list[bytes] = []
        total = 0
        hit_cap = False
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                remaining = cap - total
                if len(chunk) > remaining:
                    chunks.append(chunk[:remaining])
                    total += remaining
                    hit_cap = True
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as e:  # noqa: BLE001
            raise FetchRefused("fetch error", f"reading body failed ({type(e).__name__})") from e
        finally:
            _close(resp)
        return b"".join(chunks), hit_cap

    # -- robots.txt --------------------------------------------------------------

    def _check_robots(self, scheme: str, netloc: str, url: str) -> None:
        parser = self._robots_for(scheme, netloc)
        if not parser.can_fetch(USER_AGENT, url):
            raise FetchRefused("robots", f"disallowed by {netloc}/robots.txt")
        delay = parser.crawl_delay(USER_AGENT)
        if delay is not None:
            try:
                delay_f = float(delay)
            except (TypeError, ValueError):
                delay_f = 0.0
            if delay_f > MAX_CRAWL_DELAY_SECONDS:
                raise FetchRefused("robots", f"crawl-delay {delay_f:.0f}s is over 30 s")
            with self._hosts_lock:
                self._host_delay[netloc] = max(self._host_delay.get(netloc, 0.0), delay_f)

    def _robots_for(self, scheme: str, netloc: str) -> RobotFileParser:
        key = f"{scheme}://{netloc}"
        with self._hosts_lock:
            if key in self._robots:
                return self._robots[key]
            lock = self._robots_locks.setdefault(key, threading.Lock())
        with lock:
            with self._hosts_lock:
                if key in self._robots:
                    return self._robots[key]
            parser = self._load_robots(scheme, netloc, key)
            with self._hosts_lock:
                self._robots[key] = parser
            return parser

    def _load_robots(self, scheme: str, netloc: str, key: str) -> RobotFileParser:
        parser = RobotFileParser()
        cached = None
        if self.cache is not None and not self.refresh:
            cached = self.cache.get("robots", key, ROBOTS_TTL_SECONDS)
        if isinstance(cached, dict) and cached.get("state") in ("lines", "allow_all"):
            if cached["state"] == "allow_all":
                parser.allow_all = True
            else:
                parser.parse([str(x) for x in cached.get("lines") or []])
            return parser

        robots_url = f"{scheme}://{netloc}/robots.txt"
        try:
            resp, _final = self._get(robots_url, accept="text/plain,*/*;q=0.5", check_robots=False)
            status = int(getattr(resp, "status_code", 0) or 0)
            if status in (404, 410):
                _close(resp)
                parser.allow_all = True
                self._cache_robots(key, {"state": "allow_all"})
                return parser
            if not 200 <= status < 300:
                _close(resp)
                logger.info("robots.txt for %s returned HTTP %s; treating host as disallowed",
                            netloc, status)  # fmt: skip
                parser.disallow_all = True
                return parser
            body, _cut = self._read(resp, MAX_ROBOTS_BYTES)
        except FetchRefused as e:
            if e.reason == "budget":
                raise
            logger.info(
                "robots.txt for %s unavailable (%s); treating host as disallowed", netloc, e
            )
            parser.disallow_all = True
            return parser
        lines = body.decode("utf-8", errors="replace").splitlines()
        parser.parse(lines)
        self._cache_robots(key, {"state": "lines", "lines": lines})
        return parser

    def _cache_robots(self, key: str, payload: dict[str, Any]) -> None:
        if self.cache is not None:
            self.cache.put("robots", key, payload)

    # -- pages -----------------------------------------------------------------

    def fetch_page(self, url: str) -> FetchedPage:
        """Fetch and extract one page.

        Raises:
            FetchRefused: For every reason the page cannot be used.
        """
        if self.cache is not None and not self.refresh:
            hit = self.cache.get("pages", url, self.cache_ttl_seconds)
            page = _page_from_cache(url, hit)
            if page is not None:
                return page
            neg = self.cache.get("pages", "refused:" + url, NEGATIVE_TTL_SECONDS)
            if isinstance(neg, dict) and neg.get("reason"):
                raise FetchRefused(str(neg["reason"]), str(neg.get("detail") or "") + " (cached)")
        try:
            page = self._fetch_page_uncached(url)
        except FetchRefused as e:
            if self.cache is not None and e.reason in _CACHEABLE_REFUSALS:
                self.cache.put("pages", "refused:" + url, {"reason": e.reason, "detail": e.detail})
            raise
        if self.cache is not None:
            self.cache.put(
                "pages",
                url,
                {
                    "url": page.url,
                    "final_url": page.final_url,
                    "status": page.status,
                    "fetched_at": page.fetched_at,
                    "content_type": page.content_type,
                    "truncated": page.truncated,
                    "html": page.html_text.to_dict(),
                },
            )
        return page

    def _fetch_page_uncached(self, url: str) -> FetchedPage:
        resp, final_url = self._get(
            url, accept="text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5"
        )
        status = int(getattr(resp, "status_code", 0) or 0)
        if not 200 <= status < 300:
            _close(resp)
            # 429 is transient: never negative-cached for 24 h.
            reason = "http error" if 400 <= status < 500 and status != 429 else "fetch error"
            raise FetchRefused(reason, f"HTTP {status}")
        raw_type = str(_header(resp, "Content-Type") or "")
        media = _media_type(raw_type)
        if media not in PAGE_CONTENT_TYPES:
            _close(resp)
            raise FetchRefused("content-type", media or "(none)")
        body, hit_cap = self._read(resp, self.max_page_bytes)
        text = _decode(body, raw_type)
        if media == "text/plain":
            plain = extract(f"<pre>{_escape(text)}</pre>")
            extracted = ExtractedHtml(
                title=None, text=plain.text, text_truncated=plain.text_truncated
            )
        else:
            extracted = extract(text)
        if hit_cap and not extracted.text_truncated:
            # The body was cut mid-stream: drop the last, possibly partial word
            # (``$1,2`` of ``$1,299.99``) so no fragment reaches the excerpts.
            extracted = dataclasses.replace(extracted, text=drop_partial_tail(extracted.text))
        return FetchedPage(
            url=url,
            final_url=final_url,
            status=status,
            fetched_at=utc_now_iso(),
            content_type=media,
            html_text=extracted,
            truncated=hit_cap or extracted.text_truncated,
            from_cache=False,
        )

    # -- images ----------------------------------------------------------------

    def fetch_image(self, url: str) -> tuple[bytes, str]:
        """Download and verify one image.

        Returns:
            ``(data, ext)`` where ``ext`` is ``jpg``, ``png`` or ``webp``.

        Raises:
            FetchRefused: Wrong type, too big, too small, corrupt, or any
                fetch refusal.
        """
        if self.cache is not None and not self.refresh:
            hit = self.cache.get_bytes("images", url, self.cache_ttl_seconds)
            if hit is not None:
                data, meta, _path = hit
                try:
                    return data, verify_image(data, declared=str(meta.get("content_type") or ""))
                except FetchRefused:
                    pass
        resp, _final = self._get(url, accept="image/webp,image/jpeg,image/png;q=0.9")
        status = int(getattr(resp, "status_code", 0) or 0)
        if not 200 <= status < 300:
            _close(resp)
            reason = "http error" if status < 500 and status != 429 else "fetch error"
            raise FetchRefused(reason, f"HTTP {status}")
        media = _media_type(_header(resp, "Content-Type"))
        if media not in IMAGE_CONTENT_TYPES:
            _close(resp)
            raise FetchRefused("content-type", media or "(none)")
        length = _header(resp, "Content-Length")
        try:
            if length is not None and int(length) > self.max_image_bytes:
                _close(resp)
                raise FetchRefused("too big", f"{int(length)} bytes > {self.max_image_bytes}")
        except ValueError:
            pass
        data, hit_cap = self._read(resp, self.max_image_bytes)
        if hit_cap:
            raise FetchRefused("too big", f"over {self.max_image_bytes} bytes")
        ext = verify_image(data, declared=media)
        if self.cache is not None:
            self.cache.put_bytes("images", url, data, ext=ext, meta={"content_type": media})
        return data, ext


def verify_image(data: bytes, *, declared: str = "") -> str:
    """Check an image with Pillow; returns its extension (jpg/png/webp).

    Raises:
        FetchRefused: ``bad image`` (corrupt, wrong format, too many pixels)
            or ``too small`` (short edge under :data:`MIN_IMAGE_EDGE`).
    """
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
            img.verify()
    except Exception as e:  # noqa: BLE001 - PIL raises many types (incl. bomb errors)
        raise FetchRefused("bad image", f"not a valid image ({type(e).__name__})") from e
    ext = _PIL_FORMATS.get(fmt)
    if ext is None:
        raise FetchRefused("bad image", f"format {fmt or '(unknown)'} not allowed")
    if declared and IMAGE_CONTENT_TYPES.get(_media_type(declared)) not in (None, ext):
        raise FetchRefused("bad image", f"declared {declared} but data is {fmt}")
    limit = Image.MAX_IMAGE_PIXELS
    if limit and width * height > limit:
        raise FetchRefused("bad image", f"{width}x{height} exceeds the pixel limit")
    if min(width, height) < MIN_IMAGE_EDGE:
        raise FetchRefused("too small", f"{width}x{height}; short edge under {MIN_IMAGE_EDGE}px")
    return ext


def _page_from_cache(url: str, hit: Any) -> Optional[FetchedPage]:  # noqa: ANN401
    if not isinstance(hit, dict):
        return None
    try:
        return FetchedPage(
            url=url,
            final_url=str(hit["final_url"]),
            status=int(hit["status"]),
            fetched_at=str(hit["fetched_at"]),
            content_type=str(hit.get("content_type") or ""),
            html_text=ExtractedHtml.from_dict(hit["html"]),
            truncated=bool(hit.get("truncated", False)),
            from_cache=True,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _header(resp: Any, name: str) -> Optional[str]:  # noqa: ANN401
    headers = getattr(resp, "headers", None) or {}
    try:
        value = headers.get(name)
        if value is None:
            lower = name.lower()
            for k, v in headers.items():
                if str(k).lower() == lower:
                    return v
        return value
    except AttributeError:
        return None


def _close(resp: Any) -> None:  # noqa: ANN401
    close = getattr(resp, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
