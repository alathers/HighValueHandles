"""Tests for lib.web_fetch (fake session, resolver, clock and sleep)."""

import io
import socket
import urllib.request
from pathlib import Path
from typing import Any, Optional, Union

import pytest
from PIL import Image

from lib.web_cache import WebCache
from lib.web_fetch import (
    USER_AGENT,
    FetchRefused,
    HttpFetcher,
    parse_retry_after,
    verify_image,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web"
PUBLIC_IP = "93.184.216.34"
HTML = "text/html; charset=utf-8"


class FakeResp:
    def __init__(
        self,
        status: int = 200,
        body: Union[bytes, str] = b"",
        headers: Optional[dict] = None,
        chunk: int = 1024,
    ) -> None:
        self.status_code = status
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}
        self.chunk = chunk
        self.closed = False
        self.read = 0

    def iter_content(self, chunk_size: int = 1024):
        step = min(chunk_size, self.chunk)
        for i in range(0, len(self.body), step):
            self.read += step
            yield self.body[i : i + step]

    def close(self) -> None:
        self.closed = True


class FakeCookies(dict):
    def __init__(self) -> None:
        super().__init__()
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1
        super().clear()


class FakeSession:
    """Routes by URL; a list value is consumed in order (last one repeats)."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[dict[str, Any]] = []
        self.cookies = FakeCookies()

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # noqa: ANN401
        self.calls.append({"method": method, "url": url, **kwargs})
        if url not in self.routes:
            return FakeResp(404)
        item = self.routes[url]
        if isinstance(item, list):
            item = item.pop(0) if len(item) > 1 else item[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item()
        return item

    def urls(self) -> list[str]:
        return [c["url"] for c in self.calls]


class FakeTime:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(round(s, 6))
        self.t += s


def resolver_for(mapping: Optional[dict[str, str]] = None):
    mapping = mapping or {}

    def resolve(host, port, *args, **kwargs):
        ip = mapping.get(host, PUBLIC_IP)
        if ip is None:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    return resolve


def make_fetcher(routes: dict[str, Any], tmp_path: Optional[Path] = None, **kw: Any):
    ft = FakeTime()
    session = FakeSession(routes)
    kw.setdefault("resolver", resolver_for())
    fetcher = HttpFetcher(
        session=session,
        cache=WebCache(tmp_path / "web") if tmp_path is not None else None,
        clock=ft.clock,
        sleep=ft.sleep,
        **kw,
    )
    return fetcher, session, ft


def page(
    body: str = "<p>Example Creek</p>", status: int = 200, ctype: str = HTML, **h: str
) -> FakeResp:
    headers = {"Content-Type": ctype, **h}
    return FakeResp(status, body, headers)


def robots(text: str) -> FakeResp:
    return FakeResp(200, text, {"Content-Type": "text/plain"})


ROBOTS_OPEN = "User-agent: *\nDisallow:\n"


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def test_robots_allow(tmp_path: Path):
    f, s, _ = make_fetcher(
        {
            "https://a.example.com/robots.txt": robots(ROBOTS_OPEN),
            "https://a.example.com/p": page(),
        }
    )
    got = f.fetch_page("https://a.example.com/p")
    assert got.text == "Example Creek"
    assert got.status == 200
    assert got.from_cache is False
    assert s.urls() == ["https://a.example.com/robots.txt", "https://a.example.com/p"]


def test_robots_deny():
    f, s, _ = make_fetcher(
        {
            "https://a.example.com/robots.txt": robots("User-agent: *\nDisallow: /private\n"),
            "https://a.example.com/private/p": page(),
        }
    )
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/private/p")
    assert ei.value.reason == "robots"
    assert "https://a.example.com/private/p" not in s.urls()


def test_robots_deny_for_our_agent_only():
    f, _, _ = make_fetcher(
        {
            "https://a.example.com/robots.txt": robots(
                "User-agent: bourbon-process\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
            ),
        }
    )
    with pytest.raises(FetchRefused, match="robots"):
        f.fetch_page("https://a.example.com/p")


def test_robots_404_means_allowed():
    f, _, _ = make_fetcher({"https://a.example.com/p": page()})  # robots -> default 404
    assert f.fetch_page("https://a.example.com/p").text == "Example Creek"


@pytest.mark.parametrize("status", [500, 503])
def test_robots_5xx_means_disallowed(status: int):
    f, s, _ = make_fetcher(
        {"https://a.example.com/robots.txt": FakeResp(status), "https://a.example.com/p": page()}
    )
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "robots"
    assert "https://a.example.com/p" not in s.urls()


def test_robots_timeout_means_disallowed():
    f, _, _ = make_fetcher(
        {
            "https://a.example.com/robots.txt": TimeoutError("slow"),
            "https://a.example.com/p": page(),
        }
    )
    with pytest.raises(FetchRefused, match="robots"):
        f.fetch_page("https://a.example.com/p")


def test_robots_never_uses_urllib(monkeypatch: pytest.MonkeyPatch):
    def no_urlopen(*a, **k):
        raise AssertionError("urllib.request.urlopen must not be used")

    monkeypatch.setattr(urllib.request, "urlopen", no_urlopen)
    f, s, _ = make_fetcher(
        {"https://a.example.com/robots.txt": robots(ROBOTS_OPEN), "https://a.example.com/p": page()}
    )
    f.fetch_page("https://a.example.com/p")
    assert s.urls()[0] == "https://a.example.com/robots.txt"


def test_robots_fetched_once_per_host():
    f, s, _ = make_fetcher(
        {
            "https://a.example.com/robots.txt": robots(ROBOTS_OPEN),
            "https://a.example.com/p1": page(),
            "https://a.example.com/p2": page(),
        }
    )
    f.fetch_page("https://a.example.com/p1")
    f.fetch_page("https://a.example.com/p2")
    assert s.urls().count("https://a.example.com/robots.txt") == 1


def test_crawl_delay_over_30s_skips_host():
    f, _, _ = make_fetcher(
        {"https://a.example.com/robots.txt": robots("User-agent: *\nCrawl-delay: 60\nDisallow:\n")}
    )
    with pytest.raises(FetchRefused, match="crawl-delay"):
        f.fetch_page("https://a.example.com/p")


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://[::1]/x",
        "http://93.184.216.34/x",  # any IP literal
        "http://localhost/x",
        "http://printer.local/x",
        "http://db.internal/x",
        "http://intranet/x",
        "file:///etc/passwd",
        "ftp://a.example.com/x",
        "https://user:pw@a.example.com/x",
    ],
)
def test_unsafe_urls_refused_without_request(url: str):
    f, s, _ = make_fetcher({})
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page(url)
    assert ei.value.reason in ("unsafe host", "unsafe scheme")
    assert s.calls == []


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "192.168.1.5", "169.254.169.254", "::1"])
def test_host_resolving_to_private_address_refused(ip: str):
    f, s, _ = make_fetcher({}, resolver=resolver_for({"evil.example.com": ip}))
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://evil.example.com/x")
    assert ei.value.reason == "unsafe host"
    assert s.calls == []


@pytest.mark.parametrize(
    "location",
    ["http://127.0.0.1/admin", "http://10.0.0.1/", "http://localhost/", "http://evil.example.com/"],
)
def test_redirect_to_unsafe_target_refused(location: str):
    f, s, _ = make_fetcher(
        {"https://a.example.com/p": FakeResp(302, b"", {"Location": location})},
        resolver=resolver_for({"evil.example.com": "10.1.2.3"}),
    )
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "unsafe host"
    assert not any(u.startswith(location) for u in s.urls())


def test_every_hop_is_checked_and_followed_manually():
    checked: list[str] = []
    base = resolver_for()

    def recording(host, port, *a, **k):
        checked.append(host)
        return base(host, port)

    f, s, _ = make_fetcher(
        {
            "https://a.example.com/p": FakeResp(301, b"", {"Location": "https://b.example.com/q"}),
            "https://b.example.com/q": FakeResp(302, b"", {"Location": "/r"}),
            "https://b.example.com/r": page(),
        },
        resolver=recording,
    )
    got = f.fetch_page("https://a.example.com/p")
    assert got.final_url == "https://b.example.com/r"
    assert checked.count("a.example.com") >= 2  # robots + page
    assert checked.count("b.example.com") >= 2
    assert all(c["allow_redirects"] is False for c in s.calls)


def test_too_many_redirects():
    routes = {
        f"https://a.example.com/{i}": FakeResp(302, b"", {"Location": f"/{i + 1}"})
        for i in range(10)
    }
    f, _, _ = make_fetcher(routes)
    with pytest.raises(FetchRefused, match="redirects"):
        f.fetch_page("https://a.example.com/0")


def test_default_resolver_is_looked_up_at_call_time(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_getaddrinfo(host, port, *a, **k):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    f = HttpFetcher(session=FakeSession({}), cache=None, sleep=lambda s: None)
    with pytest.raises(FetchRefused, match="unsafe host"):
        f.fetch_page("https://a.example.com/x")
    assert calls == ["a.example.com"]


# ---------------------------------------------------------------------------
# Body, type, retries
# ---------------------------------------------------------------------------


def test_byte_cap_truncates_page():
    body = "<p>" + "a" * 5000 + "</p>"
    resp = FakeResp(200, body, {"Content-Type": HTML}, chunk=500)
    f, _, _ = make_fetcher({"https://a.example.com/p": resp}, max_page_bytes=1000)
    got = f.fetch_page("https://a.example.com/p")
    assert got.truncated is True
    assert len(got.text) <= 1000
    assert resp.read <= 1500  # stopped streaming near the cap
    assert resp.closed


def test_byte_cap_drops_partial_last_word():
    # The cap falls inside "$1,299.99": the fragment "$1,2" must not survive.
    body = "<p>Example Creek MSRP $1,299.99 today</p>"
    cap = body.index("299")
    resp = FakeResp(200, body, {"Content-Type": HTML}, chunk=8)
    f, _, _ = make_fetcher({"https://a.example.com/p": resp}, max_page_bytes=cap)
    got = f.fetch_page("https://a.example.com/p")
    assert got.truncated is True
    assert got.text == "Example Creek MSRP"


def test_page_under_cap_not_truncated():
    f, _, _ = make_fetcher({"https://a.example.com/p": page()}, max_page_bytes=1000)
    assert f.fetch_page("https://a.example.com/p").truncated is False


@pytest.mark.parametrize("ctype", ["application/pdf", "image/jpeg", "application/json", ""])
def test_content_type_refused(ctype: str):
    f, _, _ = make_fetcher({"https://a.example.com/p": page(ctype=ctype)})
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "content-type"


def test_text_plain_accepted():
    f, _, _ = make_fetcher(
        {"https://a.example.com/p": page("line one\nline <two>", ctype="text/plain")}
    )
    assert f.fetch_page("https://a.example.com/p").text == "line one\nline <two>"


def test_http_404_is_refused():
    f, _, _ = make_fetcher({"https://a.example.com/robots.txt": robots(ROBOTS_OPEN)})
    with pytest.raises(FetchRefused, match="HTTP 404"):
        f.fetch_page("https://a.example.com/missing")


def test_429_retry_after_honoured():
    f, s, ft = make_fetcher(
        {"https://a.example.com/p": [page(status=429, **{"Retry-After": "5"}), page()]},
        per_host_interval=0,
    )
    assert f.fetch_page("https://a.example.com/p").text == "Example Creek"
    assert 5 in ft.sleeps
    assert s.urls().count("https://a.example.com/p") == 2


def test_503_retried_only_once():
    f, s, _ = make_fetcher({"https://a.example.com/p": [page(status=503)]}, per_host_interval=0)
    with pytest.raises(FetchRefused):
        f.fetch_page("https://a.example.com/p")
    assert s.urls().count("https://a.example.com/p") == 2


def test_other_4xx_not_retried():
    f, s, _ = make_fetcher({"https://a.example.com/p": page(status=403)})
    with pytest.raises(FetchRefused):
        f.fetch_page("https://a.example.com/p")
    assert s.urls().count("https://a.example.com/p") == 1


def test_network_error_is_fetch_error():
    f, _, _ = make_fetcher({"https://a.example.com/p": ConnectionError("down")})
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "fetch error"


def test_parse_retry_after():
    assert parse_retry_after("7") == 7
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


# ---------------------------------------------------------------------------
# Rate limits and budget
# ---------------------------------------------------------------------------


def test_per_host_interval():
    f, _, ft = make_fetcher(
        {
            "https://a.example.com/p1": page(),
            "https://a.example.com/p2": page(),
            "https://b.example.com/p": page(),
        },
        per_host_interval=2.0,
    )
    f.fetch_page("https://a.example.com/p1")  # robots + page: one 2 s wait
    f.fetch_page("https://a.example.com/p2")  # another 2 s wait
    f.fetch_page("https://b.example.com/p")  # robots + page on b: one wait
    assert ft.sleeps == [2.0, 2.0, 2.0]


def test_crawl_delay_extends_interval():
    f, _, ft = make_fetcher(
        {
            "https://a.example.com/robots.txt": robots(
                "User-agent: *\nCrawl-delay: 7\nDisallow:\n"
            ),
            "https://a.example.com/p1": page(),
            "https://a.example.com/p2": page(),
        },
        per_host_interval=2.0,
    )
    f.fetch_page("https://a.example.com/p1")
    f.fetch_page("https://a.example.com/p2")
    assert ft.sleeps == [7.0, 7.0]


def test_run_budget():
    routes = {f"https://a.example.com/{i}": page() for i in range(5)}
    f, s, _ = make_fetcher(routes, max_requests=3)
    f.fetch_page("https://a.example.com/0")  # robots + page = 2
    f.fetch_page("https://a.example.com/1")  # 3
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/2")
    assert ei.value.reason == "budget"
    assert len(s.calls) == 3


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def test_headers_generic_no_cookie_no_referer():
    f, s, _ = make_fetcher({"https://a.example.com/p": page()})
    s.cookies["session"] = "abc"
    f.fetch_page("https://a.example.com/p")
    for call in s.calls:
        headers = {k.lower(): v for k, v in call["headers"].items()}
        assert headers["user-agent"] == USER_AGENT
        assert "cookie" not in headers
        assert "referer" not in headers
        assert headers["accept-language"] == "en-US"
        assert call["stream"] is True
    assert s.cookies.clears == len(s.calls)
    assert s.cookies == {}


def test_user_agent_has_no_personal_data():
    assert USER_AGENT.startswith("bourbon-process/")
    assert "(label-research; respects robots.txt)" in USER_AGENT
    assert "@" not in USER_AGENT


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_hit_and_refresh_bypass(tmp_path: Path):
    routes = {"https://a.example.com/p": page((FIXTURES / "product_official.html").read_text())}
    f, s, _ = make_fetcher(routes, tmp_path)
    first = f.fetch_page("https://a.example.com/p")
    n = len(s.calls)
    again = f.fetch_page("https://a.example.com/p")
    assert again.from_cache is True
    assert again.html_text == first.html_text
    assert len(s.calls) == n

    f2, s2, _ = make_fetcher(routes, tmp_path, refresh=True)
    fresh = f2.fetch_page("https://a.example.com/p")
    assert fresh.from_cache is False
    assert "https://a.example.com/p" in s2.urls()


def test_negative_result_cached(tmp_path: Path):
    routes = {"https://a.example.com/p": page(ctype="application/pdf")}
    f, s, _ = make_fetcher(routes, tmp_path)
    with pytest.raises(FetchRefused):
        f.fetch_page("https://a.example.com/p")
    n = len(s.calls)
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "content-type"
    assert len(s.calls) == n


@pytest.mark.parametrize(
    "responses",
    [
        [page(status=429), page(status=429)],
        [page(status=429, **{"Retry-After": "120"})],
        [page(status=503, **{"Retry-After": "120"})],
    ],
)
def test_transient_refusals_are_not_negative_cached(tmp_path: Path, responses: list):
    routes = {"https://a.example.com/p": [*responses, page()]}
    f, _, _ = make_fetcher(routes, tmp_path, per_host_interval=0)
    with pytest.raises(FetchRefused) as ei:
        f.fetch_page("https://a.example.com/p")
    assert ei.value.reason == "fetch error"
    # Not cached for 24 h: the next call goes back to the network and succeeds.
    assert f.fetch_page("https://a.example.com/p").text == "Example Creek"


def test_robots_cached_on_disk(tmp_path: Path):
    routes = {
        "https://a.example.com/robots.txt": robots(ROBOTS_OPEN),
        "https://a.example.com/p": page(),
    }
    f, _, _ = make_fetcher(routes, tmp_path)
    f.fetch_page("https://a.example.com/p")
    # A new fetcher (new run) reads robots.txt from the disk cache.
    f2, s2, _ = make_fetcher({"https://a.example.com/q": page()}, tmp_path)
    assert f2.fetch_page("https://a.example.com/q").text == "Example Creek"
    assert s2.urls() == ["https://a.example.com/q"]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _image(fmt: str = "JPEG", size: tuple[int, int] = (400, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 60, 20)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("fmt", "ctype", "ext"),
    [("JPEG", "image/jpeg", "jpg"), ("PNG", "image/png", "png"), ("WEBP", "image/webp", "webp")],
)
def test_fetch_image_ok(fmt: str, ctype: str, ext: str):
    data = _image(fmt)
    f, _, _ = make_fetcher(
        {"https://cdn.example.com/a": FakeResp(200, data, {"Content-Type": ctype})}
    )
    got, got_ext = f.fetch_image("https://cdn.example.com/a")
    assert got == data
    assert got_ext == ext


def test_fetch_image_wrong_type():
    f, _, _ = make_fetcher(
        {"https://cdn.example.com/a": FakeResp(200, _image("GIF"), {"Content-Type": "image/gif"})}
    )
    with pytest.raises(FetchRefused, match="content-type"):
        f.fetch_image("https://cdn.example.com/a")


def test_fetch_image_too_big():
    data = _image("PNG", (800, 800))
    f, _, _ = make_fetcher(
        {"https://cdn.example.com/a": FakeResp(200, data, {"Content-Type": "image/png"})},
        max_image_bytes=len(data) - 1,
    )
    with pytest.raises(FetchRefused, match="too big"):
        f.fetch_image("https://cdn.example.com/a")


def test_fetch_image_too_small():
    f, _, _ = make_fetcher(
        {
            "https://cdn.example.com/a": FakeResp(
                200, _image("JPEG", (299, 800)), {"Content-Type": "image/jpeg"}
            )
        }
    )
    with pytest.raises(FetchRefused, match="too small"):
        f.fetch_image("https://cdn.example.com/a")


def test_fetch_image_corrupt_or_mislabelled():
    f, _, _ = make_fetcher(
        {
            "https://cdn.example.com/a": FakeResp(
                200, b"not an image", {"Content-Type": "image/jpeg"}
            )
        }
    )
    with pytest.raises(FetchRefused, match="bad image"):
        f.fetch_image("https://cdn.example.com/a")
    with pytest.raises(FetchRefused, match="bad image"):
        verify_image(_image("PNG"), declared="image/jpeg")


def test_fetch_image_pixel_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(FetchRefused, match="bad image"):
        verify_image(_image("PNG", (400, 400)))


def test_fetch_image_cached(tmp_path: Path):
    data = _image()
    f, s, _ = make_fetcher(
        {"https://cdn.example.com/a": FakeResp(200, data, {"Content-Type": "image/jpeg"})}, tmp_path
    )
    f.fetch_image("https://cdn.example.com/a")
    n = len(s.calls)
    assert f.fetch_image("https://cdn.example.com/a") == (data, "jpg")
    assert len(s.calls) == n


def test_body_exactly_at_cap_is_not_truncated():
    body = "<p>" + "a" * 993 + "</p>"  # 1000 bytes
    resp = FakeResp(200, body, {"Content-Type": HTML}, chunk=500)
    f, _, _ = make_fetcher({"https://a.example.com/p": resp}, max_page_bytes=1000)
    assert f.fetch_page("https://a.example.com/p").truncated is False
