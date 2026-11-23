"""Tests for lib.search_providers (all HTTP through a fake session)."""

import logging
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pytest

from lib.search_providers import (
    BraveSearchProvider,
    DuckDuckGoSearchProvider,
    OllamaSearchProvider,
    SearchHit,
    SearchProviderAuthError,
    SearchProviderDisabledError,
    SearchProviderError,
    SearxngSearchProvider,
    build_provider,
    parse_ddg_results,
    redact,
    resolve_provider_name,
)
from lib.web_cache import WebCache
from lib.web_fetch import USER_AGENT

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web"
KEY = "sk-test-SECRET-KEY-12345"


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,  # noqa: ANN401
        text: str = "",
        headers: Optional[dict] = None,
    ) -> None:
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:  # noqa: ANN401
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Duck-typed session: records every call, replays queued responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.cookies: dict = {}

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # noqa: ANN401
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTime:
    def __init__(self) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _kw(ft: FakeTime) -> dict[str, Any]:
    return {"clock": ft.clock, "sleep": ft.sleep}


BRAVE_JSON = {
    "web": {
        "results": [
            {
                "url": "https://a.example.com/x",
                "title": "A <strong>X</strong>",
                "description": "d &amp; e",
            },
            {"url": "ftp://bad.example.com/", "title": "bad", "description": ""},
            {"url": "https://b.example.com/y", "title": "B", "description": "f"},
        ]
    }
}


# ---------------------------------------------------------------------------
# Keyed providers
# ---------------------------------------------------------------------------


def test_brave_request_and_parse():
    ft = FakeTime()
    session = FakeSession([FakeResponse(payload=BRAVE_JSON)])
    p = BraveSearchProvider(KEY, session, **_kw(ft))
    hits = p.search('"Example Creek" MSRP', max_results=5)
    assert hits == [
        SearchHit("https://a.example.com/x", "A X", "d & e", 1, "brave"),
        SearchHit("https://b.example.com/y", "B", "f", 2, "brave"),
    ]
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "https://api.search.brave.com/res/v1/web/search"
    assert call["params"] == {"q": '"Example Creek" MSRP', "count": 5}
    assert call["headers"]["X-Subscription-Token"] == KEY
    assert call["headers"]["User-Agent"] == USER_AGENT


def test_ollama_request_and_parse():
    ft = FakeTime()
    payload = {"results": [{"title": "T", "url": "https://o.example.com/", "content": "c" * 900}]}
    session = FakeSession([FakeResponse(payload=payload)])
    p = OllamaSearchProvider(KEY, session, **_kw(ft))
    hits = p.search("q", max_results=50)
    assert len(hits) == 1
    assert hits[0].provider == "ollama"
    assert len(hits[0].snippet) == 500
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://ollama.com/api/web_search"
    assert call["json"] == {"query": "q", "max_results": 10}
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_searxng_request_and_parse():
    ft = FakeTime()
    payload = {"results": [{"url": "https://s.example.com/", "title": "S", "content": "x"}]}
    session = FakeSession([FakeResponse(payload=payload)])
    p = SearxngSearchProvider("http://127.0.0.1:8888/", session, **_kw(ft))
    hits = p.search("q", max_results=3)
    assert [h.url for h in hits] == ["https://s.example.com/"]
    assert session.calls[0]["url"] == "http://127.0.0.1:8888/search"
    assert session.calls[0]["params"] == {"q": "q", "format": "json"}


def test_searxng_403_explains_json_format():
    session = FakeSession([FakeResponse(status=403)])
    p = SearxngSearchProvider("http://127.0.0.1:8888", session, **_kw(FakeTime()))
    with pytest.raises(
        SearchProviderError, match="enable json in searxng settings.yml search.formats"
    ) as ei:
        p.search("q", max_results=3)
    assert not isinstance(ei.value, SearchProviderAuthError)


@pytest.mark.parametrize("cls", [BraveSearchProvider, OllamaSearchProvider])
def test_key_only_in_header_never_url_repr_or_logs(cls, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    session = FakeSession([FakeResponse(payload={}), ConnectionError(f"boom {KEY}")])
    p = cls(KEY, session, **_kw(FakeTime()))
    p.search("q", max_results=3)
    with pytest.raises(SearchProviderError) as ei:
        p.search("q2", max_results=3)
    assert KEY not in str(ei.value)
    for call in session.calls:
        assert KEY not in call["url"]
        assert KEY not in repr(call.get("params"))
        assert KEY not in repr(call.get("json"))
        assert KEY in repr(call["headers"])
    assert KEY not in repr(p)
    assert "key=***" in repr(p)
    assert KEY not in str(p)
    assert KEY not in caplog.text


def test_401_raises_auth_error():
    p = BraveSearchProvider(KEY, FakeSession([FakeResponse(status=401)]), **_kw(FakeTime()))
    with pytest.raises(SearchProviderAuthError):
        p.search("q", max_results=3)


def test_429_retried_once_honouring_retry_after():
    ft = FakeTime()
    session = FakeSession(
        [FakeResponse(status=429, headers={"Retry-After": "7"}), FakeResponse(payload=BRAVE_JSON)]
    )
    p = BraveSearchProvider(KEY, session, **_kw(ft))
    assert len(p.search("q", max_results=5)) == 2
    assert 7 in ft.sleeps
    assert len(session.calls) == 2


def test_429_twice_raises():
    ft = FakeTime()
    session = FakeSession([FakeResponse(status=429), FakeResponse(status=429)])
    p = BraveSearchProvider(KEY, session, **_kw(ft))
    with pytest.raises(SearchProviderError, match="429"):
        p.search("q", max_results=5)
    assert len(session.calls) == 2


def test_retry_after_over_30s_is_not_waited():
    ft = FakeTime()
    session = FakeSession([FakeResponse(status=429, headers={"Retry-After": "3600"})])
    p = BraveSearchProvider(KEY, session, **_kw(ft))
    with pytest.raises(SearchProviderError):
        p.search("q", max_results=5)
    assert all(s <= 30 for s in ft.sleeps)


def test_rate_limit_one_per_second():
    ft = FakeTime()
    session = FakeSession([FakeResponse(payload={}), FakeResponse(payload={})])
    p = BraveSearchProvider(KEY, session, **_kw(ft))
    p.search("a", max_results=1)
    p.search("b", max_results=1)
    assert ft.sleeps == [pytest.approx(1.0)]


def test_results_cached_24h(tmp_path: Path):
    ft = FakeTime()
    cache = WebCache(tmp_path)
    session = FakeSession([FakeResponse(payload=BRAVE_JSON)])
    p = BraveSearchProvider(KEY, session, cache=cache, **_kw(ft))
    first = p.search("q", max_results=5)
    again = BraveSearchProvider(KEY, FakeSession([]), cache=cache, **_kw(ft)).search(
        "q", max_results=5
    )
    assert again == first
    for f in (tmp_path / "search").iterdir():
        assert KEY not in f.read_text(encoding="utf-8")


def test_invalid_json_is_provider_error():
    p = BraveSearchProvider(
        KEY, FakeSession([FakeResponse(payload=ValueError("x"))]), **_kw(FakeTime())
    )
    with pytest.raises(SearchProviderError, match="invalid JSON"):
        p.search("q", max_results=3)


def test_redact():
    assert redact(f"a {KEY} b", [KEY, None, ""]) == "a *** b"


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------


def _ddg(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_ddg_parse_unwraps_redirects_and_drops_ads():
    rows = parse_ddg_results(_ddg("ddg_html_results.html"))
    assert rows == [
        (
            "https://www.examplecreekdistillery.com/bourbon/small-batch?x=1",
            "Example Creek Small Batch | Official Site",
            "Our flagship bourbon & the 2023 release's tasting notes.",
        ),
        (
            "https://bottleshop.example.com/example-creek-small-batch",
            "Example Creek Small Batch 750ml",
            "In stock. Price $59.99 — ships fast.",
        ),
    ]


def test_ddg_parse_lite_page():
    rows = parse_ddg_results(_ddg("ddg_lite_results.html"))
    assert [r[0] for r in rows] == [
        "https://whiskeywiki.example.org/example-creek",
        "https://news.example.net/example-creek-2023",
    ]
    assert rows[1][2] == "The suggested retail price is $59.99."


def test_ddg_search_posts_form_with_generic_user_agent():
    ft = FakeTime()
    session = FakeSession([FakeResponse(text=_ddg("ddg_html_results.html"))])
    p = DuckDuckGoSearchProvider(session, **_kw(ft))
    hits = p.search('"Example Creek" MSRP', max_results=8)
    assert [h.rank for h in hits] == [1, 2]
    assert all(h.provider == "duckduckgo" for h in hits)
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://html.duckduckgo.com/html/"
    assert call["data"] == {"q": '"Example Creek" MSRP'}
    assert call["headers"]["User-Agent"] == USER_AGENT
    assert "Cookie" not in call["headers"]
    assert "Referer" not in call["headers"]


def test_ddg_falls_back_to_lite_when_html_has_no_results():
    ft = FakeTime()
    session = FakeSession(
        [
            FakeResponse(text=_ddg("ddg_empty.html")),
            FakeResponse(text=_ddg("ddg_lite_results.html")),
        ]
    )
    hits = DuckDuckGoSearchProvider(session, **_kw(ft)).search("q", max_results=8)
    assert len(hits) == 2
    assert session.calls[1]["url"] == "https://lite.duckduckgo.com/lite/"
    assert ft.sleeps
    assert ft.sleeps[0] >= 2.0  # politeness interval between the two posts


def test_ddg_no_results_anywhere_is_empty():
    session = FakeSession(
        [FakeResponse(text=_ddg("ddg_empty.html")), FakeResponse(text=_ddg("ddg_empty.html"))]
    )
    assert DuckDuckGoSearchProvider(session, **_kw(FakeTime())).search("q", max_results=8) == []


def test_ddg_queries_at_least_two_seconds_apart():
    ft = FakeTime()
    page = _ddg("ddg_html_results.html")
    session = FakeSession([FakeResponse(text=page), FakeResponse(text=page)])
    p = DuckDuckGoSearchProvider(session, **_kw(ft))
    p.search("a", max_results=8)
    p.search("b", max_results=8)
    assert ft.sleeps == [pytest.approx(2.0)]


@pytest.mark.parametrize(
    "first",
    [
        FakeResponse(status=202, text=_ddg("ddg_202.html")),
        FakeResponse(status=403),
        FakeResponse(status=429),
        FakeResponse(text=_ddg("ddg_anomaly.html")),
    ],
)
def test_ddg_throttle_is_retried_once_after_backoff(first):
    ft = FakeTime()
    session = FakeSession([first, FakeResponse(text=_ddg("ddg_html_results.html"))])
    p = DuckDuckGoSearchProvider(session, throttle_backoff=10.0, **_kw(ft))
    assert len(p.search("q", max_results=8)) == 2
    assert 10.0 in ft.sleeps
    assert p.consecutive_throttles == 0


def test_ddg_throttled_twice_raises_then_disables_after_three(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    ft = FakeTime()
    throttled = [FakeResponse(status=202, text=_ddg("ddg_202.html")) for _ in range(6)]
    session = FakeSession(throttled)
    p = DuckDuckGoSearchProvider(session, **_kw(ft))
    for _ in range(2):
        with pytest.raises(SearchProviderError, match="duckduckgo throttled") as ei:
            p.search("q", max_results=8)
        assert not isinstance(ei.value, SearchProviderDisabledError)
    with pytest.raises(SearchProviderDisabledError):
        p.search("q", max_results=8)
    assert p.disabled
    calls = len(session.calls)
    with pytest.raises(SearchProviderDisabledError):
        p.search("q", max_results=8)
    assert len(session.calls) == calls  # no more requests once disabled
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_ddg_success_resets_throttle_count():
    ft = FakeTime()
    t = lambda: FakeResponse(status=202)  # noqa: E731
    ok = FakeResponse(text=_ddg("ddg_html_results.html"))
    session = FakeSession([t(), t(), t(), t(), ok, t(), t(), t(), t()])
    p = DuckDuckGoSearchProvider(session, **_kw(ft))
    for _ in range(2):
        with pytest.raises(SearchProviderError):
            p.search("q", max_results=8)
    p.search("ok", max_results=8)
    assert p.consecutive_throttles == 0
    with pytest.raises(SearchProviderError):
        p.search("q2", max_results=8)
    assert not p.disabled


def test_ddg_redirect_unwrap_encoded_target():
    target = "https://shop.example.com/p?a=1&b=2"
    html = (
        f'<a class="result__a" href="//duckduckgo.com/l/?uddg={quote(target, safe="")}&rut=z">T</a>'
    )
    assert parse_ddg_results(html) == [(target, "T", "")]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "searxng", "expected"),
    [
        ({"BRAVE_API_KEY": "b" * 8, "OLLAMA_API_KEY": "o" * 8}, "http://s", "brave"),
        ({"BOURBON_BRAVE_API_KEY": "b" * 8}, "", "brave"),
        ({"OLLAMA_API_KEY": "o" * 8}, "http://s", "ollama"),
        ({}, "http://127.0.0.1:8888", "searxng"),
        ({}, "", "duckduckgo"),
        ({"BRAVE_API_KEY": "  "}, "", "duckduckgo"),
    ],
)
def test_auto_resolution_order(env, searxng, expected):
    name, _reason = resolve_provider_name("auto", env, searxng_url=searxng)
    assert name == expected
    provider = build_provider("auto", env, FakeSession([]), searxng_url=searxng)
    assert provider is not None
    assert provider.name == expected


def test_build_provider_explicit_names():
    env = {"BRAVE_API_KEY": "b" * 8, "OLLAMA_API_KEY": "o" * 8}
    s = FakeSession([])
    assert isinstance(build_provider("brave", env, s), BraveSearchProvider)
    assert isinstance(build_provider("ollama", env, s), OllamaSearchProvider)
    assert isinstance(build_provider("duckduckgo", {}, s), DuckDuckGoSearchProvider)
    assert isinstance(
        build_provider("searxng", {}, s, searxng_url="http://127.0.0.1:1"), SearxngSearchProvider
    )
    assert build_provider("none", env, s) is None


@pytest.mark.parametrize(
    ("name", "reason_part"),
    [
        ("brave", "BRAVE_API_KEY"),
        ("ollama", "OLLAMA_API_KEY"),
        ("searxng", "searxng_url"),
        ("none", "none"),
    ],
)
def test_missing_key_returns_none_with_reason(name, reason_part):
    assert build_provider(name, {}, FakeSession([])) is None
    resolved, reason = resolve_provider_name(name, {})
    assert resolved is None
    assert reason_part in reason


def test_unknown_provider_name_raises():
    with pytest.raises(ValueError, match="unknown web provider"):
        build_provider("google", {}, FakeSession([]))


def test_keyed_provider_requires_key():
    with pytest.raises(ValueError, match="needs an API key"):
        BraveSearchProvider("", FakeSession([]))


def test_build_provider_passes_cache_ttl(tmp_path: Path):
    cache = WebCache(tmp_path)
    ft = FakeTime()
    session = FakeSession([FakeResponse(text=_ddg("ddg_html_results.html"))])
    p = build_provider("duckduckgo", {}, session, cache=cache, cache_ttl_hours=1, **_kw(ft))
    p.search("q", max_results=8)
    assert p._cache_ttl == 3600
