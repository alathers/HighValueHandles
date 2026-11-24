"""Opt-in live web research check (real search + real page fetches).

Skipped unless ``BOURBON_LIVE_TESTS=1`` and ``BOURBON_LIVE_WEB=1``. Uses
Brave/Ollama/SearXNG when configured (``BRAVE_API_KEY``, ``OLLAMA_API_KEY``,
``BOURBON_SEARXNG_URL``), otherwise keyless DuckDuckGo. The cache lives in
``tmp_path``. Asserts invariants only, never values a web page could change.

Run::

    BOURBON_LIVE_TESTS=1 BOURBON_LIVE_WEB=1 \
        uv run --extra dev pytest tests/integration/test_web_live.py
"""

import os
from pathlib import Path
from typing import Any

import pytest
import requests

from image_processor.web_research import WebOptions, WebResearcher
from lib.search_providers import build_provider
from lib.web_cache import WebCache
from lib.web_fetch import USER_AGENT, HttpFetcher
from models.evidence import RESEARCH_STATUSES, sha256_text

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BOURBON_LIVE_TESTS") != "1" or os.environ.get("BOURBON_LIVE_WEB") != "1",
        reason="set BOURBON_LIVE_TESTS=1 BOURBON_LIVE_WEB=1 to run",
    ),
    pytest.mark.allow_web,
    pytest.mark.live,
]


class RecordingSession:
    """Wraps a real ``requests.Session`` and records the User-Agent sent."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self.sent_user_agents: list[str] = []

    @property
    def cookies(self) -> Any:  # noqa: ANN401
        return self._session.cookies

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # noqa: ANN401
        headers = kwargs.get("headers") or {}
        self.sent_user_agents.append(str(headers.get("User-Agent", "")))
        return self._session.request(method, url, **kwargs)


def test_live_research_invariants(tmp_path: Path):
    cache = WebCache(tmp_path / "web")
    search_session = RecordingSession()
    fetch_session = RecordingSession()
    provider = build_provider(
        "auto",
        os.environ,
        search_session,
        searxng_url=os.environ.get("BOURBON_SEARXNG_URL", ""),
        cache=cache,
    )
    assert provider is not None
    fetcher = HttpFetcher(session=fetch_session, cache=cache, max_requests=60)
    researcher = WebResearcher(provider=provider, fetcher=fetcher, options=WebOptions(max_pages=4))

    wr = researcher.research(
        {"name": "Buffalo Trace", "distillery": "Buffalo Trace", "vision_confidence": "high"}
    )

    assert wr.status in RESEARCH_STATUSES
    assert wr.status != "skipped"
    assert wr.queries
    for e in wr.evidence:
        assert e.sha256 == sha256_text(e.text)
        assert e.url.startswith(("http://", "https://"))
    for p in wr.pages:
        assert p.url.startswith(("http://", "https://"))
    assert search_session.sent_user_agents, "no search request was made"
    assert set(search_session.sent_user_agents) == {USER_AGENT}
    if any(p.retrieved_at for p in wr.pages):
        assert set(fetch_session.sent_user_agents) == {USER_AGENT}
    else:
        assert set(fetch_session.sent_user_agents) <= {USER_AGENT}
