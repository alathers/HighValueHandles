"""Tests for image_processor.web_research (FakeProvider, FakeFetcher; no network)."""

import random
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import pytest

from image_processor.web_research import (
    DENY_DOMAINS,
    WebOptions,
    WebResearcher,
    build_excerpt,
    build_queries,
    canonical_url,
    classify,
    name_tokens,
    registrable_domain,
    release_tokens,
)
from lib.html_text import extract
from lib.search_providers import (
    SearchHit,
    SearchProviderAuthError,
    SearchProviderDisabledError,
    SearchProviderError,
)
from lib.web_fetch import FetchedPage, FetchRefused
from models.evidence import WebResearch, sha256_text

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web"
OFFICIAL_URL = "https://www.examplecreekdistillery.com/bourbon/small-batch"
RETAIL_URL = "https://bottleshop.test/example-creek-small-batch"
WIKI_URL = "https://whiskeywiki.test/example-creek"

IDENTITY = {
    "name": "Example Creek Small Batch",
    "distillery": "Example Creek Distillery",
    "specific_release": "2023 Release",
    "batch_number": None,
    "proof": 94,
    "age": "6 years",
    "vision_confidence": "high",
}


def fixture_html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


WIKI_HTML = """<html><head><title>Example Creek - Whiskey Wiki</title></head><body>
<p>Example Creek Small Batch is bottled at 94 proof.</p>
<p>First released in 2010.</p></body></html>"""


def make_page(url: str, html: str, *, final_url: Optional[str] = None) -> FetchedPage:
    return FetchedPage(
        url=url,
        final_url=final_url or url,
        status=200,
        fetched_at="2026-11-23T12:00:00+00:00",
        content_type="text/html",
        html_text=extract(html),
        truncated=False,
        from_cache=False,
    )


class FakeProvider:
    def __init__(self, hits: Any = None, *, name: str = "brave", error: Optional[Exception] = None):  # noqa: ANN401
        self.name = name
        self.hits = hits if hits is not None else []
        self.error = error
        self.queries: list[str] = []
        self._lock = threading.Lock()

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        with self._lock:
            self.queries.append(query)
        if self.error is not None:
            raise self.error
        hits = self.hits(query) if callable(self.hits) else self.hits
        return [
            SearchHit(url=u, title=t, snippet="", rank=i + 1, provider=self.name)
            for i, (u, t) in enumerate(hits[:max_results])
        ]


class FakeFetcher:
    def __init__(
        self, pages: Optional[dict[str, Any]] = None, images: Optional[dict[str, Any]] = None
    ):
        self.pages = pages or {}
        self.images = images or {}
        self.fetched: list[str] = []
        self.image_calls: list[str] = []
        self._lock = threading.Lock()

    def fetch_page(self, url: str) -> FetchedPage:
        with self._lock:
            self.fetched.append(url)
        item = self.pages.get(url)
        if item is None:
            raise FetchRefused("http error", "HTTP 404")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FetchedPage):
            return item
        return make_page(url, item)

    def fetch_image(self, url: str) -> tuple[bytes, str]:
        self.image_calls.append(url)
        item = self.images.get(url)
        if item is None:
            raise FetchRefused("http error", "HTTP 404")
        if isinstance(item, Exception):
            raise item
        return item


STANDARD_HITS = [
    (RETAIL_URL, "Example Creek Small Batch 750ml"),
    (WIKI_URL, "Example Creek - Whiskey Wiki"),
    (OFFICIAL_URL, "Example Creek Small Batch | Official"),
]
STANDARD_PAGES = {
    RETAIL_URL: fixture_html("product_retailer.html"),
    WIKI_URL: WIKI_HTML,
    OFFICIAL_URL: fixture_html("product_official.html"),
}


def researcher(
    provider: Any = None,  # noqa: ANN401
    fetcher: Any = None,  # noqa: ANN401
    **opts: Any,  # noqa: ANN401
) -> WebResearcher:
    extra = {
        k: opts.pop(k)
        for k in ("vendor_verifier", "image_cache_dir", "provider_reason")
        if k in opts
    }
    return WebResearcher(
        provider=provider,
        fetcher=fetcher or FakeFetcher(dict(STANDARD_PAGES)),
        options=WebOptions(**opts),
        **extra,
    )


# ---------------------------------------------------------------------------
# Skip conditions and failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity",
    [
        {**IDENTITY, "name": "Unknown Bourbon"},
        {**IDENTITY, "name": ""},
        {**IDENTITY, "name": None},
        {**IDENTITY, "vision_confidence": "low"},
        {**IDENTITY, "vision_confidence": "LOW"},
        {**IDENTITY, "name": "Kentucky Straight Bourbon Whiskey"},  # nothing distinctive
    ],
)
def test_uncertain_identity_is_skipped(identity: dict):
    provider = FakeProvider(STANDARD_HITS)
    wr = researcher(provider).research(identity)
    assert wr.status == "skipped"
    assert wr.reason == "identity too uncertain to search"
    assert provider.queries == []


def test_no_provider_is_skipped_with_reason():
    wr = researcher(None, provider_reason="web search provider is 'none'").research(IDENTITY)
    assert wr.status == "skipped"
    assert wr.reason == "web search provider is 'none'"
    assert wr.provider is None


def test_provider_error_gives_failed_without_raising():
    provider = FakeProvider(error=SearchProviderError("brave search failed (HTTP 500)"))
    wr = researcher(provider).research(IDENTITY)
    assert wr.status == "failed"
    assert "HTTP 500" in wr.reason
    assert wr.provider == "brave"
    assert wr.queries == provider.queries


def test_unexpected_provider_exception_gives_failed():
    wr = researcher(FakeProvider(error=RuntimeError("bug"))).research(IDENTITY)
    assert wr.status == "failed"


def test_internal_crash_gives_failed():
    class Exploding(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    wr = researcher(FakeProvider(STANDARD_HITS)).research(Exploding())
    assert wr.status == "failed"
    assert "boom" in wr.reason


def test_unknown_provider_name_still_returns_result():
    wr = researcher(FakeProvider(error=SearchProviderError("x"), name="custom")).research(IDENTITY)
    assert wr.status == "failed"
    assert wr.provider is None


def test_one_failing_query_does_not_fail_the_bottle():
    def hits(query: str):
        if "MSRP" in query:
            raise SearchProviderError("duckduckgo throttled")
        return STANDARD_HITS

    class Flaky(FakeProvider):
        def search(self, query: str, *, max_results: int):
            self.queries.append(query)
            hits(query)
            return [SearchHit(u, t, "", i + 1, self.name) for i, (u, t) in enumerate(STANDARD_HITS)]

    wr = researcher(Flaky()).research(IDENTITY)
    assert wr.status == "ok"
    assert len(wr.queries) == 2


@pytest.mark.parametrize(
    ("error", "label"),
    [(SearchProviderAuthError("brave search rejected the credentials (HTTP 401)"), "credentials"),
     (SearchProviderDisabledError("duckduckgo throttled; disabled for this run"), "throttled")],
)  # fmt: skip
def test_auth_or_disabled_error_disables_provider_for_later_bottles(error, label, caplog):
    provider = FakeProvider(error=error)
    r = researcher(provider)
    first = r.research(IDENTITY)
    assert first.status == "failed"
    n = len(provider.queries)
    assert n == 1  # the second query is not sent after the provider is disabled
    second = r.research(IDENTITY)
    assert second.status == "skipped"
    assert second.reason.startswith("search provider disabled: ")
    assert label in second.reason
    assert len(provider.queries) == n


def test_auth_error_logs_one_error_line(caplog: pytest.LogCaptureFixture):
    provider = FakeProvider(error=SearchProviderAuthError("rejected (HTTP 401)"))
    r = researcher(provider)
    for _ in range(3):
        r.research(IDENTITY)
    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    assert len(errors) == 1


def test_no_hits_gives_no_results():
    wr = researcher(FakeProvider([])).research(IDENTITY)
    assert wr.status == "no_results"
    assert wr.reason == "search returned no results"
    assert len(wr.queries) == 2
    assert wr.searched_at


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_queries_are_deterministic_label_only():
    identity = {
        **IDENTITY,
        "batch_number": "Batch C923",
        "specific_release": "2023 Fall Release",
        "bottle_id": "b-1234-secret",
        "image_path": "/Users/someone/photos/IMG_0001.jpg",
        "primary_photo": "/home/x/IMG_2.jpg",
    }
    provider = FakeProvider(STANDARD_HITS)
    researcher(provider).research(
        identity, primary_photo=Path("/Users/someone/photos/IMG_0001.jpg")
    )
    assert provider.queries == [
        '"Example Creek Small Batch" Example Creek Distillery 2023 c923 MSRP',
        '"Example Creek Small Batch" Example Creek Distillery 2023 Fall Release',
    ]
    for q in provider.queries:
        assert "b-1234" not in q
        assert "/" not in q
        assert "IMG" not in q
        assert "94" not in q  # proof is not sent
    assert build_queries(identity) == provider.queries


def test_query_ascii_fold_cap_and_dedupe():
    identity = {"name": 'Élan "Réserve" ' + "x" * 300, "distillery": "Café"}
    qs = build_queries(identity, 5)
    assert len(qs) == 1  # both templates identical without a release
    assert qs[0].isascii()
    assert len(qs[0]) <= 200
    assert qs[0].startswith('"Elan Reserve ')
    assert build_queries(identity, 0) == []


def test_distillery_contained_in_name_is_not_repeated():
    qs = build_queries({"name": "Buffalo Trace", "distillery": "Buffalo Trace"})
    assert qs == ['"Buffalo Trace" MSRP', '"Buffalo Trace"']


def test_release_tokens():
    assert release_tokens(
        {"specific_release": "2021 & 2022 releases", "batch_number": "No. 7"}
    ) == [
        "2021",
        "2022",
        "no. 7",
    ]
    assert release_tokens({"specific_release": "Batch B523"}) == ["b523"]
    assert release_tokens({"batch_number": "Batch #12"}) == ["12"]
    assert release_tokens({}) == []


# ---------------------------------------------------------------------------
# Filtering, ranking, records
# ---------------------------------------------------------------------------


def test_official_ranked_first_and_w_ids_in_rank_order():
    wr = researcher(FakeProvider(STANDARD_HITS)).research(IDENTITY)
    assert wr.status == "ok"
    assert [e.source_id for e in wr.evidence] == ["W1", "W2", "W3"]
    assert [e.url for e in wr.evidence] == [OFFICIAL_URL, RETAIL_URL, WIKI_URL]
    assert [e.tier for e in wr.evidence] == ["official", "retailer", "reference"]
    assert wr.provider == "brave"
    assert [p.url for p in wr.pages] == [OFFICIAL_URL, RETAIL_URL, WIKI_URL]
    for p in wr.pages:
        assert p.outcome == "fetched_not_cited"
        assert p.reason.startswith("sent to fact check as W")
        assert p.retrieved_at == "2026-11-23T12:00:00+00:00"


def test_deny_list_and_custom_deny_domains():
    hits = [
        ("https://www.reddit.com/r/bourbon/example_creek", "reddit"),
        ("https://m.youtube.com/watch?v=1", "video"),
        ("https://www.ebay.com/itm/1", "auction"),
        ("https://blocked.sample.net/x", "custom"),
        *STANDARD_HITS,
    ]
    fetcher = FakeFetcher(dict(STANDARD_PAGES))
    wr = researcher(FakeProvider(hits), fetcher, deny_domains=("sample.net",)).research(IDENTITY)
    denied = [p for p in wr.pages if p.reason == "deny-listed"]
    assert sorted(p.domain for p in denied) == [
        "ebay.com",
        "reddit.com",
        "sample.net",
        "youtube.com",
    ]
    assert all(p.outcome == "skipped" for p in denied)
    assert not any("reddit" in u or "ebay" in u for u in fetcher.fetched)
    assert "reddit.com" in DENY_DOMAINS


def test_allow_list_keeps_only_allowed_plus_official():
    fetcher = FakeFetcher(dict(STANDARD_PAGES))
    wr = researcher(
        FakeProvider(STANDARD_HITS), fetcher, allow_domains=("whiskeywiki.test",)
    ).research(IDENTITY)
    assert sorted(fetcher.fetched) == sorted([OFFICIAL_URL, WIKI_URL])
    skipped = [p for p in wr.pages if p.reason == "not in allow_domains"]
    assert [p.url for p in skipped] == [RETAIL_URL]


def test_duplicate_urls_and_tracking_params_deduplicated():
    hits = [(OFFICIAL_URL + "?utm_source=x#top", "a"), (OFFICIAL_URL, "b")]
    fetcher = FakeFetcher(
        {OFFICIAL_URL + "?utm_source=x#top": fixture_html("product_official.html")}
    )
    wr = researcher(FakeProvider(hits), fetcher).research(IDENTITY)
    assert len(fetcher.fetched) == 1
    assert len(wr.pages) == 1
    assert (
        canonical_url("HTTPS://A.sample.test/p?utm_medium=1&id=3&gclid=z#f")
        == "https://a.sample.test/p?id=3"
    )


def test_name_mismatch_is_skipped_and_recorded():
    other = (
        "<html><head><title>Other Brand Rye</title></head>"
        "<body><p>Other Brand Rye MSRP $30.</p></body></html>"
    )
    hits = [("https://news.sample.test/other", "Other"), *STANDARD_HITS]
    fetcher = FakeFetcher({**STANDARD_PAGES, "https://news.sample.test/other": other})
    wr = researcher(FakeProvider(hits), fetcher).research(IDENTITY)
    rec = next(p for p in wr.pages if p.url == "https://news.sample.test/other")
    assert rec.outcome == "skipped"
    assert rec.reason.startswith("name mismatch")
    assert rec.retrieved_at is not None
    assert "https://news.sample.test/other" not in [e.url for e in wr.evidence]


def test_all_pages_mismatched_gives_no_results_reason():
    other = "<p>Unrelated page about something else entirely.</p>"
    hits = [(f"https://n{i}.sample.test/", "x") for i in range(4)]
    fetcher = FakeFetcher({u: other for u, _ in hits})
    wr = researcher(FakeProvider(hits), fetcher).research(IDENTITY)
    assert wr.status == "no_results"
    assert wr.reason == "4 pages fetched; none matched the bottle name"
    assert len(wr.pages) == 4


def test_fetch_refusals_recorded_with_reasons():
    hits = [
        ("https://a.sample.test/robots", "a"),
        ("https://b.sample.test/unsafe", "b"),
        ("https://c.sample.test/big", "c"),
        ("https://d.sample.test/pdf", "d"),
        ("https://e.sample.test/err", "e"),
    ]
    fetcher = FakeFetcher(
        {
            "https://a.sample.test/robots": FetchRefused(
                "robots", "disallowed by a.sample.test/robots.txt"
            ),
            "https://b.sample.test/unsafe": FetchRefused(
                "unsafe host", "resolves to a non-public address"
            ),
            "https://c.sample.test/big": FetchRefused("too big", "over 5000000 bytes"),
            "https://d.sample.test/pdf": FetchRefused("content-type", "application/pdf"),
            "https://e.sample.test/err": RuntimeError("weird"),
        }
    )
    wr = researcher(FakeProvider(hits), fetcher, max_pages=10).research(IDENTITY)
    reasons = [p.reason for p in wr.pages]
    assert reasons[0].startswith("robots.txt")
    assert reasons[1].startswith("unsafe host")
    assert reasons[2].startswith("too big")
    assert reasons[3].startswith("content-type")
    assert reasons[4].startswith("fetch error")
    assert all(p.outcome == "skipped" and p.retrieved_at is None for p in wr.pages)
    assert wr.status == "no_results"
    assert "no page could be fetched" in wr.reason


def test_max_pages_limits_fetches_and_records_the_rest():
    hits = [(f"https://s{i}.sample.test/p", "x") for i in range(7)]
    fetcher = FakeFetcher({u: WIKI_HTML for u, _ in hits})
    wr = researcher(FakeProvider(hits), fetcher, max_pages=3, max_llm_pages=4).research(IDENTITY)
    assert len(fetcher.fetched) == 3
    limited = [p for p in wr.pages if p.reason and p.reason.startswith("page limit reached")]
    assert len(limited) == 4
    assert len(wr.pages) == 7


def test_at_most_max_llm_pages_evidence():
    hits = [(f"https://s{i}.sample.test/p", "x") for i in range(5)]
    fetcher = FakeFetcher({u: WIKI_HTML for u, _ in hits})
    wr = researcher(FakeProvider(hits), fetcher, max_pages=5, max_llm_pages=2).research(IDENTITY)
    assert [e.source_id for e in wr.evidence] == ["W1", "W2"]
    assert [e.url for e in wr.evidence] == ["https://s0.sample.test/p", "https://s1.sample.test/p"]
    over = [p for p in wr.pages if p.reason and p.reason.startswith("evidence limit reached")]
    assert len(over) == 3
    assert all(p.outcome == "fetched_not_cited" for p in over)


def test_redirect_to_denied_domain_is_skipped():
    hits = [("https://short.sample.test/x", "x")]
    fetcher = FakeFetcher(
        {"https://short.sample.test/x": make_page("https://short.sample.test/x", WIKI_HTML,
                                                  final_url="https://www.reddit.com/r/x")}
    )  # fmt: skip
    wr = researcher(FakeProvider(hits), fetcher).research(IDENTITY)
    assert wr.pages[0].reason == "deny-listed (after redirect)"
    assert wr.status == "no_results"


# ---------------------------------------------------------------------------
# Excerpts and structured prices
# ---------------------------------------------------------------------------


def test_excerpts_are_exact_windows_with_matching_sha256():
    wr = researcher(FakeProvider(STANDARD_HITS)).research(IDENTITY)
    official = wr.evidence[0]
    assert "suggested retail price of $59.99" in official.text
    page_text = extract(fixture_html("product_official.html")).text
    for part in official.text.split("\n\n"):
        assert part in page_text  # every segment is an exact substring of the page text
    for e in wr.evidence:
        assert e.sha256 == sha256_text(e.text)
        assert "..." not in e.text
        assert "…" not in e.text
        assert e.query in wr.queries


def test_excerpt_cap_and_no_ellipses():
    filler = "lorem ipsum dolor sit amet " * 400
    text = (
        filler + " Example Creek Small Batch MSRP is $59.99. " + filler
        + " Example Creek 2023 release proof 94. " + filler
    )  # fmt: skip
    excerpt, cut = build_excerpt(text, name_tokens("Example Creek Small Batch"), 1500)
    assert cut is True
    assert len(excerpt) <= 1500
    assert "MSRP is $59.99" in excerpt
    assert "..." not in excerpt
    assert "…" not in excerpt
    for part in excerpt.split("\n\n"):
        assert part in text


def test_excerpt_uses_option_cap():
    long_html = "<p>" + ("Example Creek Small Batch price $59.99 2023. " * 1000) + "</p>"
    hits = [("https://s.sample.test/p", "x")]
    wr = researcher(FakeProvider(hits), FakeFetcher({"https://s.sample.test/p": long_html}),
                    excerpt_chars=3000).research(IDENTITY)  # fmt: skip
    assert len(wr.evidence[0].text) <= 3000
    assert wr.evidence[0].truncated is True


def test_excerpt_merges_windows_in_page_order():
    text = "AAA Example Creek one. " + "z" * 3000 + " Example Creek two MSRP."
    excerpt, cut = build_excerpt(text, ["example", "creek"], 10_000)
    assert cut is False
    parts = excerpt.split("\n\n")
    assert len(parts) == 2
    assert parts[0].startswith("AAA Example Creek one.")
    assert parts[1].endswith("Example Creek two MSRP.")


def _words(text: str) -> set[str]:
    return set(text.split())


@pytest.mark.parametrize("cap", [10_000, 1200])
def test_excerpt_edges_never_cut_a_word_or_number(cap: int):
    # Window edges (+-800 chars) and the cap land at arbitrary offsets; every
    # whitespace-separated word of the excerpt must be a whole word of the page.
    rng = random.Random(7)
    # Not excerpt keywords, so window edges fall inside this filler.
    vocab = ["1,299.99", "115.2", "86.5", "lorem", "ipsum", "tempus", "12"]
    for _ in range(40):
        before = " ".join(rng.choice(vocab) for _ in range(rng.randint(50, 400)))
        after = " ".join(rng.choice(vocab) for _ in range(rng.randint(50, 400)))
        text = f"{before} Example Creek Small Batch {after}"
        excerpt, _ = build_excerpt(text, ["example", "creek"], cap)
        assert excerpt
        page_words = _words(text)
        for part in excerpt.split("\n\n"):
            assert part in text
            assert part == part.strip()
            assert _words(part) <= page_words, part[:40] + " ... " + part[-40:]


def test_excerpt_window_end_inside_a_number_drops_the_fragment():
    text = "Example Creek " + "x" * 786 + " MSRP $1,299.99 per bottle"
    excerpt, _ = build_excerpt(text, ["example", "creek"], 10_000)
    assert "$1,2" not in excerpt or "$1,299.99" in excerpt
    assert not excerpt.endswith(("$1", "$1,", "$1,2", "$1,29"))


def test_redirect_to_non_allowed_domain_is_skipped():
    hits = [("https://whiskeywiki.test/x", "x")]
    fetcher = FakeFetcher(
        {"https://whiskeywiki.test/x": make_page("https://whiskeywiki.test/x", WIKI_HTML,
                                                 final_url="https://elsewhere.test/x")}
    )  # fmt: skip
    wr = researcher(
        FakeProvider(hits), fetcher, allow_domains=("whiskeywiki.test",)
    ).research(IDENTITY)
    assert wr.pages[0].reason == "not in allow_domains (after redirect)"
    assert wr.pages[0].outcome == "skipped"
    assert wr.status == "no_results"
    assert not wr.evidence


def test_name_token_match_tolerates_apostrophes_and_accents():
    text = "Try Blanton's Single Barrel and the Rhône cask today."
    excerpt, _ = build_excerpt(text, name_tokens("Blantons Rhone"), 10_000)
    assert excerpt == text


def test_structured_prices_only_on_retailer_pages_and_matching_product():
    wr = researcher(FakeProvider(STANDARD_HITS)).research(IDENTITY)
    by_tier = {e.tier: e for e in wr.evidence}
    assert [(p.amount, p.currency, p.kind) for p in by_tier["retailer"].structured_prices] == [
        (Decimal("59.99"), "USD", "retail")
    ]  # "Other Brand Rye" offer excluded
    assert by_tier["official"].structured_prices == ()
    assert by_tier["reference"].structured_prices == ()


def test_official_page_with_offers_is_not_treated_as_retailer():
    html = fixture_html("product_retailer.html")
    page = extract(html)
    assert classify("examplecreekdistillery.com", IDENTITY, page) == "official"
    assert classify("bottleshop.test", IDENTITY, page) == "retailer"
    assert classify("whiskeywiki.test", IDENTITY, extract(WIKI_HTML)) == "reference"


def test_registrable_domain():
    assert registrable_domain("www.shop.sample.test") == "sample.test"
    assert registrable_domain("shop.example.co.uk") == "example.co.uk"
    assert registrable_domain("sample.test") == "sample.test"


def test_official_needs_distinctive_brand_token():
    # Generic words never make a domain official.
    assert (
        classify(
            "bourbonreserve.com", {"name": "Old Reserve Bourbon", "distillery": ""}, extract("")
        )
        == "reference"
    )


def test_result_round_trips():
    wr = researcher(FakeProvider(STANDARD_HITS)).research(IDENTITY)
    assert WebResearch.from_dict(wr.to_dict()).to_dict() == wr.to_dict()


# ---------------------------------------------------------------------------
# research_many
# ---------------------------------------------------------------------------


def test_research_many_preserves_input_order_with_three_workers():
    names = [f"Example Creek Batch{i:02d}" for i in range(9)]

    def hits(query: str):
        time.sleep(random.uniform(0, 0.01))
        token = query.split('"')[1].split()[-1]
        return [(f"https://{token.lower()}.sample.test/p", token)]

    pages = {
        f"https://batch{i:02d}.sample.test/p": f"<p>Example Creek Batch{i:02d} price $50</p>"
        for i in range(9)
    }
    r = researcher(FakeProvider(hits), FakeFetcher(pages), max_queries=1)
    items = [({"name": n, "vision_confidence": "high"}, None) for n in names]
    results = r.research_many(items, workers=3)
    assert [res.evidence[0].url for res in results] == [
        f"https://batch{i:02d}.sample.test/p" for i in range(9)
    ]


def test_research_many_never_raises():
    r = researcher(FakeProvider(STANDARD_HITS))
    results = r.research_many([(IDENTITY, None), "garbage", ({"name": None}, None)], workers=2)  # type: ignore[list-item]
    assert [x.status for x in results] == ["ok", "failed", "skipped"]
    assert r.research_many([], workers=3) == []


# ---------------------------------------------------------------------------
# Vendor image
# ---------------------------------------------------------------------------

OG_IMAGE = "https://www.examplecreekdistillery.com/media/example-creek-small-batch.jpg"
LD_IMAGE = "https://www.examplecreekdistillery.com/media/bottle-front.jpg"


class FakeVerifier:
    def __init__(self, verdicts: list[Any]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, primary: Path, candidate: Path) -> dict:
        self.calls.append((primary, candidate))
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return v


HIGH = {
    "same_product": True,
    "same_release": "yes",
    "label_text_seen": "Example Creek",
    "confidence": "high",
}


def vendor_researcher(verifier: FakeVerifier, tmp_path: Path, images: Optional[dict] = None):
    fetcher = FakeFetcher(
        dict(STANDARD_PAGES),
        images
        if images is not None
        else {OG_IMAGE: (b"jpeg-1", "jpg"), LD_IMAGE: (b"jpeg-2", "jpg")},
    )
    r = researcher(
        FakeProvider(STANDARD_HITS),
        fetcher,
        vendor_images=True,
        vendor_verifier=verifier,
        image_cache_dir=tmp_path / "images",
    )
    return r, fetcher


def test_vendor_image_accepted_on_high_same_product(tmp_path: Path):
    verifier = FakeVerifier([HIGH])
    r, fetcher = vendor_researcher(verifier, tmp_path)
    photo = tmp_path / "IMG_1.jpg"
    wr = r.research(IDENTITY, primary_photo=photo)
    assert wr.status == "ok"
    assert wr.vendor_image is not None
    assert wr.vendor_image.image_url == OG_IMAGE  # relative og:image resolved against the page
    assert wr.vendor_image.page_url == OFFICIAL_URL
    assert Path(wr.vendor_image.cache_path).read_bytes() == b"jpeg-1"
    assert Path(wr.vendor_image.cache_path).parent == tmp_path / "images"
    assert wr.vendor_image.verdict["confidence"] == "high"
    assert "not licensed" in wr.vendor_image.usage
    assert verifier.calls[0][0] == photo


@pytest.mark.parametrize(
    "bad",
    [
        {**HIGH, "confidence": "medium"},
        {**HIGH, "same_release": "no"},
        {**HIGH, "same_product": False},
        {**HIGH, "same_product": "true"},
    ],
)
def test_vendor_image_rejected_verdicts_and_two_tries_max(tmp_path: Path, bad: dict):
    verifier = FakeVerifier([bad, bad, HIGH])
    r, fetcher = vendor_researcher(verifier, tmp_path)
    wr = r.research(IDENTITY, primary_photo=tmp_path / "p.jpg")
    assert wr.status == "ok"
    assert wr.vendor_image is None
    assert len(verifier.calls) == 2
    assert len(fetcher.image_calls) == 2


def test_vendor_image_second_candidate_accepted(tmp_path: Path):
    verifier = FakeVerifier([{**HIGH, "confidence": "medium"}, HIGH])
    r, _ = vendor_researcher(verifier, tmp_path)
    wr = r.research(IDENTITY, primary_photo=tmp_path / "p.jpg")
    assert wr.vendor_image.image_url == LD_IMAGE


def test_vendor_image_download_failure_counts_as_a_try(tmp_path: Path):
    verifier = FakeVerifier([HIGH])
    r, fetcher = vendor_researcher(verifier, tmp_path, images={LD_IMAGE: (b"x", "jpg")})
    wr = r.research(IDENTITY, primary_photo=tmp_path / "p.jpg")
    assert fetcher.image_calls[:2] == [OG_IMAGE, LD_IMAGE]
    assert wr.vendor_image is not None
    assert wr.vendor_image.image_url == LD_IMAGE


def test_vendor_verifier_exception_gives_no_image_but_ok(tmp_path: Path):
    verifier = FakeVerifier([RuntimeError("vision model unavailable")])
    r, _ = vendor_researcher(verifier, tmp_path)
    wr = r.research(IDENTITY, primary_photo=tmp_path / "p.jpg")
    assert wr.status == "ok"
    assert wr.vendor_image is None
    assert len(wr.evidence) == 3


def test_vendor_image_needs_option_verifier_and_photo(tmp_path: Path):
    verifier = FakeVerifier([HIGH, HIGH, HIGH])
    r, fetcher = vendor_researcher(verifier, tmp_path)
    assert r.research(IDENTITY).vendor_image is None  # no primary photo
    off = researcher(
        FakeProvider(STANDARD_HITS),
        FakeFetcher(dict(STANDARD_PAGES)),
        vendor_verifier=verifier,
        image_cache_dir=tmp_path,
    )
    assert off.research(IDENTITY, primary_photo=tmp_path / "p.jpg").vendor_image is None
    assert verifier.calls == []


def test_vendor_image_ignores_reference_pages(tmp_path: Path):
    wiki_with_image = WIKI_HTML.replace(
        "<head>", '<head><meta property="og:image" content="https://whiskeywiki.test/i.jpg">'
    )
    fetcher = FakeFetcher(
        {WIKI_URL: wiki_with_image}, {"https://whiskeywiki.test/i.jpg": (b"x", "jpg")}
    )
    verifier = FakeVerifier([HIGH])
    r = researcher(
        FakeProvider([(WIKI_URL, "wiki")]),
        fetcher,
        vendor_images=True,
        vendor_verifier=verifier,
        image_cache_dir=tmp_path,
    )
    wr = r.research(IDENTITY, primary_photo=tmp_path / "p.jpg")
    assert wr.status == "ok"
    assert wr.vendor_image is None
    assert fetcher.image_calls == []
