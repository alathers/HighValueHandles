"""Unit tests for models.evidence: the web-evidence contract.

Tests cover:
  - to_dict/from_dict round-trips (StructuredPrice, WebEvidence, PageRecord,
    VendorImage, WebResearch), including through JSON
  - sha256 integrity of WebEvidence (mismatch raises, also on load)
  - Decimal prices serialized as strings; JSON-LD prices are retail only
  - allowed values for tiers, page outcomes, research statuses
  - the web-research log: every page with its outcome, cited_for and reason
"""

import hashlib
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from models.evidence import (
    MAX_EVIDENCE_CHARS,
    PAGE_OUTCOMES,
    RESEARCH_STATUSES,
    SEARCH_PROVIDERS,
    SOURCE_CLASSES,
    VENDOR_IMAGE_USAGE,
    PageRecord,
    StructuredPrice,
    VendorImage,
    WebEvidence,
    WebResearch,
    sha256_text,
)

TEXT = "Eagle Rare 10 Year. Bottled at 90 proof. Suggested retail price $39.99."


def _evidence(**overrides) -> WebEvidence:
    fields = dict(
        source_id="W1",
        url="https://www.buffalotracedistillery.com/brands/eagle-rare.html",
        domain="buffalotracedistillery.com",
        tier="official",
        title="Eagle Rare | Buffalo Trace",
        fetched_at="2026-11-23T18:04:05Z",
        http_status=200,
        query="Eagle Rare 10 bourbon",
        from_cache=False,
        truncated=False,
        structured_prices=(StructuredPrice(Decimal("39.99"), "USD"),),
    )
    fields.update(overrides)
    text = fields.pop("text", TEXT)
    return WebEvidence.create(text=text, **fields)


def _pages() -> list[PageRecord]:
    return [
        PageRecord(
            url="https://www.buffalotracedistillery.com/brands/eagle-rare.html",
            title="Eagle Rare | Buffalo Trace",
            domain="buffalotracedistillery.com",
            source_class="official",
            retrieved_at="2026-11-23T18:04:05Z",
            outcome="cited",
            cited_for=["proof", "msrp"],
        ),
        PageRecord(
            url="https://shop.example.com/eagle-rare",
            title="Eagle Rare 750ml",
            domain="example.com",
            source_class="retailer",
            retrieved_at="2026-11-23T18:04:09Z",
            outcome="fetched_not_cited",
            reason="no verifiable quote",
            from_cache=True,
        ),
        PageRecord(
            url="https://blocked.example.org/review",
            title=None,
            domain="example.org",
            source_class="reference",
            retrieved_at=None,
            outcome="skipped",
            reason="robots.txt",
        ),
    ]


class TestStructuredPrice:
    def test_round_trip_and_decimal_as_string(self):
        price = StructuredPrice(Decimal("49.99"), "USD")
        data = price.to_dict()
        assert data == {"amount": "49.99", "currency": "USD", "kind": "retail", "origin": "jsonld:offers.price"}
        assert isinstance(data["amount"], str)
        assert StructuredPrice.from_dict(json.loads(json.dumps(data))) == price

    def test_float_amount_rejected_on_load(self):
        with pytest.raises(ValueError, match="string"):
            StructuredPrice.from_dict({"amount": 49.99, "currency": "USD"})

    @pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
    def test_amount_must_be_positive_and_finite(self, amount):
        with pytest.raises(ValueError):
            StructuredPrice(amount, "USD")

    def test_amount_must_be_decimal(self):
        with pytest.raises(ValueError, match="Decimal"):
            StructuredPrice(49.99, "USD")  # type: ignore[arg-type]

    def test_never_an_msrp(self):
        with pytest.raises(ValueError, match="retail"):
            StructuredPrice(Decimal("39.99"), "USD", kind="msrp")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="retail"):
            StructuredPrice.from_dict({"amount": "39.99", "currency": "USD", "kind": "msrp"})

    def test_currency_required(self):
        with pytest.raises(ValueError, match="currency"):
            StructuredPrice(Decimal("1"), " ")

    def test_bad_amount_string(self):
        with pytest.raises(ValueError, match="not a number"):
            StructuredPrice.from_dict({"amount": "about forty", "currency": "USD"})


class TestWebEvidence:
    def test_round_trip(self):
        evidence = _evidence()
        data = evidence.to_dict()
        assert data["sha256"] == sha256_text(TEXT)
        assert data["structured_prices"] == [
            {"amount": "39.99", "currency": "USD", "kind": "retail", "origin": "jsonld:offers.price"}
        ]
        loaded = WebEvidence.from_dict(json.loads(json.dumps(data)))
        assert loaded == evidence
        assert isinstance(loaded.structured_prices, tuple)

    def test_sha256_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            WebEvidence(
                source_id="W1", url="https://a.example.com/", domain="example.com", tier="reference",
                title=None, text=TEXT, sha256=sha256_text(TEXT + " "), fetched_at="2026-11-23T00:00:00Z",
                http_status=200, query="q", from_cache=False, truncated=False,
            )

    def test_tampered_text_detected_on_load(self):
        data = _evidence().to_dict()
        data["text"] = data["text"].replace("90 proof", "100 proof")
        with pytest.raises(ValueError, match="does not match"):
            WebEvidence.from_dict(data)

    def test_sha256_is_of_utf8_text(self):
        text = "Café Pappy — 107° proof"
        evidence = _evidence(text=text)
        assert evidence.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_text_limit(self):
        _evidence(text="x" * MAX_EVIDENCE_CHARS)
        with pytest.raises(ValueError, match="limit"):
            _evidence(text="x" * (MAX_EVIDENCE_CHARS + 1))

    def test_unknown_tier_rejected(self):
        with pytest.raises(ValueError, match="tier"):
            _evidence(tier="forum")

    def test_url_must_be_http(self):
        with pytest.raises(ValueError, match="http"):
            _evidence(url="file:///etc/passwd")

    def test_list_prices_normalized_to_tuple(self):
        evidence = _evidence(structured_prices=[StructuredPrice(Decimal("1"), "USD")])
        assert isinstance(evidence.structured_prices, tuple)

    def test_missing_key_on_load(self):
        data = _evidence().to_dict()
        del data["url"]
        with pytest.raises(ValueError, match="'url'"):
            WebEvidence.from_dict(data)

    def test_frozen(self):
        with pytest.raises(AttributeError):
            _evidence().text = "changed"  # type: ignore[misc]


class TestPageRecord:
    def test_round_trip_all_outcomes(self):
        for page in _pages():
            data = page.to_dict()
            assert set(data) == {
                "url", "title", "domain", "source_class", "retrieved_at",
                "outcome", "cited_for", "reason", "from_cache",
            }
            assert PageRecord.from_dict(json.loads(json.dumps(data))) == page

    def test_unknown_outcome_rejected(self):
        with pytest.raises(ValueError, match="outcome"):
            PageRecord(url="https://a.example.com", title=None, domain="example.com",
                       source_class="reference", retrieved_at=None, outcome="used")  # type: ignore[arg-type]

    def test_unknown_source_class_rejected(self):
        with pytest.raises(ValueError, match="source_class"):
            PageRecord(url="https://a.example.com", title=None, domain="example.com",
                       source_class="forum", retrieved_at=None, outcome="skipped",  # type: ignore[arg-type]
                       reason="deny-listed")

    def test_unclassified_page_allowed(self):
        page = PageRecord(url="https://a.example.com", title=None, domain="example.com",
                          source_class=None, retrieved_at=None, outcome="skipped", reason="unsafe host")
        assert PageRecord.from_dict(page.to_dict()) == page

    def test_cited_for_requires_cited_outcome(self):
        with pytest.raises(ValueError, match="cited_for"):
            PageRecord(url="https://a.example.com", title=None, domain="example.com",
                       source_class="retailer", retrieved_at="2026-11-23T00:00:00Z",
                       outcome="fetched_not_cited", cited_for=["msrp"])

    def test_cited_for_tuple_normalized_to_list(self):
        page = PageRecord(url="https://a.example.com", title=None, domain="example.com",
                          source_class="official", retrieved_at="2026-11-23T00:00:00Z",
                          outcome="cited", cited_for=("proof",))  # type: ignore[arg-type]
        assert page.cited_for == ["proof"]

    def test_outcome_can_be_updated_with_replace(self):
        page = _pages()[1]
        cited = replace(page, outcome="cited", cited_for=["proof"], reason=None)
        assert cited.outcome == "cited" and page.outcome == "fetched_not_cited"


class TestVendorImage:
    def test_round_trip_and_usage_note(self):
        image = VendorImage(
            cache_path="/tmp/cache/vendor/abc.jpg",
            image_url="https://www.example.com/img/eagle-rare.jpg",
            page_url="https://www.example.com/eagle-rare",
            retrieved_at="2026-11-23T18:04:05Z",
            verdict={"same_bottle": True, "confidence": "high"},
        )
        assert image.usage == VENDOR_IMAGE_USAGE == "reference only - not licensed for publication"
        assert VendorImage.from_dict(json.loads(json.dumps(image.to_dict()))) == image


class TestWebResearch:
    def test_full_round_trip(self):
        research = WebResearch(
            status="ok",
            provider="duckduckgo",
            queries=["Eagle Rare 10 bourbon", "Eagle Rare 10 Buffalo Trace msrp"],
            searched_at="2026-11-23T18:04:00Z",
            pages=_pages(),
            evidence=[_evidence(), _evidence(source_id="W2", tier="retailer", structured_prices=())],
            vendor_image=VendorImage(
                cache_path="c.jpg", image_url="https://e.example.com/i.jpg",
                page_url="https://e.example.com/p", retrieved_at="2026-11-23T18:04:05Z",
                verdict={"same_bottle": True},
            ),
        )
        data = json.loads(json.dumps(research.to_dict()))
        assert [p["outcome"] for p in data["pages"]] == ["cited", "fetched_not_cited", "skipped"]
        assert data["pages"][0]["cited_for"] == ["proof", "msrp"]
        assert data["pages"][1]["reason"] == "no verifiable quote"
        assert data["pages"][2]["reason"] == "robots.txt"
        loaded = WebResearch.from_dict(data)
        assert loaded == research
        assert loaded.pages_consulted == 3
        assert loaded.pages_cited == 1

    def test_skipped_minimal(self):
        research = WebResearch(status="skipped", reason="web lookups disabled")
        data = research.to_dict()
        assert data == {
            "status": "skipped", "reason": "web lookups disabled", "provider": None,
            "queries": [], "searched_at": None, "pages": [], "evidence": [], "vendor_image": None,
        }
        assert WebResearch.from_dict(data) == research

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError, match="status"):
            WebResearch(status="partial")  # type: ignore[arg-type]

    def test_from_dict_requires_status(self):
        with pytest.raises(ValueError, match="'status'"):
            WebResearch.from_dict({"provider": "brave"})

    def test_duplicate_source_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate evidence source_id 'W1'"):
            WebResearch(status="ok", provider="brave", evidence=[_evidence(), _evidence()])

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="provider"):
            WebResearch(status="skipped", reason="x", provider="auto")

    def test_ok_requires_evidence(self):
        with pytest.raises(ValueError, match="requires at least one evidence"):
            WebResearch(status="ok", provider="duckduckgo", pages=_pages()[1:])
        WebResearch(status="no_results", reason="no page matched", provider="duckduckgo")

    def test_tampered_evidence_rejected_on_load(self):
        data = WebResearch(status="ok", provider="brave", evidence=[_evidence()]).to_dict()
        data["evidence"][0]["text"] = "MSRP $19.99"
        with pytest.raises(ValueError, match="does not match"):
            WebResearch.from_dict(data)


def test_vocabularies():
    assert SOURCE_CLASSES == ("official", "retailer", "reference")
    assert PAGE_OUTCOMES == ("cited", "fetched_not_cited", "skipped")
    assert RESEARCH_STATUSES == ("ok", "no_results", "skipped", "failed")
    assert "duckduckgo" in SEARCH_PROVIDERS
    assert "auto" not in SEARCH_PROVIDERS and "none" not in SEARCH_PROVIDERS
