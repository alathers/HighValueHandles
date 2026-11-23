"""Tests of image_processor.enrichment: web research, model swaps and the fact check per batch.

Fakes only: a scripted researcher, a scripted text LLM behind the real
FactChecker, and an unload recorder. Nothing here talks to a model server or
the web.

The builders at the top (``checked_bottle``, ``web_research``, ``FakeTextLLM``)
are also used by the preview and CLI tests.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from image_processor.enrichment import (
    Enricher,
    apply_fact_values,
    approve_facts,
    format_flat_value,
    refresh_facts,
    update_page_outcomes,
    user_facts_from_corrections,
)
from image_processor.fact_checker import FactChecker
from lib.ollama_client import ChatResult
from models.bottle import Bottle, ExtractedMetadata
from models.evidence import PageRecord, VendorImage, WebEvidence, WebResearch
from models.facts import FactInput, FactSheet

TEXT_MODEL = "gpt-oss:120b"
VISION_MODEL = "qwen3.6:35b-a3b"

W1_TEXT = (
    "Example Creek Single Barrel is bottled at 101 proof. Aged 8 years | small `batch`. "
    "Distilled by Example Creek Distillery. Suggested retail price (MSRP) of $59.99."
)
W2_TEXT = "Example Creek Single Barrel. Our price $69.99. Add to cart."
W1_URL = "https://examplecreek.test/single-barrel"
W2_URL = "https://shop.test/ec-sb"

#: EXTRACT reply: proof disagrees with the label (100), age/distillery/MSRP verified on W1.
EXTRACT_REPLY = json.dumps({
    "page_match": [{"source_id": "W1", "verdict": "exact", "quote": "Example Creek Single Barrel is bottled"}],
    "claims": [
        {"field": "proof", "value": "101", "source_id": "W1", "quote": "bottled at 101 proof"},
        {"field": "age", "value": "8 years", "source_id": "W1", "quote": "Aged 8 years | small `batch`"},
        {"field": "distillery", "value": "Example Creek Distillery", "source_id": "W1",
         "quote": "Distilled by Example Creek Distillery"},
        {"field": "mash_bill", "value": "wheated", "source_id": "W2", "quote": "invented quote, not on the page"},
    ],
    "prices": [{"kind": "msrp", "amount": 59.99, "currency": "USD", "source_id": "W1",
                "quote": "Suggested retail price (MSRP) of $59.99"}],
})
#: EXTRACT reply with nothing found (MSRP must stay Unknown).
EXTRACT_EMPTY = json.dumps({"page_match": [], "claims": [], "prices": []})
#: A valid COMPOSE reply for name + distillery + age (fields 1, 2 and 3 once proof is a conflict).
COMPOSE_REPLY = json.dumps({"sentences": [
    {"text": "Example Creek Single Barrel is distilled by Example Creek Distillery.", "citations": [1, 2],
     "fields": ["name", "distillery"]},
    {"text": "It is aged 8 years.", "citations": [3], "fields": ["age"]},
]})


class FakeTextLLM:
    """Scripted ``chat_text``; records each call in ``events`` (shared log) and ``calls``."""

    def __init__(self, *replies, events=None):
        self.replies = list(replies)
        self.calls = []
        self.events = events if events is not None else []

    def chat_text(self, prompt, model=None, *, system=None, format=None, think=None, options=None,
                  keep_alive="5m", read_timeout=None):
        self.calls.append({"model": model, "think": think, "keep_alive": keep_alive, "system": system})
        self.events.append(f"text:{model}")
        if not self.replies:
            raise AssertionError("unexpected extra text call")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ChatResult(content=reply, thinking="", model=model or "", done_reason="stop")


def evidence_pages():
    w1 = WebEvidence.create(
        text=W1_TEXT, source_id="W1", url=W1_URL, domain="examplecreek.test", tier="official",
        title="Single Barrel | Example Creek", fetched_at="2026-11-23T10:00:00+00:00", http_status=200,
        query="example creek single barrel", from_cache=False, truncated=False,
    )
    w2 = WebEvidence.create(
        text=W2_TEXT, source_id="W2", url=W2_URL, domain="shop.test", tier="retailer",
        title="Example Creek SB", fetched_at="2026-11-23T10:00:05+00:00", http_status=200,
        query="example creek single barrel", from_cache=False, truncated=False,
    )
    return w1, w2


def web_research(vendor_image=None):
    """An ``ok`` research record: two evidence pages and one deny-listed hit."""
    w1, w2 = evidence_pages()
    return WebResearch(
        status="ok", provider="duckduckgo", queries=["example creek single barrel"],
        searched_at="2026-11-23T10:00:00+00:00",
        pages=[
            PageRecord(url=W1_URL, title=w1.title, domain=w1.domain, source_class="official",
                       retrieved_at=w1.fetched_at, outcome="fetched_not_cited", reason="sent to fact check as W1"),
            PageRecord(url=W2_URL, title=w2.title, domain=w2.domain, source_class="retailer",
                       retrieved_at=w2.fetched_at, outcome="fetched_not_cited", reason="sent to fact check as W2"),
            PageRecord(url="https://forum.test/thread", title="forum", domain="forum.test", source_class=None,
                       retrieved_at=None, outcome="skipped", reason="deny-listed"),
        ],
        evidence=[w1, w2],
        vendor_image=vendor_image,
    )


LABEL = {"name": "Example Creek Single Barrel", "distillery": "Example Creek Distillery", "proof": "100 Proof",
         "label_price": "64.99"}


def checked_bottle(*, compose=COMPOSE_REPLY, label=None, research=None):
    """A bottle dict as the pipeline leaves it: facts, web research with outcomes, flat values.

    The fact sheet has a proof conflict (label 100 vs W1 101), an MSRP
    verified on the official page W1, a label price (retail only) and an
    Unknown release year. Returns ``(bottle_dict, metadata, sheet)``.
    """
    label = dict(label or LABEL)
    research = research or web_research()
    inp = FactInput(
        bottle_id="IMG_1", label=label, vision_confidence="high", images=["IMG_1.jpg"], ocr_by_image={},
        web=tuple(research.evidence),
    )
    sheet = FactChecker(FakeTextLLM(EXTRACT_REPLY, compose), model=TEXT_MODEL, max_attempts=1).check(inp)
    sheet.verification.duration_s = 12.5
    research = update_page_outcomes(research, sheet)
    bottle_dict = {
        "bottle_id": "IMG_1", "name": label["name"], "label_price": label.get("label_price"),
        "images": ["/p/IMG_1.jpg"], "facts": sheet.to_dict(), "web_research": research.to_dict(),
        "confidence": "HIGH", "quantity": 1,
    }
    apply_fact_values(bottle_dict, sheet)
    metadata = ExtractedMetadata(source_images=[Path("/p/IMG_1.jpg")], vision_confidence="high", **{
        k: v for k, v in label.items() if k in ("name", "distillery", "proof", "label_price")
    })
    return bottle_dict, metadata, sheet


# ---------------------------------------------------------------------------
# Enricher fixtures
# ---------------------------------------------------------------------------


class FakeResearcher:
    """``research_many`` returns scripted results (or raises); logs 'research' in the shared events."""

    def __init__(self, results=None, *, events=None, error=None, provider_name="duckduckgo"):
        self.results = results
        self.events = events if events is not None else []
        self.error = error
        self.items = None
        self.workers = None
        self.provider = type("P", (), {"name": provider_name})()

    def research_many(self, items, *, workers=1):
        self.events.append("research")
        self.items = list(items)
        self.workers = workers
        if self.error is not None:
            raise self.error
        return list(self.results) if self.results is not None else [web_research() for _ in items]


def _metadata(name="Example Creek Single Barrel", **kw):
    fields = {"distillery": "Example Creek Distillery", "proof": "100 Proof", "label_price": "64.99"}
    fields.update(kw)
    return ExtractedMetadata(
        name=name, source_images=[Path("/photos/IMG_1.jpg")], vision_confidence="high",
        ocr_by_image={"IMG_1.jpg": "EXAMPLE CREEK\nSINGLE BARREL"}, **fields,
    )


def _bottle_dict(name="Example Creek Single Barrel"):
    return {
        "bottle_id": "IMG_1", "name": name, "distillery": "Example Creek Distillery", "proof": "100 Proof",
        "label_price": "64.99", "msrp": None, "images": ["/photos/IMG_1.jpg"],
    }


def _enricher(researcher, llm, events, *, keep=False, text=True):
    return Enricher(
        researcher=researcher,
        fact_checker=FactChecker(llm if text else None, model=TEXT_MODEL, max_attempts=1),
        vision_model=VISION_MODEL,
        text_model=TEXT_MODEL if text else None,
        keep_models_loaded=keep,
        workers=3,
        unload=lambda name: events.append(f"unload:{name}"),
    )


# ---------------------------------------------------------------------------
# Order of the phases and model swaps
# ---------------------------------------------------------------------------


class TestPhaseOrder:
    def test_research_then_unload_vision_then_text_then_unload_text(self):
        events = []
        researcher = FakeResearcher(events=events)
        llm = FakeTextLLM(EXTRACT_REPLY, COMPOSE_REPLY, EXTRACT_REPLY, COMPOSE_REPLY, events=events)
        dicts = [_bottle_dict(), {**_bottle_dict(), "bottle_id": "IMG_2", "images": ["/photos/IMG_2.jpg"]}]
        _enricher(researcher, llm, events).enrich_batch(dicts, [_metadata(), _metadata()], run_id="r1")

        assert events == [
            "research",
            f"unload:{VISION_MODEL}",
            f"text:{TEXT_MODEL}", f"text:{TEXT_MODEL}",
            f"text:{TEXT_MODEL}", f"text:{TEXT_MODEL}",
            f"unload:{TEXT_MODEL}",
        ]
        assert researcher.workers == 3
        assert all(d["run_id"] == "r1" for d in dicts)
        assert all(d["facts"]["verification"]["status"] == "verified" for d in dicts)

    def test_keep_models_loaded_means_no_unloads(self):
        events = []
        llm = FakeTextLLM(EXTRACT_REPLY, COMPOSE_REPLY, events=events)
        _enricher(FakeResearcher(events=events), llm, events, keep=True).enrich_batch(
            [_bottle_dict()], [_metadata()], run_id="r1"
        )
        assert not [e for e in events if e.startswith("unload")]
        assert llm.calls  # the text model still ran

    def test_text_inactive_no_unload_and_deterministic_sheets(self):
        events = []
        d = _bottle_dict()
        _enricher(FakeResearcher(events=events), None, events, text=False).enrich_batch([d], [_metadata()], run_id="r")
        assert events == ["research"]
        sheet = FactSheet.from_dict(d["facts"])
        assert sheet.verification.status == "skipped"
        # Web text never fills a field without the text model.
        assert sheet.fields["age"].status == "unknown"
        assert sheet.fields["msrp"].value is None
        assert d["msrp"] is None
        pages = d["web_research"]["pages"]
        assert [p["outcome"] for p in pages] == ["fetched_not_cited", "fetched_not_cited", "skipped"]
        assert "the fact check did not run" in pages[0]["reason"]

    def test_unload_failure_is_ignored(self):
        def boom(name):
            raise RuntimeError("server gone")

        enricher = Enricher(
            researcher=None, fact_checker=FactChecker(FakeTextLLM(EXTRACT_EMPTY, COMPOSE_REPLY), model=TEXT_MODEL),
            vision_model=VISION_MODEL, text_model=TEXT_MODEL, keep_models_loaded=False, unload=boom,
        )
        d = _bottle_dict()
        enricher.enrich_batch([d], [_metadata()], run_id="r")  # does not raise
        assert d["facts"]["verification"]["status"] in ("verified", "partial")
        assert d["facts"]["fields"]["name"]["status"] == "supported"

    def test_empty_batch_does_nothing(self):
        events = []
        _enricher(FakeResearcher(events=events), FakeTextLLM(events=events), events).enrich_batch([], [], run_id="r")
        assert events == []


# ---------------------------------------------------------------------------
# Failures never fail the bottle
# ---------------------------------------------------------------------------


class TestFailures:
    def test_researcher_exception_gives_failed_web_research_and_facts(self):
        events = []
        researcher = FakeResearcher(events=events, error=RuntimeError("socket exploded"))
        d = _bottle_dict()
        _enricher(researcher, FakeTextLLM(EXTRACT_EMPTY, COMPOSE_REPLY, events=events), events).enrich_batch(
            [d], [_metadata()], run_id="r"
        )
        assert d["web_research"]["status"] == "failed"
        assert "socket exploded" in d["web_research"]["reason"]
        assert d["facts"]["fields"]["name"]["status"] == "supported"

    def test_no_researcher_records_skipped_with_reason(self):
        enricher = Enricher(
            researcher=None, fact_checker=FactChecker(None, model=TEXT_MODEL), vision_model=VISION_MODEL,
            text_model=None, keep_models_loaded=False, web_skip_reason="web provider 'none' configured",
        )
        d = _bottle_dict()
        enricher.enrich_batch([d], [_metadata()], run_id="r")
        assert d["web_research"] == WebResearch(status="skipped", reason="web provider 'none' configured").to_dict()

    def test_default_skip_reason(self):
        enricher = Enricher(researcher=None, fact_checker=FactChecker(None, model=TEXT_MODEL),
                            vision_model=VISION_MODEL, text_model=None, keep_models_loaded=False)
        d = _bottle_dict()
        enricher.enrich_batch([d], [_metadata()], run_id="r")
        assert d["web_research"]["reason"] == "web lookups disabled"

    def test_checker_exception_gives_failed_deterministic_sheet(self):
        class Exploding(FactChecker):
            def check(self, inp):
                raise RuntimeError("checker bug")

        enricher = Enricher(
            researcher=FakeResearcher(), fact_checker=Exploding(FakeTextLLM(), model=TEXT_MODEL),
            vision_model=VISION_MODEL, text_model=TEXT_MODEL, keep_models_loaded=True,
        )
        d = _bottle_dict()
        enricher.enrich_batch([d], [_metadata()], run_id="r")
        v = d["facts"]["verification"]
        assert v["status"] == "failed"
        assert "checker bug" in v["reason"]
        assert d["facts"]["publish"]["ready"] is False
        assert d["name"] == "Example Creek Single Barrel"

    def test_text_model_timeout_is_recorded_not_raised(self):
        from lib.exceptions import TextTimeoutError

        events = []
        d = _bottle_dict()
        _enricher(FakeResearcher(events=events), FakeTextLLM(TextTimeoutError("slow"), events=events), events
                  ).enrich_batch([d], [_metadata()], run_id="r")
        assert d["facts"]["verification"]["status"] == "failed"
        assert any(b.startswith("text verification failed") for b in d["facts"]["publish"]["blockers"])


# ---------------------------------------------------------------------------
# Flat values and provenance
# ---------------------------------------------------------------------------


class TestFlatValues:
    def _run(self, extract=EXTRACT_REPLY, metadata=None, dict_=None, research=None):
        events = []
        d = dict_ or _bottle_dict()
        researcher = FakeResearcher([research or web_research()], events=events)
        _enricher(researcher, FakeTextLLM(extract, COMPOSE_REPLY, events=events), events).enrich_batch(
            [d], [metadata or _metadata()], run_id="r"
        )
        return d, researcher

    def test_flat_values_derived_from_facts(self):
        d, _ = self._run()
        assert d["distillery"] == "Example Creek Distillery"
        assert d["age"] == "8 years"
        assert d["proof"] is None  # label 100 vs W1 101: a conflict, no value
        assert d["msrp"] == "59.99"  # verified MSRP statement on the official page
        assert d["facts"]["fields"]["proof"]["status"] == "conflict"
        assert Bottle.from_analysis(d).msrp == Decimal("59.99")

    def test_msrp_none_unless_facts_support_it(self):
        d, _ = self._run(extract=EXTRACT_EMPTY)
        assert d["msrp"] is None
        assert d["facts"]["fields"]["msrp"]["status"] == "unknown"
        assert Bottle.from_analysis(d).msrp is None

    def test_label_price_never_reaches_msrp(self):
        d, _ = self._run(extract=EXTRACT_EMPTY)
        sheet = FactSheet.from_dict(d["facts"])
        assert [(p.kind, str(p.amount)) for p in sheet.prices] == [("retail", "64.99")]
        assert d["msrp"] is None
        assert d["label_price"] == "64.99"
        bottle = Bottle.from_analysis(d)
        assert bottle.msrp is None and bottle.label_price == Decimal("64.99")
        assert bottle.quantity.price is None and bottle.total_value is None

    def test_name_falls_back_to_label_reading_for_folder_naming(self):
        d, _ = self._run(
            extract=EXTRACT_EMPTY, metadata=_metadata(name=None), dict_=_bottle_dict(name="Example Creek Single Barrel")
        )
        assert d["facts"]["fields"]["name"]["status"] == "unknown"
        assert d["name"] == "Example Creek Single Barrel"

    def test_page_outcomes_record_cited_fields_and_reasons(self):
        d, _ = self._run()
        pages = d["web_research"]["pages"]
        assert pages[0]["outcome"] == "cited"
        assert pages[0]["cited_for"] == ["distillery", "age", "msrp"]
        assert pages[0]["reason"] is None
        assert pages[1]["outcome"] == "fetched_not_cited"
        assert pages[1]["reason"].startswith("W2: no verifiable quote accepted (quote_not_found")
        assert pages[2] == {**pages[2], "outcome": "skipped", "reason": "deny-listed"}
        wr = WebResearch.from_dict(d["web_research"])
        assert (wr.pages_consulted, wr.pages_cited) == (3, 1)

    def test_page_outcomes_are_idempotent(self):
        d, _ = self._run()
        sheet = FactSheet.from_dict(d["facts"])
        once = WebResearch.from_dict(d["web_research"])
        assert update_page_outcomes(once, sheet).to_dict() == once.to_dict()

    def test_fact_input_uses_label_file_names_ocr_and_group_warnings(self):
        captured = {}

        class Spy(FactChecker):
            def check(self, inp):
                captured["inp"] = inp
                return super().check(inp)

        metadata = _metadata(warnings=["Split from a group of 3 photos (a, b, c) because ...; review the split",
                                       "Possibly the same bottle as IMG_9 (adjacent photos, identical label)"])
        d = _bottle_dict()
        enricher = Enricher(researcher=None, fact_checker=Spy(None, model=TEXT_MODEL), vision_model=VISION_MODEL,
                            text_model=None, keep_models_loaded=False)
        enricher.enrich_batch([d], [metadata], run_id="r")
        inp = captured["inp"]
        assert inp.images == ["IMG_1.jpg"]
        assert inp.label["label_price"] == "64.99"
        assert inp.label["proof"] == "100 Proof"
        assert inp.ocr_by_image == {"IMG_1.jpg": "EXAMPLE CREEK\nSINGLE BARREL"}
        assert inp.vision_confidence == "high"
        assert list(inp.group_warnings) == [metadata.warnings[0]]
        assert any(b.startswith("group warning: Split") for b in d["facts"]["publish"]["blockers"])

    def test_hint_corrections_become_user_facts_and_identity(self):
        d = _bottle_dict()
        d["proof"] = "101 Proof"  # already applied by apply_hint_overrides
        d["corrections"] = [{
            "bottle_id": "IMG_1", "field_name": "proof", "original_value": "100 Proof", "corrected_value": "101 Proof",
            "source": "hint:hints.toml#IMG_1: Owner checked the neck tag", "corrected_at": "2026-11-23T00:00:00+00:00",
        }]
        researcher = FakeResearcher([web_research()])
        events = []
        _enricher(researcher, FakeTextLLM(EXTRACT_REPLY, COMPOSE_REPLY, events=events), events).enrich_batch(
            [d], [_metadata()], run_id="r"
        )
        identity, photo = researcher.items[0]
        assert identity["proof"] == "101 Proof"  # hint value in the search identity
        assert identity["vision_confidence"] == "high"
        assert photo == Path("/photos/IMG_1.jpg")
        proof = d["facts"]["fields"]["proof"]
        assert proof["status"] == "user"
        assert proof["value"] == 101.0
        assert d["proof"] == "101 Proof"
        user_sources = [s for s in d["facts"]["sources"] if s["kind"] == "user"]
        assert user_sources[0]["ref"] == "hint:hints.toml#IMG_1: Owner checked the neck tag"

    def test_vendor_image_kept(self):
        vendor = VendorImage(cache_path="/cache/x.jpg", image_url="https://examplecreek.test/i.jpg",
                             page_url=W1_URL, retrieved_at="2026-11-23T00:00:00+00:00",
                             verdict={"same_product": True, "confidence": "high"})
        d, _ = self._run(research=web_research(vendor_image=vendor))
        assert d["vendor_image"]["cache_path"] == "/cache/x.jpg"
        assert d["vendor_image"]["usage"] == "reference only - not licensed for publication"


class TestHelpers:
    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [("proof", 90.0, "90 Proof"), ("proof", 125.9, "125.9 Proof"), ("abv", 45.0, "45%"),
         ("msrp", Decimal("59.99"), "59.99"), ("release_year", 2024, "2024"), ("age", "8 years", "8 years"),
         ("name", None, None)],
    )
    def test_format_flat_value(self, field, value, expected):
        assert format_flat_value(field, value) == expected

    def test_user_facts_from_corrections(self, caplog):
        facts = user_facts_from_corrections([
            {"field_name": "proof", "corrected_value": "90", "source": "interactive"},
            {"field_name": "age", "corrected_value": "unknown", "source": "interactive"},  # null-like
            {"field_name": "msrp", "corrected_value": "45", "source": "feedback:c.json"},
            {"field_name": "name", "corrected_value": "X", "source": "something else"},
            {"field_name": "not_a_field", "corrected_value": "X", "source": "interactive"},
        ])
        assert [(f.field, f.value, f.ref) for f in facts] == [
            ("proof", "90", "interactive"), ("msrp", "45", "feedback:c.json"),
        ]
        assert "not a usable value" in caplog.text

    def test_refresh_facts_marks_copy_stale_when_a_cited_field_changes(self):
        d, _metadata, sheet = checked_bottle()
        assert sheet.copy is not None and not sheet.copy.stale
        d["corrections"] = [{"bottle_id": "IMG_1", "field_name": "age", "original_value": "8 years",
                             "corrected_value": "9 years", "source": "interactive"}]
        new = refresh_facts(d)
        assert new.copy.stale is True
        assert "age" in new.copy.reason
        assert any(b.startswith("description is stale") for b in new.publish["blockers"])
        assert d["age"] == "9 years"

    def test_refresh_facts_withdraws_an_approval_when_values_change(self):
        d, _metadata, sheet = checked_bottle()
        d["facts"] = approve_facts(sheet, "feedback").to_dict()
        d["corrections"] = [{"bottle_id": "IMG_1", "field_name": "proof", "original_value": "",
                             "corrected_value": "101", "source": "interactive"}]
        new = refresh_facts(d)
        assert new.review["approved"] is False
        assert "not approved by a reviewer" in new.publish["blockers"]

    def test_approve_records_reviewer_and_recomputes_gate(self):
        _d, _m, sheet = checked_bottle()
        approved = approve_facts(sheet, "interactive", at="2026-11-23T00:00:00+00:00")
        assert approved.review == {"approved": True, "by": "interactive", "at": "2026-11-23T00:00:00+00:00"}
        assert "not approved by a reviewer" not in approved.publish["blockers"]
        assert approved.publish["ready"] is False  # the proof conflict still blocks

    def test_settling_a_conflict_renumbers_the_description_markers(self):
        """Proof joins the numbered facts, so 'age' moves from [3] to [4]; the marker must follow it."""
        d, metadata, sheet = checked_bottle()
        assert sheet.copy.sentences[1]["citations"] == [3]  # compose time: 1 name, 2 distillery, 3 age
        d["corrections"] = [{"bottle_id": "IMG_1", "field_name": "proof", "original_value": "",
                             "corrected_value": "101", "source": "interactive"}]
        new = refresh_facts(d)
        assert new.copy.stale is False  # no cited value changed
        assert new.copy.sentences[0]["citations"] == [1, 2]
        assert new.copy.sentences[1]["citations"] == [4]
        assert new.copy.description_md.endswith("It is aged 8 years. [4]")
        assert d["facts"]["copy"]["sentences"][1]["citations"] == [4]
        md = Bottle.from_analysis(d, metadata).to_markdown()
        assert "It is aged 8 years. [4]" in md
        assert "[3] Proof, [4] Age" in md
        # Idempotent: a second refresh keeps the numbering.
        assert refresh_facts(d).copy.sentences[1]["citations"] == [4]

    def test_a_marker_whose_field_lost_its_number_makes_the_copy_stale(self):
        from image_processor.enrichment import renumber_copy

        _d, _m, sheet = checked_bottle()
        values = sheet.flat_values()
        renumber_copy(sheet, {1: "name", 2: "distillery", 3: "release_year"}, values)  # [3] maps to no fact
        assert sheet.copy.stale is True
        assert "[3]" in sheet.copy.reason
        assert any(b.startswith("description is stale") for b in sheet.publish["blockers"])

    def test_refresh_without_facts_is_a_no_op(self):
        assert refresh_facts({"name": "X"}) is None


class TestAcceptance:
    def test_missing_evidence_gives_blockers_not_invented_values(self):
        """No web results and nothing in the text: Unknowns with reasons and publish blockers."""
        events = []
        empty = WebResearch(status="no_results", reason="search returned no results", provider="duckduckgo",
                            queries=["example creek single barrel"])
        d = _bottle_dict()
        _enricher(FakeResearcher([empty], events=events), FakeTextLLM(EXTRACT_EMPTY, COMPOSE_REPLY, events=events),
                  events).enrich_batch([d], [_metadata(proof=None)], run_id="r")
        facts = d["facts"]
        for name in ("msrp", "age", "release_year", "mash_bill", "proof"):
            assert facts["fields"][name]["value"] is None
            assert facts["fields"][name]["reason"]
        assert d["msrp"] is None and d["age"] is None
        assert facts["publish"]["ready"] is False
        assert "core field proof or abv not supported at medium confidence or better" in facts["publish"]["blockers"]
        assert d["web_research"]["status"] == "no_results"
