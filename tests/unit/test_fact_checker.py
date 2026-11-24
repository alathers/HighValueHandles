"""Orchestration tests of image_processor.fact_checker with a scripted fake text model."""

import json
from decimal import Decimal

import pytest

from image_processor.fact_checker import FactChecker
from image_processor.fact_prompt import PROMPT_VERSION, TEXT_OPTIONS
from lib.exceptions import (
    OllamaConnectionError,
    TextModelUnavailableError,
    TextTimeoutError,
    TextTransientError,
)
from lib.ollama_client import ChatResult
from models.evidence import WebEvidence
from models.facts import FactInput, FactSheet

MODEL = "gpt-oss:120b"
W1_TEXT = (
    "Elijah Craig Barrel Proof Batch C923 is bottled at 133.2 proof. Aged 12 years. "
    "Distilled by Heaven Hill Distillery. MSRP of $59.99."
)


class FakeTextLLM:
    """Returns scripted replies (ChatResult, str or Exception) and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat_text(self, prompt, model=None, *, system=None, format=None, think=None, options=None,
                  keep_alive="5m", read_timeout=None):
        self.calls.append({
            "prompt": prompt, "model": model, "system": system, "format": format, "think": think,
            "options": options, "keep_alive": keep_alive, "read_timeout": read_timeout,
        })
        if not self.replies:
            raise AssertionError("unexpected extra call")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            return ChatResult(content=reply, thinking="...", model=model or "", done_reason="stop")
        return reply


def web(sid="W1", text=W1_TEXT, tier="official"):
    return WebEvidence.create(
        text=text, source_id=sid, url=f"https://{sid.lower()}.example.com/ec", domain=f"{sid.lower()}.example.com",
        tier=tier, title="Elijah Craig Barrel Proof", fetched_at="2026-11-23T00:00:00Z", http_status=200,
        query="elijah craig barrel proof", from_cache=False, truncated=False,
    )


def make_input(label=None, web_pages=None, bottle_id="IMG_1"):
    return FactInput(
        bottle_id=bottle_id,
        label=label or {"name": "Elijah Craig Barrel Proof", "distillery": "Heaven Hill Distillery", "proof": "133.2"},
        vision_confidence="high",
        images=["/Users/someone/Pictures/HighValueRoom/IMG_1.jpg", "C:\\photos\\IMG_2.jpg"],
        ocr_by_image={
            "/Users/someone/Pictures/HighValueRoom/IMG_1.jpg": "ELIJAH CRAIG BARREL PROOF\nHEAVEN HILL DISTILLERY\n133.2 PROOF",
        },
        web=[web()] if web_pages is None else web_pages,
    )


EXTRACT_OK = json.dumps({
    "page_match": [{"source_id": "W1", "verdict": "exact", "quote": "Elijah Craig Barrel Proof Batch C923"}],
    "claims": [
        {"field": "age", "value": "12 years", "source_id": "W1", "quote": "Aged 12 years."},
        {"field": "proof", "value": "133.2", "source_id": "W1", "quote": "bottled at 133.2 proof"},
        {"field": "name", "value": "Elijah Craig Barrel Proof", "source_id": "W1", "quote": "invented quote not on page"},
    ],
    "prices": [{"kind": "msrp", "amount": 59.99, "currency": "USD", "source_id": "W1", "quote": "MSRP of $59.99"}],
})
COMPOSE_OK = json.dumps({
    "sentences": [
        {"text": "Elijah Craig Barrel Proof is bottled at 133.2 proof.", "citations": [1, 3], "fields": ["name", "proof"]},
        {"text": "It is aged 12 years.", "citations": [4], "fields": ["age"]},
    ]
})


def checker(llm, **kw):
    return FactChecker(llm, model=MODEL, **kw)


class TestHappyPath:
    def test_full_check(self):
        llm = FakeTextLLM(EXTRACT_OK, COMPOSE_OK)
        sheet = checker(llm).check(make_input())
        assert isinstance(sheet, FactSheet)
        v = sheet.verification
        assert v.status == "verified"
        assert v.prompt_version == PROMPT_VERSION
        assert v.text_model == MODEL
        assert v.think == "high"
        assert v.attempts == 2
        assert sheet.fields["age"].value == "12 years"
        assert sheet.fields["age"].confidence == "medium"
        assert sheet.fields["proof"].confidence == "high"
        assert sheet.fields["msrp"].value == Decimal("59.99")
        assert [r["reason"] for r in v.rejected] == ["quote_not_found"]
        assert sheet.copy is not None
        assert sheet.copy.description_md.endswith("It is aged 12 years. [4]")
        assert sheet.publish["blockers"] == ["not approved by a reviewer"]

    def test_payload_think_format_and_identical_options(self):
        llm = FakeTextLLM(EXTRACT_OK, COMPOSE_OK)
        checker(llm).check(make_input())
        extract, compose = llm.calls
        assert extract["think"] == "high" and compose["think"] == "medium"
        assert isinstance(extract["think"], str)
        assert isinstance(extract["format"], dict) and isinstance(compose["format"], dict)
        assert extract["format"]["properties"]["claims"]["items"]["properties"]["source_id"]["enum"] == ["O1", "W1"]
        assert extract["options"] == compose["options"] == TEXT_OPTIONS
        assert extract["model"] == compose["model"] == MODEL
        assert extract["keep_alive"] == "10m"
        assert extract["read_timeout"] == 900.0
        assert "untrusted data" in extract["system"]

    def test_prompts_have_file_names_not_directories(self):
        llm = FakeTextLLM(EXTRACT_OK, COMPOSE_OK)
        checker(llm).check(make_input())
        for call in llm.calls:
            assert "/Users" not in call["prompt"]
            assert "C:\\" not in call["prompt"]
            assert "Pictures" not in call["prompt"]
        assert "IMG_1.jpg" in llm.calls[0]["prompt"]
        assert "IMG_2.jpg" in llm.calls[0]["prompt"]
        assert "https://" not in llm.calls[0]["prompt"]

    def test_compose_needs_name_and_two_fields(self):
        llm = FakeTextLLM(json.dumps({"page_match": [], "claims": [], "prices": []}))
        sheet = checker(llm).check(make_input(label={"name": "Elijah Craig"}, web_pages=[]))
        assert len(llm.calls) == 1
        assert sheet.copy is None
        assert sheet.verification.status == "verified"

    def test_no_sources_skips_extract(self):
        compose = json.dumps({"sentences": [
            {"text": "Elijah Craig Barrel Proof is bottled at 133.2 proof.", "citations": [1, 2], "fields": ["name", "proof"]},
        ]})
        llm = FakeTextLLM(compose)
        inp = FactInput("IMG_1", {"name": "Elijah Craig Barrel Proof", "proof": "133.2"}, "high", ["IMG_1.jpg"], {}, (), [])
        sheet = checker(llm).check(inp)
        assert len(llm.calls) == 1
        assert "<fields>" in llm.calls[0]["prompt"]
        assert sheet.verification.status == "verified"
        assert sheet.verification.reason == "no OCR or web text to check; label reading only"
        assert sheet.copy is not None

    def test_usage_recorder(self):
        seen = []
        llm = FakeTextLLM("nope", EXTRACT_OK, COMPOSE_OK)
        checker(llm, usage_recorder=lambda *a: seen.append(a)).check(make_input())
        assert [(m, ok, t) for m, _d, ok, t in seen] == [(MODEL, False, "high"), (MODEL, True, "high"), (MODEL, True, "medium")]


class TestRetries:
    def test_invalid_then_valid_retried_once(self):
        llm = FakeTextLLM("I think the proof is 133.2", EXTRACT_OK, COMPOSE_OK)
        sheet = checker(llm).check(make_input())
        assert len(llm.calls) == 3
        assert sheet.verification.status == "verified"
        assert sheet.verification.attempts == 3

    def test_schema_invalid_then_valid(self):
        llm = FakeTextLLM(json.dumps({"claims": []}), EXTRACT_OK, COMPOSE_OK)
        assert checker(llm).check(make_input()).verification.status == "verified"

    def test_invalid_twice_fails_without_raising(self):
        llm = FakeTextLLM("", "```json\n{\"oops\": 1}\n```")
        sheet = checker(llm).check(make_input())
        assert len(llm.calls) == 2
        assert sheet.verification.status == "failed"
        assert sheet.verification.attempts == 2
        assert sheet.fields["name"].status == "supported"
        assert sheet.fields["age"].status == "unknown"
        assert any(b.startswith("text verification failed") for b in sheet.publish["blockers"])

    def test_length_cutoff_retried_at_lower_think(self):
        cut = ChatResult(content='{"page_match": [', done_reason="length")
        llm = FakeTextLLM(cut, EXTRACT_OK, COMPOSE_OK)
        sheet = checker(llm).check(make_input())
        assert [c["think"] for c in llm.calls] == ["high", "medium", "medium"]
        assert sheet.verification.degraded_think is True
        assert sheet.verification.think == "medium"
        assert sheet.verification.status == "verified"

    def test_timeout_fails_without_retry(self):
        llm = FakeTextLLM(TextTimeoutError("read timed out"))
        sheet = checker(llm).check(make_input())
        assert len(llm.calls) == 1
        assert sheet.verification.status == "failed"
        assert "timed out" in sheet.verification.reason

    def test_transient_is_retried(self):
        llm = FakeTextLLM(TextTransientError("HTTP 503"), EXTRACT_OK, COMPOSE_OK)
        assert checker(llm).check(make_input()).verification.status == "verified"

    def test_unavailable_disables_later_bottles(self):
        llm = FakeTextLLM(TextModelUnavailableError("model 'gpt-oss:120b' not found"))
        fc = checker(llm)
        first = fc.check(make_input())
        assert first.verification.status == "skipped"
        assert "text model unavailable" in first.verification.reason
        assert fc.disabled_reason
        second = fc.check(make_input(bottle_id="IMG_9"))
        assert len(llm.calls) == 1
        assert second.verification.status == "skipped"
        assert second.verification.reason == fc.disabled_reason

    def test_connection_error_fails(self):
        llm = FakeTextLLM(OllamaConnectionError("refused"))
        sheet = checker(llm).check(make_input())
        assert sheet.verification.status == "failed"
        assert sheet.verification.reason == "text model server unreachable"

    def test_max_attempts_one(self):
        llm = FakeTextLLM("garbage")
        sheet = checker(llm, max_attempts=1).check(make_input())
        assert len(llm.calls) == 1
        assert sheet.verification.status == "failed"

    def test_max_attempts_capped_at_two(self):
        llm = FakeTextLLM("a", "b")
        checker(llm, max_attempts=5).check(make_input())
        assert len(llm.calls) == 2

    def test_unexpected_exception_never_raises(self):
        llm = FakeTextLLM(RuntimeError("boom"))
        sheet = checker(llm).check(make_input())
        assert sheet.verification.status == "failed"
        assert "RuntimeError: boom" in sheet.verification.reason


class TestCompose:
    def test_bad_copy_gives_partial(self):
        bad = json.dumps({"sentences": [{"text": "It is bottled at 140 proof.", "citations": [3], "fields": ["proof"]}]})
        llm = FakeTextLLM(EXTRACT_OK, bad)
        sheet = checker(llm).check(make_input())
        assert sheet.copy is None
        assert sheet.verification.status == "partial"
        assert "number_not_cited" in sheet.verification.reason
        assert any(r.get("reason") == "number_not_cited" for r in sheet.verification.rejected)
        assert any(b.startswith("text verification partial") for b in sheet.publish["blockers"])
        assert "no verified description" in sheet.publish["blockers"]

    def test_compose_failure_gives_partial(self):
        llm = FakeTextLLM(EXTRACT_OK, TextTimeoutError("slow"))
        sheet = checker(llm).check(make_input())
        assert sheet.copy is None
        assert sheet.verification.status == "partial"
        assert sheet.fields["age"].status == "supported"


class TestDeterministic:
    def test_llm_none_web_never_fills_a_field(self):
        sheet = checker(None).check(make_input())
        assert sheet.verification.status == "skipped"
        assert sheet.fields["age"].status == "unknown"
        assert sheet.fields["msrp"].status == "unknown"
        assert sheet.fields["name"].status == "supported"
        assert sheet.fields["proof"].confidence == "high"
        web_sources = [s for s in sheet.sources if s.kind == "web"]
        assert [s.url for s in web_sources] == ["https://w1.example.com/ec"]
        for fact in sheet.fields.values():
            assert not any(c.source_id.startswith("W") for c in fact.citations)
        assert any(b.startswith("text verification skipped") for b in sheet.publish["blockers"])

    def test_static_deterministic(self):
        sheet = FactChecker.deterministic(make_input(), "--no-verify")
        assert sheet.verification.status == "skipped"
        assert sheet.verification.reason == "--no-verify"
        assert sheet.copy is None


@pytest.mark.parametrize("bad", [None, 3, "x", {"claims": "no"}])
def test_parse_never_crashes(bad):
    llm = FakeTextLLM(json.dumps(bad), json.dumps(bad))
    assert checker(llm).check(make_input()).verification.status == "failed"


class TestReviewFixes:
    def test_no_extract_records_no_think(self):
        llm = FakeTextLLM(COMPOSE_OK)
        inp = FactInput(
            bottle_id="IMG_1", label={"name": "Elijah Craig Barrel Proof", "proof": "133.2"},
            vision_confidence="high", images=["IMG_1.jpg"], ocr_by_image={}, web=[],
        )
        sheet = checker(llm).check(inp)
        assert sheet.verification.think is None

    def test_unavailable_during_compose_keeps_extract_results(self):
        llm = FakeTextLLM(EXTRACT_OK, TextModelUnavailableError("model gone"))
        fc = checker(llm)
        sheet = fc.check(make_input())
        assert sheet.verification.status == "partial"
        assert sheet.fields["age"].value == "12 years"
        assert sheet.copy is None
        assert fc.disabled_reason
        later = fc.check(make_input(bottle_id="IMG_2"))
        assert later.verification.status == "skipped"
