"""Opt-in live checks of the gpt-oss fact check against a local Ollama.

Skipped unless ``BOURBON_LIVE_TESTS=1`` and ``BOURBON_LIVE_TEXT=1`` and the
model is already installed (never pulls). Catches Ollama/model upgrades that
break JSON-schema output together with a think level, and checks that the
model does not turn a shop price into an MSRP.

Run::

    BOURBON_LIVE_TESTS=1 BOURBON_LIVE_TEXT=1 \
        uv run --extra dev pytest tests/integration/test_fact_check_live.py

Override the model with ``BOURBON_LIVE_TEXT_MODEL`` (default gpt-oss:120b).
"""

import os

import pytest
import requests

from image_processor.fact_checker import FactChecker
from image_processor.fact_prompt import (
    TEXT_OPTIONS,
    Claim,
    PageMatch,
    build_evidence_pack,
    display_value,
    parse_json_reply,
)
from image_processor.fact_verify import Accepted, normalize, verify_claim
from lib.ollama_client import OllamaClient
from models.evidence import WebEvidence
from models.facts import FactInput

OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("BOURBON_LIVE_TEXT_MODEL", "gpt-oss:120b")

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BOURBON_LIVE_TESTS") != "1" or os.environ.get("BOURBON_LIVE_TEXT") != "1",
        reason="set BOURBON_LIVE_TESTS=1 BOURBON_LIVE_TEXT=1 to run",
    ),
    pytest.mark.allow_model_server,
    pytest.mark.live,
]


def _model_installed() -> bool:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json().get("models", [])
    except (requests.RequestException, ValueError):
        return False
    return any(t.get("name") == MODEL or t.get("model") == MODEL for t in tags)


@pytest.fixture(scope="module")
def client():
    if not _model_installed():
        pytest.skip(f"{MODEL} is not installed in Ollama at {OLLAMA_URL} (this test never pulls)")
    c = OllamaClient(base_url=OLLAMA_URL, model=MODEL, generate_timeout=900)
    yield c
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL, "keep_alive": 0}, timeout=60)
    finally:
        c.close()


def _web(sid, text, tier):
    return WebEvidence.create(
        text=text, source_id=sid, url=f"https://{sid.lower()}.example.com/p", domain=f"{sid.lower()}.example.com",
        tier=tier, title="Weller Special Reserve", fetched_at="2026-11-23T00:00:00Z", http_status=200,
        query="weller special reserve", from_cache=False, truncated=False,
    )


RETAIL_PAGE = (
    "Weller Special Reserve Kentucky Straight Wheated Bourbon Whiskey, 750ml. "
    "Bottled at 90 proof. Our price $64.99. Sale! Add to cart. Free shipping on orders over $99."
)
OFFICIAL_PAGE = (
    "W.L. Weller Special Reserve. Kentucky Straight Bourbon Whiskey. 90 proof. "
    "Distilled by Buffalo Trace Distillery, Frankfort, Kentucky. "
    "Mash bill: wheated recipe. Aged 7 years."
)


def _input(pages):
    return FactInput(
        bottle_id="IMG_LIVE",
        label={"name": "Weller Special Reserve", "distillery": "Buffalo Trace", "proof": "90"},
        vision_confidence="high",
        images=["IMG_LIVE.jpg"],
        ocr_by_image={"IMG_LIVE.jpg": "W.L. WELLER\nSPECIAL RESERVE\nKENTUCKY STRAIGHT BOURBON WHISKEY\n90 PROOF"},
        web=pages,
    )


def test_schema_output_with_think_level(client):
    schema = {
        "type": "object",
        "properties": {"proof": {"type": "number"}, "quote": {"type": "string"}, "source_id": {"type": "string", "enum": ["S1"]}},
        "required": ["proof", "quote", "source_id"],
        "additionalProperties": False,
    }
    prompt = '<source id="S1">Bottled at 115 proof.</source>\nGive the proof, a verbatim quote and the source id.'
    result = client.chat_text(prompt, MODEL, format=schema, think="medium", options=dict(TEXT_OPTIONS), keep_alive="10m")
    data = parse_json_reply(result.content)
    assert data is not None, result.content
    assert data["proof"] == 115
    assert data["source_id"] == "S1"
    assert result.thinking.strip()


def test_retail_only_page_gives_no_msrp(client):
    sheet = FactChecker(client, model=MODEL).check(_input([_web("W1", RETAIL_PAGE, "retailer")]))
    assert sheet.verification.status in ("verified", "partial"), sheet.verification.reason
    assert not [p for p in sheet.prices if p.kind == "msrp"], sheet.prices
    assert sheet.fields["msrp"].status == "unknown"


def test_full_check_citations_reverify(client):
    inp = _input([_web("W1", OFFICIAL_PAGE, "official"), _web("W2", RETAIL_PAGE, "retailer")])
    sheet = FactChecker(client, model=MODEL).check(inp)
    assert sheet.verification.status in ("verified", "partial"), sheet.verification.reason
    pack = build_evidence_pack(inp)
    exact = {sid: PageMatch(sid, "exact", None) for sid in pack.web_ids}
    checked = 0
    for name, fact in sheet.fields.items():
        for cand in fact.candidates:
            if cand.origin not in ("web", "ocr"):
                continue
            for cit in cand.citations:
                claim = Claim(name, display_value(name, cand.value), cit.source_id, cit.quote)
                if name in ("proof", "abv", "age", "batch_number", "release_year") and cit.source_id.startswith("W"):
                    # Page-match quotes are not stored; re-check the quote and value only.
                    src = pack.get(cit.source_id)
                    assert normalize(cit.quote) in normalize(src.text)
                    continue
                result = verify_claim(claim, pack, exact)
                assert isinstance(result, Accepted), (name, cit, result)
                checked += 1
        for cit in fact.citations:
            if cit.source_id.startswith(("O", "W")) and cit.quote_kind == "verbatim":
                assert normalize(cit.quote) in normalize(pack.ocr_full_texts().get(cit.source_id) or pack.get(cit.source_id).text)
    print(f"re-verified {checked} citation(s); rejected={len(sheet.verification.rejected)}; "
          f"blockers={sheet.publish['blockers']}; copy={sheet.copy.description_md if sheet.copy else None}")
