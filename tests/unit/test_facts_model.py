"""Tests of models.facts: JSON round trip and flat_values."""

import json
from decimal import Decimal

import pytest

from models.facts import (
    FACT_FIELDS,
    Candidate,
    Citation,
    Conflict,
    FactSheet,
    FieldFact,
    PriceObservation,
    PublishCopy,
    Source,
    UserFact,
    Verification,
)


def sample_sheet():
    label = Citation("L1", "125.9 PROOF", "reading")
    ocr = Citation("O1", "BARREL PROOF 125.9 PROOF")
    web = Citation("W1", "bottled at 130.2 proof")
    fields = {name: FieldFact.unknown(f"{name} not stated") for name in FACT_FIELDS}
    fields["name"] = FieldFact("Booker's", "supported", "high", "label(high)+ocr", [label], [Candidate("Booker's", [label], "label")])
    fields["proof"] = FieldFact(
        None, "conflict", None, None, [],
        [Candidate(125.9, [label, ocr], "label"), Candidate(130.2, [web], "web")],
        "sources disagree",
    )
    fields["msrp"] = FieldFact(Decimal("89.99"), "user", "high", "user", [Citation("U1", "interactive", "reading")],
                               [Candidate(Decimal("89.99"), [Citation("U1", "interactive", "reading")], "user")])
    fields["release_year"] = FieldFact(2023, "supported", "medium", "web[x(official)]", [Citation("W1", "released in 2023")], [])
    return FactSheet(
        fields=fields,
        sources=[
            Source("L1", "label", "IMG_1.jpg"),
            Source("O1", "ocr", "IMG_1.jpg", content_sha256="ab" * 32),
            Source("W1", "web", "https://x.example/p", url="https://x.example/p", title="T", tier="official",
                   fetched_at="2026-11-23T00:00:00Z", content_sha256="cd" * 32, evidence_path="evidence/W1.txt"),
            Source("U1", "user", "interactive"),
        ],
        conflicts=[Conflict("proof-1", "proof", "value_mismatch", fields["proof"].candidates, "L1 125.9 vs W1 130.2"),
                   Conflict("msrp-1", "msrp", "value_mismatch", [Candidate(Decimal("79.99"), [web], "web")], "n",
                            resolved_by=Citation("U1", "interactive", "reading"))],
        prices=[PriceObservation("retail", Decimal("64.99"), "USD", Citation("L1", "price visible on label", "reading")),
                PriceObservation("msrp", Decimal("79.99"), "USD", Citation("W1", "MSRP $79.99"), "low")],
        verification=Verification("verified", None, "gpt-oss:120b", "high", "facts-1", 12.5, 2, True,
                                  [{"stage": "extract", "reason": "quote_not_found", "source_id": "W2"}]),
        copy=PublishCopy([{"text": "Booker's.", "citations": [1], "fields": ["name"]}], "Booker's. [1]"),
        publish={"ready": False, "blockers": ["unresolved conflict proof-1"], "group_warnings": []},
        review={"approved": False, "by": None, "at": None},
    )


def test_round_trip_through_json():
    sheet = sample_sheet()
    data = json.loads(json.dumps(sheet.to_dict()))
    again = FactSheet.from_dict(data)
    assert again == sheet
    assert again.to_dict() == sheet.to_dict()
    assert data["fields"]["msrp"]["value"] == "89.99"
    assert data["prices"][0]["amount"] == "64.99"
    assert isinstance(again.fields["msrp"].value, Decimal)
    assert again.conflicts[1].candidates[0].value == Decimal("79.99")
    assert data["fields"]["proof"]["candidates"][0]["value"] == 125.9


def test_flat_values():
    flat = sample_sheet().flat_values()
    assert set(flat) == set(FACT_FIELDS)
    assert flat["name"] == "Booker's"
    assert flat["proof"] is None  # conflict
    assert flat["msrp"] == Decimal("89.99")  # user
    assert flat["release_year"] == 2023
    assert flat["age"] is None  # unknown


def test_validation():
    with pytest.raises(ValueError, match="status"):
        FieldFact(None, "verified")
    with pytest.raises(ValueError, match="value None"):
        FieldFact(115, "conflict")
    with pytest.raises(ValueError, match="kind"):
        PriceObservation("street", Decimal(1), "USD", Citation("W1", "x"))
    with pytest.raises(ValueError, match="positive"):
        PriceObservation("retail", Decimal(0), "USD", Citation("W1", "x"))
    with pytest.raises(ValueError, match="float"):
        PriceObservation.from_dict({"kind": "retail", "amount": 1.5, "currency": "USD", "citation": {"source_id": "W1", "quote": "x"}})
    with pytest.raises(ValueError, match="field"):
        UserFact("ohr", 1, "interactive")
    with pytest.raises(ValueError, match="kind"):
        Source("X1", "rumor", "x")


def test_user_fact_round_trip():
    uf = UserFact("msrp", Decimal("59.99"), "hint:hints.md#IMG_1: price sticker")
    assert UserFact.from_dict(json.loads(json.dumps(uf.to_dict()))) == uf
