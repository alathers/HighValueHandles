"""Unit tests for models.bottle."""

from decimal import Decimal
from pathlib import Path

import pytest

from image_processor.fact_checker import FactChecker
from models.bottle import (
    Bottle,
    Correction,
    ExtractedMetadata,
    Quantity,
    ReferenceLink,
    parse_money,
)
from models.evidence import PageRecord, VendorImage, WebEvidence, WebResearch
from models.facts import FactInput, UserFact


def _sheet(user=(), web=(), label=None):
    """A deterministic fact sheet (no model): label reading plus user facts."""
    inp = FactInput(
        bottle_id="IMG_1",
        label=label or {"name": "X", "distillery": "D"},
        vision_confidence="high",
        images=["IMG_1.jpg"],
        ocr_by_image={},
        user_facts=tuple(UserFact(f, v, "interactive") for f, v in user),
        web=tuple(web),
    )
    return FactChecker.deterministic(inp, "test")


def _msrp_facts(amount):
    return _sheet(user=[("msrp", amount)]).to_dict()


class TestExtractedMetadataVisionFields:
    """vision_model / vision_confidence / warnings serialise and stay backward compatible."""

    def test_defaults(self):
        metadata = ExtractedMetadata()
        assert metadata.vision_model is None
        assert metadata.vision_confidence is None
        assert metadata.warnings == []
        assert ExtractedMetadata().warnings is not metadata.warnings  # no shared default

    def test_round_trip(self):
        metadata = ExtractedMetadata(
            name="Four Roses Single Barrel",
            proof="110.4 Proof",
            source_images=["/a.jpg", "/b.jpg"],
            vision_model="qwen3.6:35b-a3b",
            vision_confidence="high",
            warnings=["check grouping"],
            same_bottle=False,
        )
        data = metadata.to_dict()
        assert data["same_bottle"] is False
        assert data["vision_model"] == "qwen3.6:35b-a3b"
        assert data["vision_confidence"] == "high"
        assert data["warnings"] == ["check grouping"]

        restored = ExtractedMetadata.from_dict(data)
        assert restored.vision_model == "qwen3.6:35b-a3b"
        assert restored.vision_confidence == "high"
        assert restored.warnings == ["check grouping"]
        assert restored.same_bottle is False
        assert restored.source_images == [Path("/a.jpg"), Path("/b.jpg")]

    def test_old_json_without_vision_fields_loads(self):
        restored = ExtractedMetadata.from_dict(
            {"name": "Old", "full_text": "t", "selected_text": "t", "source_images": []}
        )
        assert restored.name == "Old"
        assert restored.vision_model is None
        assert restored.vision_confidence is None
        assert restored.warnings == []
        assert restored.same_bottle is None

    def test_null_warnings_load_as_empty(self):
        assert ExtractedMetadata.from_dict({"warnings": None}).warnings == []

    def test_round_trip_label_price_attempts_and_ocr_by_image(self):
        metadata = ExtractedMetadata(
            name="Stagg",
            label_price="59.99",
            vision_attempts=3,
            ocr_by_image={"IMG_1.jpg": "GEORGE T STAGG", "IMG_2.jpg": "135.4 PROOF"},
        )
        data = metadata.to_dict()
        assert data["label_price"] == "59.99"
        assert "msrp" not in data  # a visible price is never an MSRP
        assert data["vision_attempts"] == 3
        assert data["ocr_by_image"] == {"IMG_1.jpg": "GEORGE T STAGG", "IMG_2.jpg": "135.4 PROOF"}

        restored = ExtractedMetadata.from_dict(data)
        assert restored.label_price == "59.99"
        assert restored.vision_attempts == 3
        assert restored.ocr_by_image == metadata.ocr_by_image
        assert not hasattr(restored, "msrp")

    def test_new_field_defaults_and_older_files(self):
        assert ExtractedMetadata().vision_attempts == 1
        assert ExtractedMetadata().ocr_by_image == {}
        old = ExtractedMetadata.from_dict({"name": "Old", "msrp": "45.00"})
        # Older files called the visible price "msrp"; it was always read from the photos.
        assert old.label_price == "45.00"
        assert old.vision_attempts == 1
        assert old.ocr_by_image == {}


class TestParseMoney:
    """parse_money() accepts the price styles OCR and vision models produce."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("59.99", Decimal("59.99")),
            ("$1,299.00", Decimal("1299.00")),
            ("MSRP $45", Decimal("45.00")),
            (45, Decimal("45.00")),
            (45.5, Decimal("45.50")),
            (Decimal("12.345"), Decimal("12.34")),
            ("N/A", None),
            ("", None),
            (None, None),
            (True, None),
            ("-3", None),
            (float("nan"), None),
            (float("inf"), None),
        ],
    )
    def test_parse_money(self, value, expected):
        assert parse_money(value) == expected


class TestBottleFromAnalysis:
    """Bottle.from_analysis() is the one way to build a Bottle from analyzer output."""

    def test_int_quantity_priced_at_msrp(self):
        bottle = Bottle.from_analysis({"name": "Pappy", "facts": _msrp_facts("$1,299.00")}, quantity=2)

        assert bottle.msrp == Decimal("1299.00")
        assert bottle.quantity == Quantity(count=2, price=Decimal("1299.00"), source="extracted")
        assert bottle.total_value == Decimal("2598.00")

    @pytest.mark.parametrize("msrp", [None, "N/A", 45.5, "0.00", "$59.99"])
    def test_msrp_without_facts_or_a_person_is_ignored(self, msrp):
        """A bare dict msrp (label or shop price) never becomes an MSRP."""
        bottle = Bottle.from_analysis({"name": "X", "msrp": msrp, "label_price": "59.99"})
        assert bottle.msrp is None
        assert bottle.quantity.price is None

    def test_msrp_without_facts_set_by_a_person_is_kept(self):
        correction = {"bottle_id": "B", "field_name": "msrp", "original_value": "", "corrected_value": "45.00",
                      "source": "feedback:c.json"}
        bottle = Bottle.from_analysis({"name": "X", "msrp": "45.00", "corrections": [correction]})
        assert bottle.msrp == Decimal("45.00")

    def test_msrp_comes_only_from_facts(self):
        facts = _sheet().to_dict()  # no MSRP fact
        bottle = Bottle.from_analysis({"name": "X", "msrp": "99.99", "label_price": "79.99", "facts": facts})
        assert bottle.msrp is None
        assert bottle.label_price == Decimal("79.99")
        assert bottle.to_dict()["label_price"] == "79.99"
        assert Bottle.from_analysis({"name": "X", "facts": _msrp_facts("45.5")}).msrp == Decimal("45.50")

    def test_missing_fields_get_placeholders(self):
        bottle = Bottle.from_analysis({"name": None, "distillery": "  ", "proof": "", "age": None})

        assert bottle.name == "Unknown Bottle"
        assert bottle.distillery == "Unknown Distillery"
        assert bottle.specific_release == ""
        assert bottle.proof is None
        assert bottle.age is None
        assert bottle.metadata is not None
        assert bottle.quantity.count == 1

    def test_source_images_merged_in_order_without_duplicates(self):
        metadata = ExtractedMetadata(full_text="ocr", source_images=["/p/IMG_2.jpg"])
        bottle = Bottle.from_analysis(
            {"name": "X", "images": ["/p/IMG_1.jpg", "/p/IMG_2.jpg", "/p/IMG_3.jpg"]}, metadata
        )

        assert bottle.metadata.source_images == [Path("/p/IMG_2.jpg"), Path("/p/IMG_1.jpg"), Path("/p/IMG_3.jpg")]
        assert bottle.metadata.full_text == "ocr"
        assert metadata.source_images == ["/p/IMG_2.jpg"]  # the caller's metadata is not modified

    def test_quantity_object_passes_through(self):
        quantity = Quantity(count=12, price=Decimal("0"), source="manual")

        assert Bottle.from_analysis({"name": "X", "msrp": "10"}, quantity=quantity).quantity is quantity

    @pytest.mark.parametrize(("raw", "expected"), [(3, 3), ("2", 2), (0, 1), (-4, 1), ("many", 1), (None, 1)])
    def test_quantity_from_dict_is_clamped(self, raw, expected):
        assert Bottle.from_analysis({"name": "X", "quantity": raw}).quantity.count == expected

    def test_dict_corrections_are_converted(self):
        correction = Correction(bottle_id="IMG_1", field_name="age", original_value="10", corrected_value="12")
        bottle = Bottle.from_analysis(
            {"name": "X"},
            corrections=[
                {"bottle_id": "IMG_1", "field_name": "name", "original_value": "A", "corrected_value": "B"},
                correction,
            ],
        )

        assert isinstance(bottle.corrections[0], Correction)
        assert bottle.corrections[0].corrected_value == "B"
        assert bottle.corrections[0].bottle_id == "IMG_1"
        assert bottle.corrections[1] is correction

    def test_correction_round_trip_keeps_bottle_id_and_source(self):
        correction = Correction(
            bottle_id="IMG_1", field_name="age", original_value="10", corrected_value="12",
            source="feedback:corrections.json",
        )
        bottle = Bottle.from_analysis({"name": "X"}, corrections=[correction])

        restored = Bottle.from_json(bottle.to_json())

        assert restored.corrections == [correction]
        assert bottle.to_dict()["corrections"][0]["bottle_id"] == "IMG_1"
        assert bottle.to_dict()["corrections"][0]["source"] == "feedback:corrections.json"

    def test_batch_number_copied_from_dict(self):
        bottle = Bottle.from_analysis({"name": "X", "batch_number": " B-7 "})

        assert bottle.metadata.batch_number == "B-7"

    def test_round_trips_through_json(self):
        bottle = Bottle.from_analysis(
            {"name": "X", "distillery": "D", "facts": _msrp_facts("19.99"), "images": ["/p/a.jpg"]}, quantity=2
        )

        restored = Bottle.from_json(bottle.to_json())

        assert restored.msrp == Decimal("19.99")
        assert restored.quantity == bottle.quantity
        assert restored.metadata.source_images == [Path("/p/a.jpg")]

    def test_round_trip_keeps_aware_timestamp(self):
        bottle = Bottle.from_analysis({"name": "X"})
        assert bottle.extracted_at.tzinfo is not None
        assert bottle.to_dict()["extracted_at"].endswith("+00:00")
        assert Bottle.from_json(bottle.to_json()).extracted_at == bottle.extracted_at


class TestUnknownMsrp:
    """A price that is not visible is unknown (null/'Unknown'), never $0.00."""

    def test_unknown_msrp_serialises_as_null(self):
        bottle = Bottle.from_analysis({"name": "Stagg", "msrp": None}, quantity=2)

        assert bottle.msrp is None
        assert bottle.quantity.price is None
        assert bottle.total_value is None
        data = bottle.to_dict()
        assert data["msrp"] is None
        assert data["quantity"]["price"] is None
        restored = Bottle.from_dict(data)
        assert restored.msrp is None
        assert restored.quantity.price is None

    def test_unknown_msrp_markdown(self):
        markdown = Bottle.from_analysis({"name": "Stagg"}, quantity=2).to_markdown()
        assert "**MSRP:** Unknown" in markdown
        assert "Total value: Unknown" in markdown
        assert "$0.00" not in markdown

    def test_older_files_with_zero_msrp_still_load(self):
        data = {
            "name": "X", "distillery": "D", "specific_release": "", "msrp": "0.00",
            "quantity": {"count": 1, "price": "0.00", "source": "extracted"},
            "extracted_at": "2026-11-23T03:38:09",
        }
        bottle = Bottle.from_dict(data)
        assert bottle.msrp == Decimal("0.00")
        assert bottle.total_value == Decimal("0.00")
        # Naive timestamps from older versions are UTC.
        assert bottle.extracted_at.utcoffset().total_seconds() == 0


class TestTotalValue:
    """total_value is quantity * price, or None when the price is unknown (no ohr estimate)."""

    def test_total_value_is_count_times_price(self):
        bottle = Bottle.from_analysis({"name": "X", "facts": _msrp_facts("45.00")}, quantity=3)
        assert bottle.total_value == Decimal("135.00")
        assert Bottle.from_analysis({"name": "X"}, quantity=3).total_value is None

    def test_ohr_is_gone(self):
        bottle = Bottle.from_analysis({"name": "X"})
        assert not hasattr(bottle, "ohr")
        assert "ohr" not in bottle.to_dict()


class TestAbv:
    def test_abv_round_trip(self):
        metadata = ExtractedMetadata(name="X", abv="55.2%")
        assert ExtractedMetadata.from_dict(metadata.to_dict()).abv == "55.2%"
        assert ExtractedMetadata.from_dict({"name": "X"}).abv is None  # older files

        bottle = Bottle.from_analysis({"name": "X", "abv": "55.2%"}, metadata)
        assert bottle.abv == "55.2%"
        assert Bottle.from_dict(bottle.to_dict()).abv == "55.2%"
        assert "| ABV | 55.2% | - | not fact-checked |" in bottle.to_markdown()

    def test_abv_from_metadata_when_dict_lacks_it(self):
        bottle = Bottle.from_analysis({"name": "X"}, ExtractedMetadata(abv="45%"))
        assert bottle.abv == "45%"


def _evidence(sid, url, title="Example Creek Single Barrel"):
    return WebEvidence.create(
        text=f"Example Creek Single Barrel is bottled at 100 proof. ({sid})", source_id=sid, url=url,
        domain=url.split("/")[2], tier="official" if "examplecreek" in url else "retailer", title=title,
        fetched_at="2026-11-23T00:00:00+00:00", http_status=200, query="example creek", from_cache=False,
        truncated=False,
    )


def _research():
    w1 = _evidence("W1", "https://shop.test/ec")
    w2 = _evidence("W2", "https://examplecreek.test/sb")
    return WebResearch(
        status="ok", provider="duckduckgo", queries=["example creek single barrel"],
        searched_at="2026-11-23T00:00:00+00:00",
        pages=[
            PageRecord(url=w1.url, title="Shop", domain="shop.test", source_class="retailer",
                       retrieved_at=w1.fetched_at, outcome="fetched_not_cited", reason="W1: no verified fact cites it"),
            PageRecord(url=w2.url, title="Official", domain="examplecreek.test", source_class="official",
                       retrieved_at=w2.fetched_at, outcome="cited", cited_for=["proof"]),
            PageRecord(url="https://forum.test/x", title=None, domain="forum.test", source_class=None,
                       retrieved_at=None, outcome="skipped", reason="deny-listed"),
        ],
        evidence=[w1, w2],
        vendor_image=VendorImage(cache_path="/tmp/x.jpg", image_url="https://examplecreek.test/i.jpg",
                                 page_url=w2.url, retrieved_at="2026-11-23T00:00:00+00:00",
                                 verdict={"same_product": True, "confidence": "high"}),
    )


class TestSchemaV2:
    def _sheet_citing_w2(self):
        from models.facts import Citation

        sheet = _sheet(label={"name": "Example Creek Single Barrel", "distillery": "Example Creek", "proof": "100"},
                       web=_research().evidence)
        sheet.fields["proof"].citations.append(Citation("W2", "bottled at 100 proof"))
        return sheet

    def test_v2_round_trip(self):
        research = _research()
        sheet = self._sheet_citing_w2()
        bottle = Bottle.from_analysis(
            {
                "name": "Example Creek Single Barrel", "facts": sheet.to_dict(), "web_research": research.to_dict(),
                "vendor_image": research.vendor_image.to_dict(), "run_id": "20261123_000000",
                "label_price": "64.99",
            },
            ExtractedMetadata(source_images=["/p/IMG_1.jpg"]),
        )
        data = bottle.to_dict()
        assert {"facts", "web_research", "vendor_image", "run_id", "reference_links", "label_price",
                "corrections", "review_required"} <= set(data)
        restored = Bottle.from_dict(data)
        assert restored.facts.to_dict() == sheet.to_dict()
        assert restored.web_research.to_dict() == research.to_dict()
        assert [p.outcome for p in restored.web_research.pages] == ["fetched_not_cited", "cited", "skipped"]
        assert restored.web_research.pages[2].reason == "deny-listed"
        assert restored.vendor_image["usage"] == "reference only - not licensed for publication"
        assert restored.run_id == "20261123_000000"
        assert restored.label_price == Decimal("64.99")
        assert restored.msrp is None
        assert restored.reference_links == bottle.reference_links

    def test_reference_links_cited_first(self):
        bottle = Bottle.from_analysis(
            {"name": "X", "facts": self._sheet_citing_w2().to_dict(), "web_research": _research().to_dict()}
        )
        assert [link.url for link in bottle.reference_links] == [
            "https://examplecreek.test/sb", "https://shop.test/ec",
        ]
        first, second = bottle.reference_links
        assert first == ReferenceLink(
            url="https://examplecreek.test/sb", label="Example Creek Single Barrel", source_class="official",
            retrieved_at="2026-11-23T00:00:00+00:00", used_as_evidence=True,
        )
        assert second.used_as_evidence is False and second.source_class == "retailer"

    def test_reference_links_always_emitted(self):
        assert Bottle.from_analysis({"name": "X"}).to_dict()["reference_links"] == []
        assert ReferenceLink.from_dict({"url": "https://a.test", "label": "A"}).used_as_evidence is False

    def test_review_required(self):
        from models.facts import Candidate, Citation, Conflict

        sheet = _sheet()
        assert Bottle.from_analysis({"name": "X", "facts": sheet.to_dict()}).review_required is False
        sheet.conflicts.append(Conflict("proof-1", "proof", "value_mismatch",
                                        [Candidate(90.0, [Citation("L1", "90", "reading")], "label")], "x"))
        assert Bottle.from_analysis({"name": "X", "facts": sheet.to_dict()}).review_required is True
