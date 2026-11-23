"""Unit tests for models.bottle."""

from decimal import Decimal
from pathlib import Path

import pytest

from models.bottle import Bottle, Correction, ExtractedMetadata, Quantity, parse_money


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
        bottle = Bottle.from_analysis({"name": "Pappy", "msrp": "$1,299.00"}, quantity=2)

        assert bottle.msrp == Decimal("1299.00")
        assert bottle.quantity == Quantity(count=2, price=Decimal("1299.00"), source="extracted")
        assert bottle.total_value == Decimal("2598.00")

    @pytest.mark.parametrize(
        ("msrp", "expected"), [(None, None), ("N/A", None), (45.5, Decimal("45.50")), ("0.00", Decimal("0.00"))]
    )
    def test_msrp_parsing(self, msrp, expected):
        assert Bottle.from_analysis({"name": "X", "msrp": msrp}).msrp == expected

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
            {"name": "X", "distillery": "D", "msrp": "19.99", "images": ["/p/a.jpg"]}, quantity=2
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
        assert "**Total Value:** Unknown" in markdown
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
        assert Bottle.from_analysis({"name": "X", "msrp": "45.00"}, quantity=3).total_value == Decimal("135.00")

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
        assert "**ABV:** 55.2%" in bottle.to_markdown()

    def test_abv_from_metadata_when_dict_lacks_it(self):
        bottle = Bottle.from_analysis({"name": "X"}, ExtractedMetadata(abv="45%"))
        assert bottle.abv == "45%"
