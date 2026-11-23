"""Unit tests for image_processor.corrections.apply_corrections and Correction.

A correction is aimed at one bottle by ``bottle_id`` and applied only while
that bottle still holds ``original_value``; nothing ever lands on another
bottle.
"""

from datetime import UTC, datetime

import pytest

from image_processor.corrections import (
    CORRECTABLE_FIELDS,
    STALE_ORIGINAL_VALUE,
    UNKNOWN_BOTTLE_ID,
    UNKNOWN_FIELD,
    apply_corrections,
)
from models.bottle import Bottle, Correction, ExtractedMetadata


def _bottles():
    """Two bottles that share a distillery (the case field-value matching got wrong)."""
    return [
        {"bottle_id": "IMG_1", "name": "Eagle Rare", "distillery": "Buffalo Trace"},
        {"bottle_id": "IMG_3", "name": "Stagg", "distillery": "Buffalo Trace"},
    ]


def _correction(bottle_id, field_name, original, corrected, **kwargs):
    return Correction(
        bottle_id=bottle_id,
        field_name=field_name,
        original_value=original,
        corrected_value=corrected,
        **kwargs,
    )


class TestApplyCorrections:
    def test_only_the_targeted_bottle_changes(self):
        bottles = _bottles()
        correction = _correction("IMG_3", "distillery", "Buffalo Trace", "Sazerac")

        result, unapplied = apply_corrections(bottles, [], [correction])

        assert result is bottles
        assert unapplied == []
        assert bottles[1]["distillery"] == "Sazerac"
        assert bottles[0]["distillery"] == "Buffalo Trace"
        assert "corrections" not in bottles[0]
        assert bottles[1]["corrections"] == [correction.to_dict()]

    def test_unknown_bottle_id_changes_nothing_and_is_returned(self):
        bottles = _bottles()
        before = [dict(b) for b in bottles]
        correction = _correction("IMG_9", "distillery", "Buffalo Trace", "Sazerac")

        _, unapplied = apply_corrections(bottles, [], [correction])

        assert unapplied == [(correction, UNKNOWN_BOTTLE_ID)]
        assert bottles == before

    def test_stale_original_value_is_returned_unapplied(self):
        bottles = _bottles()
        correction = _correction("IMG_1", "name", "Blanton's", "Eagle Rare 10")

        _, unapplied = apply_corrections(bottles, [], [correction])

        assert unapplied == [(correction, STALE_ORIGINAL_VALUE)]
        assert bottles[0]["name"] == "Eagle Rare"
        assert "corrections" not in bottles[0]

    def test_original_value_compared_as_stripped_strings(self):
        bottles = [{"bottle_id": "b1", "proof": 101}, {"bottle_id": "b2"}]
        corrections = [
            _correction("b1", "proof", " 101 ", "100"),
            _correction("b2", "msrp", "", "49.99"),  # a missing value counts as ""
        ]

        _, unapplied = apply_corrections(bottles, [], corrections)

        assert unapplied == []
        assert bottles[0]["proof"] == "100"
        assert bottles[1]["msrp"] == "49.99"

    def test_sequential_corrections_of_one_field(self):
        bottles = _bottles()
        corrections = [
            _correction("IMG_1", "name", "Eagle Rare", "Eagle Rare 10"),
            _correction("IMG_1", "name", "Eagle Rare 10", "Eagle Rare 10 Year"),
        ]

        _, unapplied = apply_corrections(bottles, [], corrections)

        assert unapplied == []
        assert bottles[0]["name"] == "Eagle Rare 10 Year"
        assert [c["original_value"] for c in bottles[0]["corrections"]] == ["Eagle Rare", "Eagle Rare 10"]

    def test_bottle_id_falls_back_to_primary_image_stem(self):
        """Batch JSONs from runs without assigned IDs: the metadata's first image names the bottle."""
        bottles = [{"name": "A"}, {"name": "A"}]
        metadata = [
            ExtractedMetadata(source_images=["/p/IMG_7.jpg"]),
            ExtractedMetadata(source_images=["/p/IMG_8.jpg"]),
        ]
        correction = _correction("IMG_8", "name", "A", "B")

        _, unapplied = apply_corrections(bottles, metadata, [correction])

        assert unapplied == []
        assert [b["name"] for b in bottles] == ["A", "B"]

    def test_ambiguous_bottle_id_is_never_applied(self):
        bottles = [{"bottle_id": "dup", "name": "A"}, {"bottle_id": "dup", "name": "A"}]
        correction = _correction("dup", "name", "A", "B")

        _, unapplied = apply_corrections(bottles, [], [correction])

        assert unapplied == [(correction, UNKNOWN_BOTTLE_ID)]
        assert [b["name"] for b in bottles] == ["A", "A"]

    @pytest.mark.parametrize("field_name", ["nmae", "corrections", "bottle_id", "images", "confidence"])
    def test_uncorrectable_field_is_returned_unapplied(self, field_name):
        """A typo or a bookkeeping key never "applies" to a key nothing reads."""
        bottles = _bottles()
        before = [dict(b) for b in bottles]
        correction = _correction("IMG_1", field_name, "", "x")

        _, unapplied = apply_corrections(bottles, [], [correction])

        assert unapplied == [(correction, UNKNOWN_FIELD)]
        assert bottles == before

    def test_label_and_hint_fields_are_correctable(self):
        assert {
            "name", "distillery", "bottler", "specific_release", "batch_number",
            "proof", "abv", "age", "release_year", "msrp",
        } <= CORRECTABLE_FIELDS

    def test_empty_corrections(self):
        bottles = _bottles()
        assert apply_corrections(bottles, [], []) == (bottles, [])

    def test_applied_corrections_rebuild_on_the_bottle(self):
        bottles = _bottles()
        apply_corrections(bottles, [], [_correction("IMG_3", "name", "Stagg", "Stagg Jr")])

        bottle = Bottle.from_analysis(bottles[1], corrections=bottles[1]["corrections"])

        assert bottle.name == "Stagg Jr"
        assert bottle.corrections[0].bottle_id == "IMG_3"


class TestCorrectionModel:
    def test_round_trip_keeps_bottle_id_and_source(self):
        correction = _correction(
            "IMG_3",
            "distillery",
            "Buffalo Trace",
            "Sazerac",
            source="feedback:/tmp/corrections.json",
            corrected_at=datetime(2026, 11, 23, 12, 0, tzinfo=UTC),
        )

        data = correction.to_dict()
        restored = Correction.from_dict(data)

        assert data["bottle_id"] == "IMG_3"
        assert data["source"] == "feedback:/tmp/corrections.json"
        assert restored == correction

    def test_source_defaults_to_interactive(self):
        assert _correction("b", "name", "A", "B").source == "interactive"
        data = _correction("b", "name", "A", "B").to_dict()
        del data["source"]
        assert Correction.from_dict(data).source == "interactive"

    def test_hint_source_form(self):
        source = "hint:hints.toml#IMG_3: barrel number is on the back label"
        assert _correction("b", "name", "A", "B", source=source).source == source

    def test_from_dict_without_bottle_id_raises(self):
        with pytest.raises(ValueError, match="bottle_id"):
            Correction.from_dict({"field_name": "name", "original_value": "A", "corrected_value": "B"})

    @pytest.mark.parametrize("bottle_id", ["", "   "])
    def test_empty_bottle_id_raises(self, bottle_id):
        with pytest.raises(ValueError, match="bottle_id"):
            _correction(bottle_id, "name", "A", "B")

    @pytest.mark.parametrize("source", ["", "web", "feedback:", "hint:", "llm:gpt-oss"])
    def test_invalid_source_raises(self, source):
        with pytest.raises(ValueError, match="source"):
            _correction("b", "name", "A", "B", source=source)

    def test_null_corrected_value_raises(self):
        with pytest.raises(ValueError, match="corrected_value"):
            _correction("b", "name", "A", None)

    @pytest.mark.parametrize("field_name", ["", "  "])
    def test_empty_field_name_raises(self, field_name):
        with pytest.raises(ValueError, match="field_name"):
            _correction("b", field_name, "A", "B")

    def test_three_positional_arguments_are_not_enough(self):
        with pytest.raises(TypeError):
            Correction("name", "A", "B")  # type: ignore[call-arg]
