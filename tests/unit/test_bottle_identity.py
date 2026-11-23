"""Unit tests for image_processor.bottle_identity (tolerant same-bottle comparison)."""

import pytest

from image_processor.bottle_identity import (
    identity_conflicts,
    names_match,
    normalize_text,
    token_set_ratio,
)
from models.bottle import ExtractedMetadata

STAGG = {"name": "George T. Stagg", "distillery": "Buffalo Trace", "proof": "135.4 Proof"}


def meta(**fields):
    return ExtractedMetadata(**fields)


class TestNormalisation:
    def test_case_accents_punctuation(self):
        assert normalize_text("  Blanton's  Single-Barrel ") == "blantons single barrel"
        assert normalize_text("Château") == "chateau"
        assert normalize_text("Blanton\u2019s") == "blantons"  # curly apostrophe
        assert normalize_text(None) == ""

    def test_token_set_ratio(self):
        assert token_set_ratio("weller special reserve", "reserve special weller") == 1.0
        assert token_set_ratio("blantons", "blantons single barrel") == 1.0  # subset
        assert token_set_ratio("weller special reserve", "weller full proof") < 0.9
        assert token_set_ratio("", "x") == 0.0


class TestNamesMatch:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("George T. Stagg", "GEORGE T STAGG"),
            ("Blanton's", "Blantons Single Barrel"),
            ("E.H. Taylor, Jr. Small Batch", "EH Taylor Jr Small Batch Bourbon"),
            ("Stagg Jr", "Stagg Jr 2023"),  # number on one side only: wildcard
            ("1792 Full Proof", "1792 Full Proof 125"),  # extra number on one view only
            ("Old Fitzgerald Bottled in Bond 8", "Old Fitzgerald 8 Year Old Bottled in Bond 100 Proof"),
            ("Elmer T. Lee", None),
            ("", "Anything"),
        ],
    )
    def test_match(self, a, b):
        assert names_match(a, b)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("George T. Stagg 23A", "George T. Stagg 23C"),  # token-set ratio alone is ~0.93
            ("Old Fitzgerald 8", "Old Fitzgerald 10"),
            ("Old Fitzgerald 8 Year 100 Proof", "Old Fitzgerald 10 Year 100 Proof"),  # shared 100, 8 vs 10
            ("Eagle Rare 10", "Eagle Rare 17"),
            ("Weller Special Reserve", "Weller Full Proof"),
            ("Eagle Rare", "Blanton's"),
        ],
    )
    def test_no_match(self, a, b):
        assert not names_match(a, b)


class TestIdentityConflicts:
    def test_stagg_batches_stay_apart(self):
        """Stagg 23A vs 23C: same name and proof, batch read on both sides differs."""
        a = meta(**STAGG, batch_number="23A")
        b = meta(**STAGG, batch_number="Batch 23C")
        assert identity_conflicts(a, b) == ["batch_number"]

    def test_batch_missing_on_one_side_is_a_wildcard(self):
        assert identity_conflicts(meta(**STAGG, batch_number="23A"), meta(**STAGG)) == []

    def test_batch_spelling_variants_match(self):
        assert identity_conflicts(meta(**STAGG, batch_number="Batch No. 23-A"), meta(**STAGG, batch_number="23a")) == []

    def test_proof_tolerance(self):
        assert identity_conflicts(meta(name="X", proof="135.4 Proof"), meta(name="X", proof="135")) == []
        assert identity_conflicts(meta(name="X", proof="135.4"), meta(name="X", proof="130.0")) == ["proof"]
        assert identity_conflicts(meta(name="X", proof="135.4"), meta(name="X")) == []

    def test_age_in_years_or_months(self):
        assert identity_conflicts(meta(name="X", age="8 Years Old"), meta(name="X", age="96 months")) == []
        assert identity_conflicts(meta(name="X", age="8 Year"), meta(name="X", age="10 Year")) == ["age"]

    def test_release_compared_like_names(self):
        a = meta(name="Old Forester", specific_release="1920 Prohibition Style")
        b = meta(name="Old Forester", specific_release="1910 Old Fine Whisky")
        assert identity_conflicts(a, b) == ["specific_release"]
        assert identity_conflicts(a, meta(name="Old Forester")) == []

    def test_name_missing_on_one_side(self):
        assert identity_conflicts(meta(distillery="Buffalo Trace", proof="90"), meta(name="Weller", proof="90 Proof")) == []

    def test_several_conflicts_listed(self):
        a = meta(name="Weller Special Reserve", proof="90")
        b = meta(name="Weller Full Proof", proof="114")
        assert identity_conflicts(a, b) == ["name", "proof"]

    def test_number_seen_on_one_view_only_is_not_a_conflict(self):
        """Front reads '1792 Full Proof', back adds the proof to the name: one bottle."""
        a = meta(name="1792 Full Proof", distillery="Barton 1792")
        b = meta(name="1792 Full Proof 125", proof="125 Proof")
        assert identity_conflicts(a, b) == []

    def test_subset_name_matches_documented_limitation(self):
        """Documented limitation: a subset name matches; only another field read on both sides separates."""
        assert identity_conflicts(meta(name="Stagg"), meta(name="Stagg Jr")) == []
        assert identity_conflicts(meta(name="Stagg", proof="130"), meta(name="Stagg Jr", proof="128")) == ["proof"]

    def test_sibling_without_distinguishing_field_is_merged(self):
        """Documented limitation: identical-label siblings compare equal."""
        assert identity_conflicts(meta(name="Elmer T. Lee", proof="90"), meta(name="Elmer T. Lee", proof="90 Proof")) == []
