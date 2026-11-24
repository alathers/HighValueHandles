"""Unit tests for lib.hints: loading, notes, sourced overrides, unused keys."""

from pathlib import Path

import pytest

from image_processor.corrections import CORRECTABLE_FIELDS
from lib.hints import HINT_FIELDS, NOTE_MAX_CHARS, HintBook, apply_hint_overrides
from lib.settings import SettingsError
from models.bottle import ExtractedMetadata

GOOD = """
[bottle.IMG_4643]
note = "Batch number is on the neck label in the third photo."
source = "Owner inspected the bottle, 2026-11-20"
[bottle.IMG_4643.fields]
batch_number = "C923"
proof = 124.6
release_year = 2023

[bottle.IMG_5000]
note = "Back label is upside down."
"""


def _book(tmp_path, text=GOOD, name="hints.toml") -> HintBook:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return HintBook.load(path)


def _load_errors(tmp_path, text) -> str:
    with pytest.raises(SettingsError) as excinfo:
        _book(tmp_path, text)
    return "\n".join(excinfo.value.errors)


class TestLoad:
    def test_load(self, tmp_path):
        book = _book(tmp_path)
        assert len(book) == 2
        assert book.note_for("IMG_4643") == "Batch number is on the neck label in the third photo."
        assert dict(book.fields_for("IMG_4643")) == {"batch_number": "C923", "proof": 124.6, "release_year": 2023}
        assert book.source_for("IMG_4643") == "Owner inspected the bottle, 2026-11-20"
        assert dict(book.fields_for("IMG_5000")) == {}
        assert book.note_for("IMG_9999") == ""

    def test_empty_book(self):
        assert HintBook.EMPTY.note_for("x") == ""
        assert dict(HintBook.EMPTY.fields_for("x")) == {}
        assert HintBook.EMPTY.unused(["x"]) == []

    def test_unknown_field(self, tmp_path):
        text = '[bottle.A]\nsource = "me"\n[bottle.A.fields]\nmsrp_guess = 40\n'
        assert "unknown field 'msrp_guess'" in _load_errors(tmp_path, text)

    def test_fields_without_source(self, tmp_path):
        text = '[bottle.A]\nnote = "x"\n[bottle.A.fields]\nbatch_number = "C1"\n'
        assert "'source' is required" in _load_errors(tmp_path, text)

    @pytest.mark.parametrize(
        ("field", "value", "needle"),
        [
            ("proof", '"124.6"', "proof must be a number"),
            ("abv", "150", "abv must be > 0"),
            ("msrp", "true", "msrp must be a number"),
            ("release_year", "23", "4-digit"),
            ("release_year", '"2023"', "4-digit"),
            ("name", '""', "non-empty string"),
            ("batch_number", "923", "non-empty string"),
        ],
    )
    def test_values_type_checked(self, tmp_path, field, value, needle):
        text = f'[bottle.A]\nsource = "me"\n[bottle.A.fields]\n{field} = {value}\n'
        assert needle in _load_errors(tmp_path, text)

    def test_bad_toml_and_unknown_keys(self, tmp_path):
        assert "invalid TOML" in _load_errors(tmp_path, "[bottle.A\n")
        assert "unknown key 'hint'" in _load_errors(tmp_path, '[bottle.A]\nhint = "x"\n')
        assert "unknown table [bottles]" in _load_errors(tmp_path, '[bottles.A]\nnote = "x"\n')

    def test_missing_file(self, tmp_path):
        with pytest.raises(SettingsError, match="hints file not found"):
            HintBook.load(tmp_path / "nope.toml")

    def test_errors_collected(self, tmp_path):
        text = '[bottle.A]\n[bottle.A.fields]\nproof = "x"\nfoo = 1\n'
        with pytest.raises(SettingsError) as excinfo:
            _book(tmp_path, text)
        assert len(excinfo.value.errors) == 3

    def test_hint_fields_are_correctable(self):
        assert set(HINT_FIELDS) <= CORRECTABLE_FIELDS


class TestNote:
    def test_note_capped_and_collapsed(self, tmp_path):
        book = _book(tmp_path, f'[bottle.A]\nnote = """{"word " * 300}\nmore"""\n')
        note = book.note_for("A")
        assert len(note) == NOTE_MAX_CHARS == 500
        assert "\n" not in note


class TestOverrides:
    @staticmethod
    def _bottles():
        bottle_dicts = [
            {"bottle_id": "IMG_4643", "name": "Stagg", "batch_number": "C922", "proof": "124.6 Proof"},
            {"bottle_id": "IMG_5000", "name": "Weller"},
        ]
        metadata = [
            ExtractedMetadata(name="Stagg", batch_number="C922", proof="124.6 Proof"),
            ExtractedMetadata(name="Weller"),
        ]
        return bottle_dicts, metadata

    def test_overrides_become_sourced_corrections_with_conflict_warning(self, tmp_path):
        book = _book(tmp_path)
        bottle_dicts, metadata = self._bottles()

        applied, unapplied = apply_hint_overrides(bottle_dicts, metadata, book, tmp_path / "hints.toml")

        assert unapplied == []
        assert {c.field_name for c in applied} == {"batch_number", "proof", "release_year"}
        assert bottle_dicts[0]["batch_number"] == "C923"
        assert bottle_dicts[0]["proof"] == "124.6 Proof"
        assert bottle_dicts[0]["release_year"] == "2023"
        recorded = {c["field_name"]: c for c in bottle_dicts[0]["corrections"]}
        assert recorded["batch_number"]["original_value"] == "C922"
        assert recorded["batch_number"]["bottle_id"] == "IMG_4643"
        assert recorded["batch_number"]["source"] == (
            "hint:hints.toml#IMG_4643: Owner inspected the bottle, 2026-11-20"
        )
        assert recorded["release_year"]["original_value"] == ""
        # The label read C922; the hint wins, and the disagreement is kept for review.
        assert "hint overrides label read for batch_number: label 'C922', hint 'C923'" in metadata[0].warnings
        # Same proof (124.6 == "124.6 Proof"): no conflict warning.
        assert not any("for proof" in w for w in metadata[0].warnings)
        # The other bottle is untouched.
        assert "corrections" not in bottle_dicts[1]
        assert metadata[1].warnings == []

    def test_no_hints_changes_nothing(self):
        bottle_dicts, metadata = self._bottles()
        assert apply_hint_overrides(bottle_dicts, metadata, HintBook.EMPTY) == ([], [])
        assert "corrections" not in bottle_dicts[0]

    def test_label_unread_gives_no_conflict(self, tmp_path):
        book = _book(tmp_path, '[bottle.A]\nsource = "me"\n[bottle.A.fields]\nage = "12 years"\n')
        bottle_dicts = [{"bottle_id": "A", "name": "X"}]
        metadata = [ExtractedMetadata(name="X")]
        apply_hint_overrides(bottle_dicts, metadata, book)
        assert bottle_dicts[0]["age"] == "12 years"
        assert metadata[0].warnings == []

    def test_unused_keys(self, tmp_path):
        book = _book(tmp_path)
        assert book.unused(["IMG_4643"]) == ["IMG_5000"]
        assert book.unused(["IMG_4643", "IMG_5000", "other"]) == []


def test_example_file_in_scratch_layout_loads(tmp_path):
    """The documented format (the example shipped for the owner) is accepted."""
    text = (
        '[bottle.IMG_4643]\nnote = "Batch number is on the neck label in the third photo."\n'
        'source = "Owner inspected the bottle, 2026-11-20"\n[bottle.IMG_4643.fields]\nbatch_number = "C923"\n'
    )
    book = _book(tmp_path, text)
    assert dict(book.fields_for("IMG_4643")) == {"batch_number": "C923"}
    assert isinstance(book.path, Path)
