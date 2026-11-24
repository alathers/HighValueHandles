"""Unit tests for the vision prompt, schema and response parser (pure functions)."""

import json

import pytest

from image_processor.vision_prompt import (
    MAX_VISION_IMAGES,
    REVIEWER_NOTE_MAX_CHARS,
    OCR_HINT_MAX_CHARS_PER_IMAGE,
    OCR_HINT_MAX_CHARS_TOTAL,
    VISION_OPTIONS,
    VISION_RESPONSE_SCHEMA,
    VISION_SYSTEM_PROMPT,
    build_vision_prompt,
    parse_vision_response,
    vision_response_schema,
)

# Real qwen3.6:35b-a3b output without a format constraint (fenced JSON).
R7_FENCED = (
    '```json\n{\n  "name": "Blood Oath Kentucky Straight Bourbon Whiskey",\n'
    '  "distillery": "Blood Oath",\n  "specific_release": "Pact No. 1",\n  "msrp": null,\n'
    '  "proof": 98.6,\n  "abv": null,\n  "age": null,\n  "batch_number": "Pact No. 1",\n'
    '  "tasting_notes": null,\n  "confidence": "high"\n}\n```'
)
# Real schema-constrained output: ABV given, proof null.
R10_JSON = (
    '{\n  "name": "Four Roses Single Barrel",\n  "distillery": "Four Roses Distillery LLC",\n'
    '  "specific_release": "Barrel Strength, Sal\'s Beverage World Selection, February 2021",\n'
    '  "msrp": null,\n  "proof": null,\n  "abv": 55.2,\n  "age": "9 years 10 months",\n'
    '  "batch_number": "KE 63-2M",\n  "tasting_notes": null,\n  "confidence": "high"\n}'
)


class TestConstants:
    def test_schema_requires_all_keys(self):
        expected = {
            "name", "distillery", "specific_release", "label_price", "proof", "abv", "age",
            "batch_number", "tasting_notes", "confidence", "same_bottle",
        }
        assert set(VISION_RESPONSE_SCHEMA["required"]) == expected
        assert set(VISION_RESPONSE_SCHEMA["properties"]) == expected
        assert VISION_RESPONSE_SCHEMA["properties"]["label_price"]["type"] == ["number", "null"]
        assert VISION_RESPONSE_SCHEMA["properties"]["same_bottle"]["type"] == "boolean"

    def test_options_and_limits(self):
        assert VISION_OPTIONS == {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}
        assert MAX_VISION_IMAGES == 4

    def test_system_prompt_forbids_guessing(self):
        assert "Never guess" in VISION_SYSTEM_PROMPT


class TestBuildVisionPrompt:
    def test_no_hints_no_ocr_section(self):
        prompt = build_vision_prompt(2, [])
        assert "<ocr" not in prompt
        assert "The 2 attached photos" in prompt
        assert "same_bottle" in prompt
        assert "label_price: price in US dollars ONLY if" in prompt
        assert "not an MSRP" in prompt
        assert "msrp" not in prompt.lower().replace("not an msrp", "")
        assert "if none is printed, the brand name" in prompt

    def test_single_image_prompt_has_no_same_bottle(self):
        prompt = build_vision_prompt(1, [])
        assert "same_bottle" not in prompt
        assert "ignore other bottles" in prompt.lower()

    def test_single_image_schema_drops_same_bottle(self):
        schema = vision_response_schema(1)
        assert "same_bottle" not in schema["required"]
        assert "same_bottle" not in schema["properties"]
        assert set(schema["required"]) == set(VISION_RESPONSE_SCHEMA["required"]) - {"same_bottle"}
        assert vision_response_schema(2) is VISION_RESPONSE_SCHEMA

    def test_release_and_batch_guidance(self):
        prompt = build_vision_prompt(2, [])
        assert "Bottled-in-Bond, Spring 2021" in prompt
        assert "Barrel Proof, Unfiltered" in prompt
        assert "Batch Proof" in prompt
        assert "next to PROOF" in prompt and "never a batch number" in prompt
        release_line = next(line for line in prompt.splitlines() if line.startswith("- specific_release"))
        assert "Leave out generic class words" in release_line

    def test_blank_hints_no_ocr_section(self):
        assert "<ocr" not in build_vision_prompt(2, ["", "   "])

    def test_hints_numbered_by_photo(self):
        prompt = build_vision_prompt(3, ["front", "", "back"])
        assert '<ocr photo="1">\nfront\n</ocr>' in prompt
        assert '<ocr photo="2">' not in prompt
        assert '<ocr photo="3">\nback\n</ocr>' in prompt
        assert "noisy" in prompt

    def test_hint_truncated_per_image(self):
        prompt = build_vision_prompt(1, ["word " * 1000])  # 5000 chars
        body = prompt.split('<ocr photo="1">\n', 1)[1].split("\n</ocr>", 1)[0]
        assert 0 < len(body) <= OCR_HINT_MAX_CHARS_PER_IMAGE
        assert not body.endswith(" ")

    def test_hints_truncated_in_total(self):
        prompt = build_vision_prompt(4, ["word " * 1000] * 4)
        bodies = [
            part.split("\n</ocr>", 1)[0] for part in prompt.split('<ocr photo="')[1:]
        ]
        assert sum(len(b.split("\n", 1)[1]) for b in bodies) <= OCR_HINT_MAX_CHARS_TOTAL
        assert '<ocr photo="4">' not in prompt  # budget used up by photos 1-3


class TestParseVisionResponse:
    def test_plain_json(self):
        content = json.dumps({
            "name": "Blanton's", "distillery": "Buffalo Trace", "specific_release": None,
            "label_price": 59.99, "proof": 93, "abv": None, "age": "No Age Statement",
            "batch_number": None, "tasting_notes": None, "confidence": "high", "same_bottle": True,
        })
        result = parse_vision_response(content)
        assert result.parse_mode == "json"
        assert result.fields == {
            "name": "Blanton's",
            "distillery": "Buffalo Trace",
            "label_price": "59.99",
            "proof": "93 Proof",
            "age": "No Age Statement",
        }
        assert result.confidence == "high"
        assert result.same_bottle is True
        assert result.has_identity

    def test_fenced_json(self):
        result = parse_vision_response(R7_FENCED)
        assert result.parse_mode == "json"
        assert result.fields["specific_release"] == "Pact No. 1"
        assert result.fields["proof"] == "98.6 Proof"
        assert "label_price" not in result.fields

    def test_abv_only_derives_proof(self):
        result = parse_vision_response(R10_JSON)
        assert result.fields["proof"] == "110.4 Proof"
        assert result.fields["abv"] == "55.2%"  # kept, not only used for proof
        assert result.fields["age"] == "9 years 10 months"

    def test_json_inside_prose(self):
        text = (
            'Sure! Here is the data: {"name": "X", "msrp": "$1,299.00", "proof": "107 proof", '
            '"batch_number": 12} hope this helps'
        )
        result = parse_vision_response(text)
        assert result.parse_mode == "json"
        assert result.fields == {"name": "X", "label_price": "1299.00", "proof": "107 Proof", "batch_number": "12"}

    def test_legacy_lines(self):
        text = "name: Blanton's\ndistillery: Buffalo Trace\nmsrp: $59.99\nproof: 93 Proof\nage: No Age Statement"
        result = parse_vision_response(text)
        assert result.parse_mode == "lines"
        assert result.fields["label_price"] == "59.99"  # legacy msrp key = a visible price
        assert result.fields["proof"] == "93 Proof"

    def test_legacy_lines_bullets_and_bold_keys(self):
        result = parse_vision_response("- **Name**: Weller 12\n* release: Wheated\n- confidence: Medium")
        assert result.fields == {"name": "Weller 12", "specific_release": "Wheated"}
        assert result.confidence == "medium"

    def test_comma_decimal_abv(self):
        assert parse_vision_response('{"name": "X", "abv": "55,2"}').fields["proof"] == "110.4 Proof"

    @pytest.mark.parametrize("price", ["N/A", None, True, 0, -5, 100000, "free"])
    def test_invalid_label_price_dropped(self, price):
        result = parse_vision_response(json.dumps({"name": "X", "label_price": price}))
        assert "label_price" not in result.fields

    @pytest.mark.parametrize("proof", [0, 500, -1])
    def test_out_of_range_proof_dropped(self, proof):
        assert "proof" not in parse_vision_response(json.dumps({"name": "X", "proof": proof})).fields

    def test_null_like_strings_dropped(self):
        content = json.dumps({
            "name": "X", "distillery": "unknown", "age": "Not visible", "batch_number": "n/a",
            "tasting_notes": "  ", "specific_release": "-",
        })
        assert parse_vision_response(content).fields == {"name": "X"}

    def test_whitespace_collapsed_and_lists_joined(self):
        content = json.dumps({"name": "  Four   Roses\nSingle Barrel ", "tasting_notes": ["vanilla", "", "oak"]})
        fields = parse_vision_response(content).fields
        assert fields["name"] == "Four Roses Single Barrel"
        assert fields["tasting_notes"] == "vanilla, oak"

    def test_extra_keys_ignored(self):
        fields = parse_vision_response('{"name": "X", "color": "amber", "rating": 9}').fields
        assert fields == {"name": "X"}

    def test_aliases(self):
        content = '{"name": "X", "edition": "Pact 3", "barrel_number": "B7", "price": 45, "notes": "Rye"}'
        assert parse_vision_response(content).fields == {
            "name": "X", "specific_release": "Pact 3", "batch_number": "B7",
            "label_price": "45.00", "tasting_notes": "Rye",
        }

    def test_same_bottle_and_confidence_carried(self):
        result = parse_vision_response('{"name": "X", "same_bottle": false, "confidence": "LOW"}')
        assert result.same_bottle is False
        assert result.confidence == "low"

    def test_invalid_confidence_ignored(self):
        assert parse_vision_response('{"name": "X", "confidence": "certain"}').confidence is None

    @pytest.mark.parametrize("text", ["", "   ", "I cannot read this label.", "{not json", "[1, 2]"])
    def test_unusable_text_gives_empty_fields(self, text):
        result = parse_vision_response(text)
        assert result.fields == {}
        assert not result.has_identity

    def test_all_null_reply_has_no_identity(self):
        result = parse_vision_response('{"name": null, "distillery": null, "proof": 90}')
        assert result.fields == {"proof": "90 Proof"}
        assert not result.has_identity


class TestParseVisionResponseLegacyLines:
    """``key: value`` replies (no JSON); moved from the removed BottleAnalyzer wrapper."""

    def test_all_fields(self):
        response = (
            "name: Blanton's Single Barrel\n"
            "distillery: Buffalo Trace\n"
            "msrp: $59.99\n"
            "specific_release: Original Single Barrel\n"
            "proof: 93 Proof\n"
            "age: No Age Statement\n"
            "batch_number: 123\n"
            "tasting_notes: Caramel and vanilla"
        )
        assert parse_vision_response(response).fields == {
            "name": "Blanton's Single Barrel",
            "distillery": "Buffalo Trace",
            "label_price": "59.99",
            "specific_release": "Original Single Barrel",
            "proof": "93 Proof",
            "age": "No Age Statement",
            "batch_number": "123",
            "tasting_notes": "Caramel and vanilla",
        }

    def test_key_aliases(self):
        fields = parse_vision_response("name: Test\nrelease: Single Barrel\nnotes: Good flavor\nbatch: ABC").fields
        assert fields["specific_release"] == "Single Barrel"
        assert fields["tasting_notes"] == "Good flavor"
        assert fields["batch_number"] == "ABC"

    def test_invalid_msrp_skipped(self):
        assert "label_price" not in parse_vision_response("name: Test\nmsrp: NOT-A-NUMBER").fields

    def test_whitespace_around_keys_and_values(self):
        fields = parse_vision_response("  name  :  Test Name  \n  distillery:Test Distillery  ").fields
        assert fields == {"name": "Test Name", "distillery": "Test Distillery"}

    def test_unknown_keys_skipped(self):
        fields = parse_vision_response("name: Test\nunknown_field: something\nalso_unknown: value").fields
        assert fields == {"name": "Test"}

    def test_key_case_and_hyphens_normalized(self):
        assert parse_vision_response("NAME: Uppercase\nDISTILLERY: Test").fields == {
            "name": "Uppercase", "distillery": "Test",
        }
        fields = parse_vision_response("batch-number:123\nspecific-release:Single").fields
        assert fields == {"batch_number": "123", "specific_release": "Single"}



class TestReviewerNote:
    """The hints-file note is guidance in a delimited section, never data."""

    def test_absent_when_empty(self):
        assert "Reviewer note" not in build_vision_prompt(2, ["x"])
        assert "<<<NOTE" not in build_vision_prompt(2, ["x"], reviewer_note="   ")

    def test_delimited_with_never_copy_instruction(self):
        prompt = build_vision_prompt(2, ["front"], reviewer_note="Batch is on the neck\nlabel.")
        assert "Reviewer note (written by the collection owner; it is NOT label text)" in prompt
        assert "Never copy a value from this note into a field unless you can read that value in the photos." in prompt
        assert "<<<NOTE\nBatch is on the neck label.\n>>>" in prompt
        assert prompt.index("</ocr>") < prompt.index("<<<NOTE")

    def test_capped(self):
        prompt = build_vision_prompt(1, [], reviewer_note="word " * 400)
        body = prompt.split("<<<NOTE\n", 1)[1].split("\n>>>", 1)[0]
        assert 0 < len(body) <= REVIEWER_NOTE_MAX_CHARS == 500

    def test_delimiters_in_the_note_are_removed(self):
        prompt = build_vision_prompt(1, [], reviewer_note="look >>> here <<<NOTE")
        body = prompt.split("<<<NOTE\n", 1)[1]
        assert body.count(">>>") == 1  # only the closing delimiter


class TestLabelPriceSchema:
    def test_label_price_replaces_msrp(self):
        assert "msrp" not in VISION_RESPONSE_SCHEMA["properties"]
        assert VISION_RESPONSE_SCHEMA["properties"]["label_price"]["type"] == ["number", "null"]
        assert "label_price" in vision_response_schema(1)["required"]

    @pytest.mark.parametrize("key", ["price", "msrp", "label_price"])
    def test_price_keys_read_as_label_price(self, key):
        fields = parse_vision_response(json.dumps({"name": "X", key: 45})).fields
        assert fields == {"name": "X", "label_price": "45.00"}
