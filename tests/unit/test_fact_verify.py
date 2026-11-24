"""Adversarial tests of image_processor.fact_verify (pure, no model)."""

from decimal import Decimal

import pytest

from image_processor.fact_prompt import Claim, NumberedField, PageMatch, PriceClaim, build_evidence_pack
from image_processor.fact_verify import (
    Accepted,
    AcceptedPrice,
    Rejected,
    label_release_tokens,
    normalize,
    ocr_backing,
    verify_claim,
    verify_copy,
    verify_price,
)
from models.evidence import WebEvidence
from models.facts import FactInput


def web(sid, text, tier="retailer", domain=None):
    return WebEvidence.create(
        text=text,
        source_id=sid,
        url=f"https://{domain or sid.lower() + '.example.com'}/p",
        domain=domain or f"{sid.lower()}.example.com",
        tier=tier,
        title=f"Page {sid}",
        fetched_at="2026-11-23T00:00:00Z",
        http_status=200,
        query="q",
        from_cache=False,
        truncated=False,
    )


def pack_of(*pages, label=None, ocr=None):
    inp = FactInput(
        bottle_id="IMG_1",
        label=label or {"name": "Elijah Craig Barrel Proof", "batch_number": "C923"},
        vision_confidence="high",
        images=["IMG_1.jpg"],
        ocr_by_image=ocr or {"IMG_1.jpg": "ELIJAH CRAIG\nBARREL PROOF\nBATCH C923\n133.2 PROOF"},
        web=list(pages),
    )
    return build_evidence_pack(inp)


EXACT = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C923")}
W1_TEXT = (
    "Elijah Craig Barrel Proof Batch C923 is bottled at 133.2 proof (66.6% ABV). "
    "Aged 12 years. Distilled by Heaven Hill Distillery. Bottled by Heaven Hill Brands. "
    "Released in 2023. MSRP of $59.99. Our price $64.99."
)


def claim(field, value, quote, sid="W1"):
    return Claim(field, value, sid, quote)


def rejected_reason(result):
    assert isinstance(result, Rejected), result
    return result.reason


class TestNormalize:
    def test_quotes_dashes_spaces_invisibles(self):
        raw = "“Booker’s” Batch–C923​\n\n soft­hyphen"
        assert normalize(raw) == "\"booker's\" batch-c923 softhyphen"

    def test_casefold_and_nfkc(self):
        assert normalize("ＢＡＴＣＨ  Straße") == "batch strasse"


class TestQuotes:
    def test_invented_quote_rejected(self):
        p = pack_of(web("W1", W1_TEXT))
        assert rejected_reason(verify_claim(claim("proof", "133.2", "bottled at 133.2 proof strength"), p, EXACT)) == "quote_not_found"

    def test_one_word_changed_rejected(self):
        p = pack_of(web("W1", W1_TEXT))
        result = verify_claim(claim("distillery", "Heaven Hill", "Distilled by Heaven Hill Distilling"), p, EXACT)
        assert rejected_reason(result) == "quote_not_found"

    def test_quote_only_in_other_source_rejected(self):
        p = pack_of(web("W1", W1_TEXT), web("W2", "Elijah Craig is distilled by Heaven Hill in Bardstown."))
        result = verify_claim(claim("distillery", "Heaven Hill", "distilled by Heaven Hill in Bardstown"), p, EXACT)
        assert rejected_reason(result) == "quote_in_other_source"
        assert "W2" in result.detail

    @pytest.mark.parametrize("sid", ["W9", "L1", ""])
    def test_unknown_source_rejected(self, sid):
        p = pack_of(web("W1", W1_TEXT))
        assert rejected_reason(verify_claim(claim("proof", "133.2", "bottled at 133.2 proof", sid), p, EXACT)) == "unknown_source"

    @pytest.mark.parametrize("quote", ["bottled at ... proof", "bottled at … proof"])
    def test_elided_quote_rejected(self, quote):
        p = pack_of(web("W1", W1_TEXT))
        assert rejected_reason(verify_claim(claim("proof", "133.2", quote), p, EXACT)) == "elided_quote"

    def test_three_char_quote_rejected(self):
        p = pack_of(web("W1", W1_TEXT))
        assert rejected_reason(verify_claim(claim("release_year", "2023", "202"), p, EXACT)) == "quote_too_short"

    def test_typography_and_line_breaks_accepted(self):
        text = "Booker’s Batch 2023–01 “Kentucky Tea” is bottled\nat 125.9 proof."
        p = pack_of(web("W1", text), label={"name": "Booker's", "batch_number": "2023-01"})
        pm = {"W1": PageMatch("W1", "exact", "Booker's Batch 2023-01 \"Kentucky Tea\"")}
        result = verify_claim(claim("proof", "125.9", "Booker's Batch 2023-01 \"Kentucky Tea\" is bottled at 125.9 proof"), p, pm)
        assert isinstance(result, Accepted), result
        assert result.value == pytest.approx(125.9)

    def test_url_in_value_rejected(self):
        p = pack_of(web("W1", W1_TEXT))
        result = verify_claim(claim("name", "https://evil.example/Elijah", "Elijah Craig Barrel Proof Batch C923"), p, EXACT)
        assert rejected_reason(result) == "url_in_output"


class TestValues:
    def test_proof_number_mismatch(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923, now at 115 proof."))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Batch C923")}
        assert rejected_reason(verify_claim(claim("proof", "116", "now at 115 proof"), p, pm)) == "number_mismatch"

    def test_proof_derived_from_abv(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923 is bottled at 57.5% ABV."))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Batch C923")}
        result = verify_claim(claim("proof", "115", "bottled at 57.5% ABV"), p, pm)
        assert isinstance(result, Accepted), result
        assert result.citation.derived is True
        assert result.value == 115.0

    def test_proof_needs_proof_word(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923 sold 115 cases this year."))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Batch C923")}
        assert rejected_reason(verify_claim(claim("proof", "115", "sold 115 cases this year"), p, pm)) == "unit_cue_missing"

    def test_batch_c923_is_not_c924(self):
        p = pack_of(web("W1", "Elijah Craig Barrel Proof Batch C924 is here."))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C924")}
        result = verify_claim(claim("batch_number", "C923", "Barrel Proof Batch C924 is here"), p, pm)
        assert rejected_reason(result) == "value_not_in_quote"

    def test_age_in_months(self):
        p = pack_of(web("W1", W1_TEXT))
        assert isinstance(verify_claim(claim("age", "12 years", "Batch C923 is bottled at 133.2 proof (66.6% ABV). Aged 12 years"), p, EXACT), Accepted)
        assert rejected_reason(verify_claim(claim("age", "10 years", "proof (66.6% ABV). Aged 12 years"), p, EXACT)) == "number_mismatch"

    def test_text_value_tokens_must_be_in_quote(self):
        p = pack_of(web("W1", W1_TEXT))
        ok = verify_claim(claim("bottler", "Heaven Hill Brands", "Bottled by Heaven Hill Brands"), p, EXACT)
        assert isinstance(ok, Accepted)
        bad = verify_claim(claim("bottler", "Beam Suntory", "Bottled by Heaven Hill Brands"), p, EXACT)
        assert rejected_reason(bad) == "value_not_in_quote"

    def test_role_cue_missing(self):
        p = pack_of(web("W1", W1_TEXT))
        result = verify_claim(claim("distillery", "Heaven Hill Brands", "Bottled by Heaven Hill Brands"), p, EXACT)
        assert rejected_reason(result) == "role_cue_missing"
        result = verify_claim(claim("bottler", "Heaven Hill", "Distilled by Heaven Hill Distillery"), p, EXACT)
        assert rejected_reason(result) == "role_cue_missing"


class TestPageMatch:
    def test_other_release_page_not_matched(self):
        p = pack_of(web("W1", W1_TEXT))
        pm = {"W1": PageMatch("W1", "same_product_other_release", "Elijah Craig Barrel Proof Batch C923")}
        result = verify_claim(claim("proof", "133.2", "bottled at 133.2 proof"), p, pm)
        assert rejected_reason(result) == "page_not_matched"

    def test_page_match_quote_must_be_on_page(self):
        p = pack_of(web("W1", W1_TEXT))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C923 2023 release notes")}
        assert rejected_reason(verify_claim(claim("proof", "133.2", "bottled at 133.2 proof"), p, pm)) == "page_not_matched"

    def test_missing_verdict_not_matched(self):
        p = pack_of(web("W1", W1_TEXT))
        assert rejected_reason(verify_claim(claim("proof", "133.2", "bottled at 133.2 proof"), p, {})) == "page_not_matched"

    def test_release_token_far_away(self):
        text = "Elijah Craig Barrel Proof Batch C923 overview. " + ("filler words here. " * 120) + "It is bottled at 140.2 proof."
        p = pack_of(web("W1", text))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C923")}
        result = verify_claim(claim("proof", "140.2", "It is bottled at 140.2 proof"), p, pm)
        assert rejected_reason(result) == "release_not_near_quote"

    def test_release_year_far_away(self):
        label = {"name": "Old Forester Birthday Bourbon 2023"}
        text = "Old Forester Birthday Bourbon 2023 arrives. " + ("x" * 2000) + " The 12 year old whiskey was released in 2019."
        p = pack_of(web("W1", text), label=label)
        pm = {"W1": PageMatch("W1", "exact", "Old Forester Birthday Bourbon 2023 arrives")}
        result = verify_claim(claim("release_year", "2019", "whiskey was released in 2019"), p, pm)
        assert rejected_reason(result) == "release_not_near_quote"

    def test_non_batch_field_does_not_need_page_match(self):
        p = pack_of(web("W1", W1_TEXT))
        result = verify_claim(claim("bottler", "Heaven Hill Brands", "Bottled by Heaven Hill Brands"), p, {})
        assert isinstance(result, Accepted)

    def test_label_release_tokens(self):
        tokens = label_release_tokens({"name": "Birthday Bourbon 2023", "specific_release": "Fall 2023", "batch_number": "Batch C923"})
        assert tokens == ["2023", "c923"]


class TestPrices:
    def test_retail_price_called_msrp_is_downgraded(self):
        p = pack_of(web("W1", "Elijah Craig Barrel Proof Batch C923. Our price $64.99. Sale!", tier="retailer"))
        result = verify_price(PriceClaim("msrp", Decimal("64.99"), "USD", "W1", "Our price $64.99"), p, EXACT)
        assert isinstance(result, AcceptedPrice)
        assert result.kind == "retail"
        assert result.downgraded

    def test_msrp_without_cue_on_reference_rejected(self):
        p = pack_of(web("W1", "Elijah Craig Barrel Proof Batch C923 costs about $64.99 today.", tier="reference"))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C923")}
        result = verify_price(PriceClaim("msrp", Decimal("64.99"), "USD", "W1", "costs about $64.99 today"), p, pm)
        assert rejected_reason(result) == "msrp_cue_missing"

    def test_msrp_accepted(self):
        p = pack_of(web("W1", W1_TEXT, tier="official"))
        result = verify_price(PriceClaim("msrp", Decimal("59.99"), "USD", "W1", "MSRP of $59.99"), p, EXACT)
        assert isinstance(result, AcceptedPrice)
        assert result.kind == "msrp"

    def test_amount_off_by_a_cent(self):
        p = pack_of(web("W1", W1_TEXT, tier="official"))
        result = verify_price(PriceClaim("msrp", Decimal("59.98"), "USD", "W1", "MSRP of $59.99"), p, EXACT)
        assert rejected_reason(result) == "amount_mismatch"

    def test_msrp_needs_exact_page(self):
        p = pack_of(web("W1", W1_TEXT, tier="official"))
        pm = {"W1": PageMatch("W1", "unclear", None)}
        assert rejected_reason(verify_price(PriceClaim("msrp", Decimal("59.99"), "USD", "W1", "MSRP of $59.99"), p, pm)) == "page_not_matched"

    def test_amount_range(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923 MSRP $0.00 here"))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Batch C923")}
        assert rejected_reason(verify_price(PriceClaim("msrp", Decimal("0"), "USD", "W1", "Batch C923 MSRP $0.00 here"), p, pm)) == "amount_out_of_range"

    def test_secondary_needs_cue(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923 sold for $310 at auction. Also listed $250 elsewhere."))
        pm = {"W1": PageMatch("W1", "exact", "Elijah Craig Batch C923")}
        ok = verify_price(PriceClaim("secondary", Decimal("310"), "USD", "W1", "sold for $310 at auction"), p, pm)
        assert isinstance(ok, AcceptedPrice) and ok.kind == "secondary"

    def test_secondary_without_cue_rejected(self):
        p = pack_of(web("W1", "Elijah Craig Batch C923. " + "y" * 300 + " Listed $250 elsewhere."))
        assert rejected_reason(verify_price(PriceClaim("secondary", Decimal("250"), "USD", "W1", "Listed $250 elsewhere"), p, {})) == "secondary_cue_missing"

    def test_different_product_price_rejected(self):
        p = pack_of(web("W1", "Some other whiskey. Our price $29.99."))
        pm = {"W1": PageMatch("W1", "different_product", None)}
        assert rejected_reason(verify_price(PriceClaim("retail", Decimal("29.99"), "USD", "W1", "Our price $29.99"), p, pm)) == "page_not_matched"


class TestOcrBacking:
    def test_found(self):
        cit = ocr_backing("133.2", "proof", {"O1": "ELIJAH CRAIG\nBATCH C923\n133.2 PROOF"})
        assert cit is not None
        assert cit.source_id == "O1"
        assert cit.quote == "133.2 PROOF"

    def test_text_field_found(self):
        cit = ocr_backing("Elijah Craig", "name", {"O1": "ELIJAH CRAIG\nBATCH C923"})
        assert cit is not None and cit.quote == "ELIJAH CRAIG"

    def test_not_found(self):
        assert ocr_backing("134.2", "proof", {"O1": "133.2 PROOF"}) is None
        assert ocr_backing("C924", "batch_number", {"O1": "BATCH C923"}) is None
        assert ocr_backing("133.2", "proof", {"O1": "BATCH 133.2"}) is None


def numbered():
    return [
        NumberedField(1, "name", "Elijah Craig Barrel Proof", "supported", "label"),
        NumberedField(2, "proof", 133.2, "supported", "label"),
        NumberedField(3, "age", "12 years", "supported", "W1"),
    ]


class TestVerifyCopy:
    def test_good_copy(self):
        sentences = [
            {"text": "Elijah Craig Barrel Proof is bottled at 133.2 proof (66.6% ABV).", "citations": [1, 2], "fields": ["name", "proof"]},
            {"text": "It is aged 12 years.", "citations": [3], "fields": ["age"]},
        ]
        assert verify_copy(sentences, numbered()).ok

    def test_uncited_number(self):
        s = [{"text": "It is bottled at 140 proof.", "citations": [2], "fields": ["proof"]}]
        check = verify_copy(s, numbered())
        assert not check.ok and check.failures[0]["reason"] == "number_not_cited"

    def test_no_citation(self):
        check = verify_copy([{"text": "A bourbon.", "citations": [], "fields": ["name"]}], numbered())
        assert any(f["reason"] == "no_citation" for f in check.failures)

    def test_bad_index(self):
        check = verify_copy([{"text": "A bourbon.", "citations": [9], "fields": ["name"]}], numbered())
        assert any(f["reason"] == "bad_citation" for f in check.failures)

    def test_unverified_field(self):
        check = verify_copy([{"text": "Elijah Craig Barrel Proof.", "citations": [1], "fields": ["name", "msrp"]}], numbered())
        assert any(f["reason"] == "unverified_field" for f in check.failures)

    def test_superlative(self):
        check = verify_copy([{"text": "The best Elijah Craig Barrel Proof.", "citations": [1], "fields": ["name"]}], numbered())
        assert any(f["reason"] == "unverifiable_claim" for f in check.failures)

    def test_banned_word_inside_cited_value_allowed(self):
        nf = [NumberedField(1, "name", "Limited Edition Small Batch", "supported", "label")]
        assert verify_copy([{"text": "This is the Limited Edition Small Batch.", "citations": [1], "fields": ["name"]}], nf).ok

    def test_url(self):
        check = verify_copy([{"text": "See www.example.com for Elijah Craig.", "citations": [1], "fields": ["name"]}], numbered())
        assert any(f["reason"] == "url_in_output" for f in check.failures)

    def test_one_bad_sentence_fails_all(self):
        sentences = [
            {"text": "Elijah Craig Barrel Proof is bottled at 133.2 proof.", "citations": [1, 2], "fields": ["name", "proof"]},
            {"text": "It is a legendary pour.", "citations": [1], "fields": ["name"]},
        ]
        check = verify_copy(sentences, numbered())
        assert not check.ok
        assert "sentence 2" in check.reason


class TestReviewHardening:
    """Holes found in review: a number or year merely present in the quote is not enough."""

    PM = {"W1": PageMatch("W1", "exact", "Elijah Craig Barrel Proof Batch C923")}

    def _pack(self, text, tier="retailer"):
        return pack_of(web("W1", "Elijah Craig Barrel Proof Batch C923. " + text, tier=tier))

    def test_proof_must_touch_its_unit(self):
        p = self._pack("Bottled at 133.2 proof, aged 12 years.")
        result = verify_claim(claim("proof", "12", "Bottled at 133.2 proof, aged 12 years"), p, self.PM)
        assert rejected_reason(result) == "unit_cue_missing"

    def test_abv_must_touch_its_unit(self):
        p = self._pack("It is 66.6% ABV, aged 12 years.")
        result = verify_claim(claim("abv", "12", "It is 66.6% ABV, aged 12 years"), p, self.PM)
        assert rejected_reason(result) == "unit_cue_missing"

    def test_barrel_proof_age_is_not_proof(self):
        p = self._pack("Barrel Proof 12 Year Old release.")
        result = verify_claim(claim("proof", "12", "Barrel Proof 12 Year Old release"), p, self.PM)
        assert rejected_reason(result) == "unit_cue_missing"

    @pytest.mark.parametrize(
        "field,value,quote",
        [
            ("proof", "115", "Proof: 115 for this batch"),
            ("proof", "115", "a 115-proof whiskey"),
            ("proof", "115", "bottled at 115 proof"),
            ("abv", "57.5", "ABV 57.5 for this batch"),
            ("abv", "57.5", "Alc. 57.5% by vol."),
        ],
    )
    def test_unit_forms_accepted(self, field, value, quote):
        p = self._pack(quote)
        assert isinstance(verify_claim(claim(field, value, quote), p, self.PM), Accepted)

    def test_founding_year_is_not_release_year(self):
        p = self._pack("Founded in 1935 by the family.")
        result = verify_claim(claim("release_year", "1935", "Founded in 1935 by the family"), p, self.PM)
        assert rejected_reason(result) == "number_mismatch"
        p = self._pack("Released in 2023 by the family.")
        assert isinstance(verify_claim(claim("release_year", "2023", "Released in 2023 by the family"), p, self.PM), Accepted)

    def test_any_claim_on_different_product_page_rejected(self):
        p = self._pack("Weller 12 Year is distilled by Buffalo Trace Distillery.")
        pm = {"W1": PageMatch("W1", "different_product", None)}
        result = verify_claim(claim("distillery", "Buffalo Trace", "distilled by Buffalo Trace Distillery"), p, pm)
        assert rejected_reason(result) == "page_not_matched"

    def test_msrp_cue_binds_to_nearest_amount(self):
        text = "MSRP $59.99. Our price $64.99."
        p = self._pack(text, tier="official")
        wrong = verify_price(PriceClaim("msrp", Decimal("64.99"), "USD", "W1", "Our price $64.99"), p, self.PM)
        assert rejected_reason(wrong) == "msrp_cue_missing"
        wrong_in_quote = verify_price(PriceClaim("msrp", Decimal("64.99"), "USD", "W1", text), p, self.PM)
        assert rejected_reason(wrong_in_quote) == "msrp_cue_missing"
        right = verify_price(PriceClaim("msrp", Decimal("59.99"), "USD", "W1", "MSRP $59.99"), p, self.PM)
        assert isinstance(right, AcceptedPrice) and right.kind == "msrp"

    def test_retailer_price_next_to_other_msrp_is_downgraded(self):
        p = self._pack("MSRP $59.99. Our price $64.99.", tier="retailer")
        result = verify_price(PriceClaim("msrp", Decimal("64.99"), "USD", "W1", "Our price $64.99"), p, self.PM)
        assert isinstance(result, AcceptedPrice) and result.kind == "retail" and result.downgraded

    def test_amount_before_cue(self):
        p = self._pack("Expect $59.99 MSRP at launch.", tier="official")
        result = verify_price(PriceClaim("msrp", Decimal("59.99"), "USD", "W1", "Expect $59.99 MSRP at launch"), p, self.PM)
        assert isinstance(result, AcceptedPrice) and result.kind == "msrp"

    def test_ocr_backing_needs_number_next_to_unit(self):
        assert ocr_backing("12", "proof", {"O1": "AGED 12 YEARS 133.2 PROOF"}) is None
        assert ocr_backing("133.2", "proof", {"O1": "AGED 12 YEARS 133.2 PROOF"}) is not None


class TestUnsupportedName:
    def test_name_not_in_cited_fields_fails(self):
        s = [{"text": "Elijah Craig Barrel Proof comes from Bardstown Kentucky.", "citations": [1], "fields": ["name"]}]
        check = verify_copy(s, numbered())
        assert any(f["reason"] == "unsupported_name" for f in check.failures)

    def test_cited_name_passes(self):
        s = [{"text": "The bottle is labeled Elijah Craig Barrel Proof.", "citations": [1], "fields": ["name"]}]
        assert verify_copy(s, numbered()).ok

    def test_name_in_provenance_quote_passes(self):
        nf = [NumberedField(1, "name", "Weller Special Reserve", "supported", 'OCR of IMG_1.jpg "KENTUCKY STRAIGHT BOURBON"')]
        s = [{"text": "Weller Special Reserve is a Kentucky Straight Bourbon.", "citations": [1], "fields": ["name"]}]
        assert verify_copy(s, nf).ok
