"""Tests for lib.html_text (stdlib HTML to text + JSON-LD)."""

from decimal import Decimal
from pathlib import Path

from lib.html_text import (
    MAX_TEXT_CHARS,
    ExtractedHtml,
    drop_partial_tail,
    extract,
    jsonld_products,
    offer_prices,
    product_images,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_script_style_noscript_svg_are_stripped():
    page = extract(_fixture("product_official.html"))
    assert "tracking" not in page.text
    assert "font-family" not in page.text
    assert "Enable JavaScript" not in page.text
    assert "SVG TEXT" not in page.text
    assert "logo" not in page.text
    assert "Bottled at 94 proof" in page.text


def test_title_og_image_canonical_and_description():
    page = extract(_fixture("product_official.html"))
    assert page.title == "Example Creek Small Batch Bourbon | Example Creek Distillery"
    assert page.og_image == "/media/example-creek-small-batch.jpg"
    assert page.canonical == "https://www.examplecreekdistillery.com/bourbon/small-batch"
    assert page.meta_description == "Example Creek Small Batch Kentucky Straight Bourbon."
    # The title is a separate field, not body text.
    assert "| Example Creek Distillery" not in page.text


def test_entities_are_unescaped():
    page = extract(_fixture("product_official.html"))
    assert "Tom & Jerry's favourite — since 1889." in page.text
    assert "© Example Creek Distillery" in page.text


def test_block_tags_become_lines_and_whitespace_collapses():
    page = extract("<div>one   two\n\t three</div><p>four</p>a<br>b<ul><li>x</li><li>y</li></ul>")
    assert page.text.split("\n") == ["one two three", "four", "a", "b", "x", "y"]


def test_inline_tags_do_not_break_lines():
    assert (
        extract("<p>Example <b>Creek</b> <i>Small</i> Batch</p>").text
        == "Example Creek Small Batch"
    )


def test_jsonld_product_offers_and_image():
    page = extract(_fixture("product_retailer.html"))
    products = jsonld_products(page.jsonld)
    names = [p["name"] for p in products]
    assert names == ["Example Creek Small Batch Bourbon", "Other Brand Rye"]
    prices = offer_prices(products[0])
    assert [(p.amount, p.currency, p.kind) for p in prices] == [(Decimal("59.99"), "USD", "retail")]
    assert prices[0].origin == "jsonld:offers.price"
    assert offer_prices(products[1])[0].amount == Decimal("34.5")
    assert product_images(products[0]) == ["https://cdn.bottleshop.test/img/ecsb-large.jpg"]


def test_jsonld_graph_is_walked():
    jsonld = [{"@graph": [{"@type": "Organization"}, {"@type": "Product", "name": "A"}]}]
    assert [p["name"] for p in jsonld_products(jsonld)] == ["A"]


def test_malformed_jsonld_is_ignored():
    page = extract(_fixture("product_retailer.html"))
    assert len(page.jsonld) == 1  # the invalid block is dropped
    bad = extract('<script type="application/ld+json">{"a": </script><p>ok</p>')
    assert bad.jsonld == []
    assert bad.text == "ok"


def test_jsonld_list_at_top_level_is_flattened():
    page = extract(
        '<script type="application/ld+json">[{"@type":"Product","name":"X"}, 3]</script>'
    )
    assert page.jsonld == [{"@type": "Product", "name": "X"}]


def test_offer_prices_rules():
    product = {
        "@type": "Product",
        "offers": [
            {"price": "0", "priceCurrency": "USD"},  # not > 0
            {"price": "49.99"},  # no currency: never assumed
            {"price": "$1,299.00", "priceCurrency": "usd"},
            {"lowPrice": 30, "priceCurrency": "EUR"},
            {"price": True, "priceCurrency": "USD"},
            {"price": "call for price", "priceCurrency": "USD"},
            {"priceSpecification": {"price": "45.00", "priceCurrency": "GBP"}},
        ],
    }
    got = [(p.amount, p.currency) for p in offer_prices(product)]
    assert got == [(Decimal("1299.00"), "USD"), (Decimal("30"), "EUR"), (Decimal("45.00"), "GBP")]


def test_product_images_variants():
    assert product_images({"image": "a.jpg"}) == ["a.jpg"]
    assert product_images({"image": [{"contentUrl": "b.png"}, "a.jpg", "a.jpg"]}) == [
        "b.png",
        "a.jpg",
    ]
    assert product_images({}) == []


def test_text_is_capped():
    html = "<p>" + ("word " * (MAX_TEXT_CHARS // 4)) + "</p>"
    page = extract(html)
    assert MAX_TEXT_CHARS - 10 <= len(page.text) <= MAX_TEXT_CHARS
    assert page.text_truncated is True
    assert page.text.endswith(" word")


def test_text_cap_never_leaves_a_partial_number():
    # The cap falls inside "$1,299.99": the fragment must be dropped, not kept.
    filler = "a" * (MAX_TEXT_CHARS - 4)
    page = extract(f"<p>{filler} $1,299.99 MSRP</p>")
    assert page.text_truncated is True
    assert "$1," not in page.text
    assert page.text == filler


def test_drop_partial_tail():
    assert drop_partial_tail("MSRP is $1,2") == "MSRP is"
    assert drop_partial_tail("line one\nsecond li") == "line one\nsecond"
    assert drop_partial_tail("nospace") == "nospace"


def test_jsonld_script_type_with_parameters_is_parsed():
    page = extract(
        '<script type="application/ld+json; charset=utf-8">'
        '{"@type": "Product", "name": "Example Creek"}</script>'
    )
    assert page.jsonld == [{"@type": "Product", "name": "Example Creek"}]
    assert "Example Creek" not in page.text


def test_garbage_input_does_not_raise():
    for html in ("", "<<<>>>", "<div", "<script>never closed", None):
        assert isinstance(extract(html).text, str)  # type: ignore[arg-type]


def test_round_trip_dict():
    page = extract(_fixture("product_official.html"))
    assert ExtractedHtml.from_dict(page.to_dict()) == page
