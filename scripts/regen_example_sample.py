#!/usr/bin/env python3
"""Regenerate ``examples/processed_sample/`` from the real output code.

The sample bottle is fictional ("Example Creek Single Barrel"). It goes
through the same code a ``process`` run uses after the vision step:

1. a bottle dict and ``ExtractedMetadata`` shaped like the vision output
   (label reading, OCR text, a price seen on the label);
2. a hint value (``apply_hint_overrides``) that disagrees with the label;
3. ``Enricher.enrich_batch`` with a scripted search result (two fetched
   pages, two skipped ones) and a scripted text-model reply, behind the real
   ``FactChecker`` (quote verification, reconciliation, publish gate);
4. ``Bottle.from_analysis`` and ``PreviewGenerator.save_bottle_files``.

The resulting fact sheet has a field supported by the label, the OCR text
and a web page (proof), an MSRP that stays Unknown with a reason (only a
retail price was found), and a resolved conflict (the batch number, where
the person's hint wins over the label reading).

No model is called and nothing touches the network. Output is written to a
temporary folder first, absolute temporary paths are replaced by
``/path/to/photos``, and the bottle folder then replaces the old sample.

Usage::

    uv run python scripts/regen_example_sample.py            # rewrite the sample
    uv run python scripts/regen_example_sample.py --out DIR  # write elsewhere (a new or empty folder)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_processor.enrichment import Enricher  # noqa: E402
from image_processor.fact_checker import FactChecker  # noqa: E402
from image_processor.preview_generator import PreviewGenerator  # noqa: E402
from image_processor.web_research import build_queries  # noqa: E402
from lib.hints import HintBook, apply_hint_overrides  # noqa: E402
from lib.ollama_client import ChatResult  # noqa: E402
from models.bottle import Bottle, ExtractedMetadata  # noqa: E402
from models.evidence import PageRecord, WebEvidence, WebResearch  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "examples" / "processed_sample"
#: Replaces the temporary photo folder in every written text file.
PHOTO_DIR_PLACEHOLDER = "/path/to/photos"

BOTTLE_ID = "IMG_0001"
RUN_ID = "20261123_120000"
VISION_MODEL = "qwen3.6:35b-a3b"
TEXT_MODEL = "gpt-oss:120b"
SEARCHED_AT = "2026-11-23T12:00:00+00:00"
EXTRACTED_AT = datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC)

OCR_TEXT = (
    "EXAMPLE CREEK\n"
    "SINGLE BARREL\n"
    "KENTUCKY STRAIGHT BOURBON WHISKEY\n"
    "100 PROOF\n"
    "BATCH C92\n"
    "DISTILLED BY EXAMPLE CREEK DISTILLERY"
)

W1_URL = "https://www.examplecreek.test/single-barrel"
W1_TEXT = (
    "Example Creek Single Barrel\n"
    "Each barrel is chosen by hand. Example Creek Single Barrel is distilled by "
    "Example Creek Distillery and bottled at 100 proof."
)
W2_URL = "https://shop.test/example-creek-single-barrel"
W2_TEXT = "Example Creek Single Barrel 750ml\nOur price $69.99\nAdd to cart"

HINTS_TOML = f"""\
[bottle.{BOTTLE_ID}]
note = "The batch number is on the neck label."
source = "Owner read the neck label on the bottle, 2026-11-20"

[bottle.{BOTTLE_ID}.fields]
batch_number = "C923"
"""

EXTRACT_REPLY = json.dumps({
    "page_match": [
        {"source_id": "W1", "verdict": "exact", "quote": "Example Creek Single Barrel is distilled by"},
        {"source_id": "W2", "verdict": "exact", "quote": "Example Creek Single Barrel 750ml"},
    ],
    "claims": [
        {"field": "proof", "value": "100", "source_id": "W1", "quote": "bottled at 100 proof"},
        {"field": "distillery", "value": "Example Creek Distillery", "source_id": "W1",
         "quote": "distilled by Example Creek Distillery"},
    ],
    "prices": [
        {"kind": "retail", "amount": 69.99, "currency": "USD", "source_id": "W2", "quote": "Our price $69.99"},
    ],
})
COMPOSE_REPLY = json.dumps({"sentences": [
    {"text": "Example Creek Single Barrel is distilled by Example Creek Distillery.",
     "citations": [1, 2], "fields": ["name", "distillery"]},
]})


class ScriptedTextLLM:
    """Stands in for ``OllamaClient.chat_text``: returns the scripted replies in order."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def chat_text(self, prompt, model=None, **_kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        if not self.replies:
            raise RuntimeError("unexpected extra text-model call")
        return ChatResult(content=self.replies.pop(0), thinking="", model=model or "", done_reason="stop")


class ScriptedProvider:
    name = "duckduckgo"


class ScriptedResearcher:
    """Stands in for ``WebResearcher``: one research record per bottle, no network.

    The queries are built from the search identity the real ``Enricher``
    passes in, with the real ``build_queries``, so they match a live run.
    """

    provider = ScriptedProvider()

    def research_many(self, items, *, workers=1):  # noqa: ANN001, ANN201, ARG002
        return [web_research(build_queries(identity)) for identity, _photo in items]


def web_research(queries: list[str]) -> WebResearch:
    """Two fetched pages (official, retailer) and two pages skipped before fetching."""
    if not queries:
        raise RuntimeError("the sample identity produced no search query")
    w1 = WebEvidence.create(
        text=W1_TEXT, source_id="W1", url=W1_URL, domain="examplecreek.test", tier="official",
        title="Single Barrel | Example Creek", fetched_at="2026-11-23T12:00:04+00:00", http_status=200,
        query=queries[0], from_cache=False, truncated=False,
    )
    w2 = WebEvidence.create(
        text=W2_TEXT, source_id="W2", url=W2_URL, domain="shop.test", tier="retailer",
        title="Example Creek Single Barrel 750ml", fetched_at="2026-11-23T12:00:07+00:00", http_status=200,
        query=queries[0], from_cache=False, truncated=False,
    )
    return WebResearch(
        status="ok",
        provider="duckduckgo",
        queries=list(queries),
        searched_at=SEARCHED_AT,
        pages=[
            PageRecord(url=W1_URL, title=w1.title, domain=w1.domain, source_class="official",
                       retrieved_at=w1.fetched_at, outcome="fetched_not_cited",
                       reason="sent to fact check as W1"),
            PageRecord(url=W2_URL, title=w2.title, domain=w2.domain, source_class="retailer",
                       retrieved_at=w2.fetched_at, outcome="fetched_not_cited",
                       reason="sent to fact check as W2"),
            PageRecord(url="https://forum.test/t/example-creek", title="Example Creek thread",
                       domain="forum.test", source_class=None, retrieved_at=None, outcome="skipped",
                       reason="deny-listed"),
            PageRecord(url="https://blog.test/bourbon-list", title="Ten bourbons to try",
                       domain="blog.test", source_class="reference", retrieved_at=None, outcome="skipped",
                       reason="robots.txt disallows fetching"),
        ],
        evidence=[w1, w2],
    )


def _write_photo(path: Path) -> None:
    """A small synthetic placeholder "photo" (not a real bottle)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (240, 360), (196, 140, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 90, 210, 300), fill=(245, 236, 214), outline=(60, 40, 20), width=3)
    for i, line in enumerate(("EXAMPLE CREEK", "SINGLE BARREL", "100 PROOF", "SAMPLE IMAGE")):
        draw.text((48, 120 + 40 * i), line, fill=(40, 25, 10))
    image.save(path, "JPEG", quality=70)


def build_sample(output_dir: Path, work_dir: Path) -> Path:
    """Run the sample bottle through the post-vision pipeline; return its saved folder."""
    photos = work_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    photo = photos / f"{BOTTLE_ID}.jpg"
    _write_photo(photo)

    metadata = ExtractedMetadata(
        name="Example Creek Single Barrel",
        distillery="Example Creek Distillery",
        specific_release="Single Barrel",
        proof="100 Proof",
        batch_number="C92",
        label_price="64.99",
        full_text=OCR_TEXT,
        selected_text=OCR_TEXT,
        source_images=[photo],
        vision_model=VISION_MODEL,
        vision_confidence="high",
        ocr_by_image={photo.name: OCR_TEXT},
    )
    bottle_dict = {
        "bottle_id": BOTTLE_ID,
        "name": metadata.name,
        "distillery": metadata.distillery,
        "specific_release": metadata.specific_release,
        "proof": metadata.proof,
        "abv": None,
        "age": None,
        "batch_number": metadata.batch_number,
        "tasting_notes": None,
        "label_price": metadata.label_price,
        "msrp": None,
        "images": [str(photo)],
        "model_used": VISION_MODEL,
        "confidence": "HIGH",
        "quantity": 1,
        "run_id": RUN_ID,
    }

    hints_path = work_dir / "hints.toml"
    hints_path.write_text(HINTS_TOML, encoding="utf-8")
    _, unapplied = apply_hint_overrides([bottle_dict], [metadata], HintBook.load(hints_path), hints_path)
    if unapplied:
        raise RuntimeError(f"hint not applied: {unapplied}")

    enricher = Enricher(
        researcher=ScriptedResearcher(),
        fact_checker=FactChecker(ScriptedTextLLM(EXTRACT_REPLY, COMPOSE_REPLY), model=TEXT_MODEL, max_attempts=1),
        vision_model=VISION_MODEL,
        text_model=TEXT_MODEL,
        keep_models_loaded=True,
        workers=1,
        unload=lambda _name: None,
    )
    enricher.enrich_batch([bottle_dict], [metadata], run_id=RUN_ID)
    facts = bottle_dict.get("facts") or {}
    verification = facts.get("verification") or {}
    if verification.get("status") != "verified":
        raise RuntimeError(f"sample fact check did not verify: {verification}")
    verification["duration_s"] = 74.2  # a typical gpt-oss:120b time; the scripted reply takes ~0 s

    bottle = Bottle.from_analysis(
        bottle_dict, metadata, quantity=bottle_dict.get("quantity"),
        corrections=bottle_dict.get("corrections") or [],
    )
    bottle.extracted_at = EXTRACTED_AT
    for correction in bottle.corrections:  # wall-clock time; pinned so the sample is reproducible
        correction.corrected_at = EXTRACTED_AT
    return PreviewGenerator(output_dir=output_dir).save_bottle_files(
        bottle, metadata, model_used=bottle_dict.get("model_used"), confidence=bottle_dict.get("confidence"),
    )


def _replace_paths(folder: Path, old: str, new: str) -> None:
    """Replace ``old`` with ``new`` in every text file of ``folder`` (JSON strings stay valid)."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".txt"} and "evidence" not in path.parts:
            text = path.read_text(encoding="utf-8")
            if old in text:
                path.write_text(text.replace(old, new), encoding="utf-8")


def _check_target(target: Path) -> None:
    """Refuse to replace a folder that is not a sample folder.

    ``regenerate`` deletes ``target`` first, so ``--out`` must name a new or
    empty folder, or one that holds only an earlier sample (bottle folders
    with the ``.bourbon_output`` marker). This keeps ``--out .`` or a typo
    from deleting unrelated files.
    """
    if not target.exists():
        return
    if not target.is_dir():
        raise SystemExit(f"refusing to replace {target}: not a folder")
    for child in target.iterdir():
        if not (child.is_dir() and (child / ".bourbon_output").is_file()):
            raise SystemExit(
                f"refusing to replace {target}: {child.name} is not an earlier sample bottle folder; "
                "pass --out with a new or empty folder"
            )


def regenerate(target: Path) -> Path:
    """Write the sample bottle folder into ``target`` (replacing an earlier sample); return the folder."""
    _check_target(target)
    with tempfile.TemporaryDirectory(prefix="bourbon-sample-") as tmp, _utc_local_time():
        work = Path(tmp).resolve()
        folder = build_sample(work / "out", work)
        _replace_paths(folder, str(work / "photos"), PHOTO_DIR_PLACEHOLDER)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        dest = target / folder.name
        shutil.copytree(folder, dest)
    return dest


@contextlib.contextmanager
def _utc_local_time() -> Iterator[None]:
    """Render local times in UTC (the sample must not depend on the machine's time zone)."""
    saved = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        if hasattr(time, "tzset"):
            time.tzset()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=SAMPLE_DIR, help=f"Target folder (default: {SAMPLE_DIR})")
    args = parser.parse_args(argv)
    dest = regenerate(args.out)
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
