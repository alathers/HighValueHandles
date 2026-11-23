"""Prompt, response schema and response parser for bottle-label vision analysis.

Pure and side-effect free: builds the request text sent with the photos of
one bottle and turns the model's reply into ``ExtractedMetadata`` field
values. The request itself is made by ``OllamaClient.chat_with_images``.

The prompt forbids outside knowledge (a vision model asked for an MSRP will
otherwise invent one) and the parser accepts plain JSON, fenced JSON, JSON
embedded in prose and, as a last resort, legacy ``key: value`` lines.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: JSON schema passed as Ollama's ``format``. Every key is required, with null
#: allowed, so the model has to say "not visible" instead of omitting a key.
VISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"]},
        "distillery": {"type": ["string", "null"]},
        "specific_release": {"type": ["string", "null"]},
        "msrp": {"type": ["number", "null"]},
        "proof": {"type": ["number", "null"]},
        "abv": {"type": ["number", "null"]},
        "age": {"type": ["string", "null"]},
        "batch_number": {"type": ["string", "null"]},
        "tasting_notes": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "same_bottle": {"type": "boolean"},
    },
    "required": [
        "name",
        "distillery",
        "specific_release",
        "msrp",
        "proof",
        "abv",
        "age",
        "batch_number",
        "tasting_notes",
        "confidence",
        "same_bottle",
    ],
}



def vision_response_schema(n_images: int) -> dict[str, Any]:
    """The response schema for a request with ``n_images`` photos.

    ``same_bottle`` only means something when several photos are compared,
    so a single-photo request neither asks for nor requires it.
    """
    if n_images > 1:
        return VISION_RESPONSE_SCHEMA
    return {
        "type": "object",
        "properties": {k: v for k, v in VISION_RESPONSE_SCHEMA["properties"].items() if k != "same_bottle"},
        "required": [k for k in VISION_RESPONSE_SCHEMA["required"] if k != "same_bottle"],
    }


VISION_SYSTEM_PROMPT = (
    "You are an expert in American whiskey labels. You extract catalogue data for ONE bottle "
    "from photos of that bottle (front label, back label, neck tag, box). Report only what is "
    "printed or handwritten in the photos. Never guess or use outside knowledge for prices, proof, "
    "age or batch numbers; use null when a field is not visible."
)

#: Generation options, identical on every call (changing them can make the
#: runner reconfigure). 8192 tokens fit four 1280px images plus prompt and answer.
VISION_OPTIONS: dict[str, Any] = {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}

#: Most photos of one bottle sent in a single vision call.
MAX_VISION_IMAGES = 4
#: Long edge (pixels) photos are downscaled to before upload.
VISION_MAX_IMAGE_EDGE = 1280
#: OCR hint budget per photo and in total (characters).
OCR_HINT_MAX_CHARS_PER_IMAGE = 1200
OCR_HINT_MAX_CHARS_TOTAL = 3000
#: Remaining total budget below which no further (fragmentary) hint is added.
_MIN_HINT_CHARS = 20

#: Text fields copied (whitespace-collapsed) from the reply.
TEXT_FIELDS = ("name", "distillery", "specific_release", "age", "batch_number", "tasting_notes")
#: Fields that identify a bottle; a reply with none of them is unusable.
IDENTIFYING_FIELDS = ("name", "distillery", "specific_release")

_KEY_ALIASES = {
    "release": "specific_release",
    "edition": "specific_release",
    "batch": "batch_number",
    "barrel_number": "batch_number",
    "notes": "tasting_notes",
    "price": "msrp",
    "brand": "distillery",
}
_NULL_STRINGS = frozenset({"", "null", "none", "n/a", "na", "unknown", "not visible", "not shown", "-"})
_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_MAX_MSRP = 100000
_MAX_PROOF = 200
_MAX_ABV = 100


@dataclass
class VisionExtraction:
    """Normalised result of parsing one vision reply.

    Attributes:
        fields: ExtractedMetadata field values as strings (absent when not
            visible): name, distillery, specific_release, msrp ('NNN.NN'),
            proof ('<n> Proof'), abv ('<n>%'), age, batch_number, tasting_notes.
        confidence: 'high', 'medium' or 'low' as reported, or None.
        same_bottle: False if the model saw a different product in a photo.
        parse_mode: 'json' (a JSON object was found) or 'lines' (fallback).
    """

    fields: dict[str, str]
    confidence: str | None = None
    same_bottle: bool | None = None
    parse_mode: str = "json"

    @property
    def has_identity(self) -> bool:
        """Whether a name, distillery or release was read."""
        return any(self.fields.get(f) for f in IDENTIFYING_FIELDS)


def _truncate(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars``, at a word boundary when possible."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    cut = text.rfind(" ", 0, max_chars + 1)
    return text[:cut].rstrip() if cut > 0 else text[:max_chars]


def build_vision_prompt(n_images: int, ocr_hints: Sequence[str] = ()) -> str:
    """Build the user message for one bottle's photos.

    Args:
        n_images: Number of photos attached to the request.
        ocr_hints: OCR text per photo, aligned with the photos (``''`` when a
            photo has none). Each hint is truncated to
            ``OCR_HINT_MAX_CHARS_PER_IMAGE`` and all hints together to
            ``OCR_HINT_MAX_CHARS_TOTAL``; with no non-empty hint the OCR section
            is omitted.

    Returns:
        Prompt text.
    """
    if n_images > 1:
        intro = f"The {n_images} attached photos should all show the same bottle. Return a JSON object with:"
    else:
        intro = (
            "The attached photo shows one bottle. Read the labelled bottle in focus and ignore other "
            "bottles or boxes in the background. Return a JSON object with:"
        )
    lines = [
        intro,
        (
            '- name: product name as printed, e.g. "Four Roses Single Barrel". Do not include proof, age, '
            'or generic words like "Kentucky Straight Bourbon Whiskey" unless they are part of the brand name.'
        ),
        (
            "- distillery: distillery or producer as printed (look for 'Distilled by', 'Bottled by'); "
            "if none is printed, the brand name."
        ),
        (
            "- specific_release: edition, series, finish, season/year, store pick or selection details and "
            "designations such as Bottled-in-Bond, Barrel Proof, Unfiltered or Batch Proof, comma separated, "
            "e.g. \"Barrel Strength, OBSK recipe, Sal's Beverage World selection\", "
            '"Bottled-in-Bond, Spring 2021", "Barrel Proof, Unfiltered". Leave out generic class words '
            'such as "Kentucky Straight Bourbon Whiskey"; null if none.'
        ),
        "- msrp: price in US dollars ONLY if a price tag or printed price is visible; otherwise null.",
        "- proof: proof as a number, e.g. 110.4; null if only ABV is shown.",
        "- abv: alcohol by volume percent as a number, e.g. 55.2; null if not shown.",
        '- age: age statement as printed, e.g. "9 years 10 months"; null if none.',
        (
            "- batch_number: batch, barrel, warehouse/barrel code or pact number as printed; null if none. "
            'A number next to PROOF (e.g. "Batch Proof 121.2") is the proof, never a batch number.'
        ),
        "- tasting_notes: tasting notes printed on the label; null if none.",
        "- confidence: high, medium or low for your overall reading.",
    ]
    if n_images > 1:
        lines.append(
            "- same_bottle: false if any photo shows a different product from the others "
            "(then describe the product shown in most photos). Ignore bottles in the background."
        )

    blocks: list[str] = []
    budget = OCR_HINT_MAX_CHARS_TOTAL
    for index, hint in enumerate(ocr_hints, 1):
        if budget < _MIN_HINT_CHARS:
            break
        text = " ".join(str(hint or "").split())
        text = _truncate(text, min(OCR_HINT_MAX_CHARS_PER_IMAGE, budget))
        if not text:
            continue
        budget -= len(text)
        blocks.append(f'<ocr photo="{index}">\n{text}\n</ocr>')

    if blocks:
        lines += [
            "",
            (
                "OCR text from the same photos follows. It is noisy and may be wrong; use it only to confirm "
                "words you can see, and trust the photos when they disagree."
            ),
            *blocks,
        ]
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in ``text``: whole text, inside a ``` fence, or embedded in prose."""
    s = text.strip()
    fenced = _FENCE.match(s)
    if fenced:
        s = fenced.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = s.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = s.find("{", start + 1)
    return None


def _parse_lines(text: str) -> dict[str, str]:
    """Legacy ``key: value`` lines (bullets and ``**bold**`` keys tolerated)."""
    found: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            found[key.strip().strip("*").strip()] = value.strip()
    return found


def _number(value: Any) -> float | None:
    """Numeric value of a JSON number or text like '$1,299.00', '55,2' or '107 proof'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = _THOUSANDS_COMMA.sub("", str(value)).replace(",", ".")
    match = _NUMBER.search(s)
    return float(match.group()) if match else None


def _text(value: Any) -> str | None:
    """Whitespace-collapsed text, or None for null-like values ('n/a', 'unknown', ...)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value if v not in (None, ""))
    s = " ".join(str(value).split())
    return None if s.lower() in _NULL_STRINGS else s


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    return None


def _canonical_keys(raw: Mapping[Any, Any]) -> dict[str, Any]:
    """Lower-case, snake_case and alias the keys; the first non-empty value wins."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        name = _KEY_ALIASES.get(name, name)
        if name not in out or out[name] in (None, ""):
            out[name] = value
    return out


def normalize_fields(raw: Mapping[Any, Any]) -> dict[str, str]:
    """Convert raw reply values to ExtractedMetadata's string fields.

    - Text fields are whitespace-collapsed; null-like strings are dropped.
    - msrp is kept only when 0 < msrp < 100000, as 'NNN.NN'.
    - proof is derived as round(abv * 2, 1) when only ABV is given, kept only
      when 0 < proof <= 200, and formatted '<n> Proof'.
    - abv is kept when 0 < abv <= 100, formatted '<n>%'.
    - Unknown keys are ignored.

    Args:
        raw: Parsed reply (JSON object or ``key: value`` lines).

    Returns:
        Field values; absent fields are omitted.
    """
    data = _canonical_keys(raw)
    out: dict[str, str] = {}
    for name in TEXT_FIELDS:
        text = _text(data.get(name))
        if text:
            out[name] = text

    msrp = _number(data.get("msrp"))
    if msrp is not None and 0 < msrp < _MAX_MSRP:
        with contextlib.suppress(InvalidOperation):
            out["msrp"] = str(Decimal(str(msrp)).quantize(Decimal("0.01")))

    proof = _number(data.get("proof"))
    abv = _number(data.get("abv"))
    if proof is None and abv is not None and 0 < abv <= _MAX_ABV:
        proof = round(abv * 2, 1)
    if proof is not None and 0 < proof <= _MAX_PROOF:
        out["proof"] = f"{proof:g} Proof"
    if abv is not None and 0 < abv <= _MAX_ABV:
        out["abv"] = f"{abv:g}%"
    return out


def parse_vision_response(text: str) -> VisionExtraction:
    """Parse and normalise a vision model reply. Never raises.

    Tries, in order: the whole text as JSON, JSON inside a ``` fence, the
    first JSON object embedded in prose, then legacy ``key: value`` lines.

    Args:
        text: The model's reply (message content).

    Returns:
        VisionExtraction; ``fields`` is empty when nothing usable was found.
    """
    text = text or ""
    obj = _extract_json_object(text)
    if obj is not None:
        raw: Mapping[Any, Any] = obj
        mode = "json"
    else:
        raw = _parse_lines(text)
        mode = "lines"

    data = _canonical_keys(raw)
    confidence = _text(data.get("confidence"))
    confidence = confidence.lower() if confidence and confidence.lower() in _CONFIDENCE_LEVELS else None
    return VisionExtraction(
        fields=normalize_fields(raw),
        confidence=confidence,
        same_bottle=_bool(data.get("same_bottle")),
        parse_mode=mode,
    )
