"""Vision check that a vendor's product photo shows the user's bottle.

:func:`make_vision_verifier` returns ``verify(primary_photo, candidate)``,
which asks the vision model to compare photo 1 (the user's bottle) with
photo 2 (a reference image from a vendor page) and returns a verdict dict:

``{"same_product": bool, "same_release": "yes"|"no"|"unclear",
"label_text_seen": str, "confidence": "high"|"medium"|"low", "model": str}``

The researcher accepts an image only when ``same_product is True``,
``confidence == "high"`` and ``same_release != "no"``. A malformed reply is
returned as ``{"same_product": False, "confidence": "low", "error": ...}``.
Client exceptions propagate; the researcher treats them as "vendor image
skipped", never as a research failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

#: JSON schema the model must answer with (every key required).
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_product": {"type": "boolean"},
        "same_release": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "label_text_seen": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["same_product", "same_release", "label_text_seen", "confidence"],
    "additionalProperties": False,
}

#: Generation options: deterministic and short.
OPTIONS: dict[str, Any] = {"temperature": 0, "num_ctx": 8192, "num_predict": 512}

PROMPT = """You compare two photos of whiskey bottles.
Photo 1 is the user's own bottle. Photo 2 is a reference image from a web page.

Answer same_product=true ONLY if the brand and the product name are readable on BOTH
photos and they match. If either label is unreadable, blurry, hidden or different,
answer same_product=false. Do not guess from bottle shape or color alone.

same_release: "yes" if a readable release detail (year, batch, edition, proof) matches
on both photos, "no" if a readable release detail differs, "unclear" otherwise.
label_text_seen: the brand and product words you could read on photo 2 (copied exactly).
confidence: "high" only when the label text on both photos is clearly legible.

Reply with JSON only:
{"same_product": true|false, "same_release": "yes"|"no"|"unclear",
 "label_text_seen": "...", "confidence": "high"|"medium"|"low"}"""

try:  # reuse the tolerant parser of the vision stage when available
    from image_processor.vision_prompt import _extract_json_object as _extract
except Exception:  # noqa: BLE001 - fall back to plain json

    def _extract(text: str) -> Optional[dict[str, Any]]:
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None


def parse_verdict(text: Any) -> dict[str, Any]:  # noqa: ANN401
    """Verdict dict from the model reply; never raises.

    Unknown enum values are made conservative (``same_release='unclear'``,
    ``confidence='low'``); ``same_product`` is true only for a JSON ``true``.
    """
    try:
        obj = _extract(str(text or ""))
    except Exception as e:  # noqa: BLE001
        return {"same_product": False, "confidence": "low", "error": f"unparseable reply ({e})"}
    if not isinstance(obj, dict):
        return {"same_product": False, "confidence": "low", "error": "reply is not a JSON object"}
    missing = [k for k in SCHEMA["required"] if k not in obj]
    if missing:
        return {
            "same_product": False,
            "confidence": "low",
            "error": f"reply misses {', '.join(missing)}",
        }
    release = obj.get("same_release")
    confidence = obj.get("confidence")
    return {
        "same_product": obj.get("same_product") is True,
        "same_release": release if release in ("yes", "no", "unclear") else "unclear",
        "label_text_seen": str(obj.get("label_text_seen") or "")[:300],
        "confidence": confidence if confidence in ("high", "medium", "low") else "low",
    }


def make_vision_verifier(
    client: Any,  # noqa: ANN401
    model: str,
    *,
    read_timeout: Optional[float] = None,
) -> Callable[[Path, Path], dict]:
    """``verify(primary_photo, candidate) -> verdict`` using ``client.chat_with_images``.

    Args:
        client: An :class:`lib.ollama_client.OllamaClient` (or fake).
        model: Vision model name.
        read_timeout: Read timeout for the call, in seconds.
    """

    def verify(primary_photo: Path, candidate: Path) -> dict:
        result = client.chat_with_images(
            PROMPT,
            [Path(primary_photo), Path(candidate)],
            model,
            format=SCHEMA,
            think=False,
            options=dict(OPTIONS),
            read_timeout=read_timeout,
        )
        verdict = parse_verdict(getattr(result, "content", result))
        verdict["model"] = model
        return verdict

    return verify
