"""Tests for image_processor.vendor_image (fake client; no model calls)."""

import json
from pathlib import Path
from typing import Any

import pytest

from image_processor.vendor_image import (
    OPTIONS,
    PROMPT,
    SCHEMA,
    make_vision_verifier,
    parse_verdict,
)
from lib.ollama_client import ChatResult


class FakeClient:
    def __init__(self, content: Any) -> None:  # noqa: ANN401
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat_with_images(self, prompt, image_paths, model=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "image_paths": list(image_paths), "model": model, **kwargs}
        )
        if isinstance(self.content, Exception):
            raise self.content
        return ChatResult(content=self.content, model=model or "")


GOOD = {
    "same_product": True,
    "same_release": "yes",
    "label_text_seen": "Example Creek",
    "confidence": "high",
}


def test_payload_two_images_think_false_schema(tmp_path: Path):
    client = FakeClient(json.dumps(GOOD))
    verify = make_vision_verifier(client, "qwen3.6:35b-a3b", read_timeout=42)
    primary, candidate = tmp_path / "mine.jpg", tmp_path / "vendor.jpg"
    verdict = verify(primary, candidate)
    call = client.calls[0]
    assert call["image_paths"] == [primary, candidate]
    assert call["model"] == "qwen3.6:35b-a3b"
    assert call["think"] is False
    assert call["format"] == SCHEMA
    assert call["options"] == OPTIONS == {"temperature": 0, "num_ctx": 8192, "num_predict": 512}
    assert call["read_timeout"] == 42
    assert "Photo 1 is the user's own bottle" in call["prompt"]
    assert call["prompt"] == PROMPT
    assert verdict == {**GOOD, "model": "qwen3.6:35b-a3b"}


def test_schema_requires_every_key():
    assert set(SCHEMA["required"]) == {
        "same_product",
        "same_release",
        "label_text_seen",
        "confidence",
    }
    assert set(SCHEMA["properties"]) == set(SCHEMA["required"])


@pytest.mark.parametrize(
    "content",
    ["not json at all", "", "[1, 2]", json.dumps({"same_product": True}), "{broken"],
)
def test_malformed_reply_is_not_same_product(content: str, tmp_path: Path):
    verdict = make_vision_verifier(FakeClient(content), "m")(tmp_path / "a", tmp_path / "b")
    assert verdict["same_product"] is False
    assert verdict["confidence"] == "low"
    assert "error" in verdict


def test_fenced_reply_is_parsed():
    v = parse_verdict("```json\n" + json.dumps(GOOD) + "\n```")
    assert v["same_product"] is True
    assert v["confidence"] == "high"


def test_out_of_enum_values_made_conservative():
    v = parse_verdict(
        json.dumps({**GOOD, "same_product": "yes", "confidence": "very", "same_release": "maybe"})
    )
    assert v["same_product"] is False
    assert v["confidence"] == "low"
    assert v["same_release"] == "unclear"


def test_client_exception_propagates(tmp_path: Path):
    verify = make_vision_verifier(FakeClient(RuntimeError("model gone")), "m")
    with pytest.raises(RuntimeError, match="model gone"):
        verify(tmp_path / "a", tmp_path / "b")
