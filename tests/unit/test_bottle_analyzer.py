"""Unit tests for BottleAnalyzer class.

Tests metadata extraction, image grouping, and bottle analysis functionality.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import threading

from image_processor.bottle_analyzer import LATE_ANSWER_WARNING, MIXED_GROUP_WARNING, BottleAnalyzer
from image_processor.text_extractor import TextExtractor
from image_processor.vision_prompt import (
    VISION_MAX_IMAGE_EDGE,
    VISION_OPTIONS,
    VISION_RESPONSE_SCHEMA,
    VISION_SYSTEM_PROMPT,
)
from lib.constants import DEFAULT_VISION_MODEL
from lib.duplicates import DuplicateDetector
from lib.exceptions import (
    AIModelError,
    ModelServerError,
    OllamaConnectionError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionResponseError,
    VisionTimeoutError,
)
from lib.model_selector import ModelStage
from lib.ollama_client import OllamaClient, ChatResult
from lib.retry import RetryPolicy
from models.bottle import ExtractedMetadata
from models.image import Image, ImageCategory


def _reply(content=None, done_reason="stop", **fields) -> ChatResult:
    """A vision reply: raw ``content``, or a schema-shaped JSON object from ``fields``."""
    if content is None:
        body = {
            k: None
            for k in ("name", "distillery", "specific_release", "label_price", "proof", "abv", "age",
                      "batch_number", "tasting_notes")
        }
        body.update({"confidence": "high", "same_bottle": True})
        body.update(fields)
        content = json.dumps(body)
    return ChatResult(content=content, done_reason=done_reason, prompt_eval_count=10, eval_count=5)


def _vision_analyzer(**kwargs) -> tuple[BottleAnalyzer, Mock]:
    """Analyzer with a spec'd OllamaClient mock and a no-op OCR mock."""
    client = Mock(spec=OllamaClient)
    extractor = Mock(spec=TextExtractor)
    extractor.extract_with_retry.return_value = ""
    kwargs.setdefault("sleep", lambda _seconds: None)
    analyzer = BottleAnalyzer(
        text_extractor=kwargs.pop("text_extractor", extractor),
        ollama_client=client,
        **kwargs,
    )
    return analyzer, client


class TestBottleAnalyzerInitialization:
    """Test BottleAnalyzer initialization."""

    @patch("image_processor.bottle_analyzer.TextExtractor")
    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_analyzer_default_initialization(self, mock_ollama, mock_extractor):
        """Defaults: own OCR extractor and client, a 3-attempt retry policy."""
        mock_extractor.return_value = Mock(spec=TextExtractor)
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer()

        assert analyzer.text_extractor is not None
        assert analyzer.ollama_client is not None
        assert analyzer.retry_policy == RetryPolicy()
        assert not hasattr(analyzer, "retry_manager")
        assert not hasattr(analyzer, "image_organizer")
        mock_extractor.assert_called_once_with(analyzer.retry_policy)
        mock_ollama.assert_called_once()

    def test_analyzer_custom_dependencies(self):
        """Test analyzer uses provided custom dependencies."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_ollama_client = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama_client,
            retry_policy=RetryPolicy.DISABLED,
        )

        assert analyzer.text_extractor is mock_text_extractor
        assert analyzer.ollama_client is mock_ollama_client
        assert analyzer.retry_policy is RetryPolicy.DISABLED


class TestBottleAnalyzerProcessSingleBottle:
    """Test process_single_bottle method."""

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_success(self, mock_ollama):
        """Test successful processing of single bottle images."""
        # Setup mocks
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "Test OCR text from bottle"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        # Mock analyze_with_vision_model
        mock_metadata = ExtractedMetadata(
            name="Test Bottle",
            distillery="Test Distillery",
            label_price="49.99",
            specific_release="Single Barrel",
            full_text="Test OCR text",
            selected_text="Test OCR text",
            source_images=["/path/to/image.jpg"],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            # Create test images
            images = [
                Image(path="/path/to/img1.jpg", category="BOTTLE", is_primary=True),
                Image(path="/path/to/img2.jpg", category="BOTTLE", is_primary=False),
            ]

            # Process bottle
            bottle_dict, metadata = analyzer.process_single_bottle(images, vision_model="test-model")

            # Verify results
            assert bottle_dict["name"] == "Test Bottle"
            assert bottle_dict["distillery"] == "Test Distillery"
            assert bottle_dict["label_price"] == "49.99"
            assert bottle_dict["msrp"] is None  # a visible price is never an MSRP
            assert bottle_dict["specific_release"] == "Single Barrel"
            assert len(bottle_dict["images"]) == 2
            assert metadata.name == "Test Bottle"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_with_bill_images(self, mock_ollama):
        """Test processing preferentially skips BILL category images."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "OCR text"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        mock_metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            label_price="29.99",
            specific_release="Test",
            full_text="OCR",
            selected_text="OCR",
            source_images=["/path.jpg"],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            # Create images with BILL category first
            images = [
                Image(path="/bill.jpg", category="BILL", is_primary=False),
                Image(path="/bottle1.jpg", category="BOTTLE", is_primary=True),
            ]

            bottle_dict, metadata = analyzer.process_single_bottle(images)

            # Should use non-BILL image for primary
            assert bottle_dict["name"] == "Test"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_ocr_extraction_error_continues(self, mock_ollama):
        """Test OCR extraction errors are caught and logged, processing continues."""
        mock_text_extractor = Mock(spec=TextExtractor)
        # First image raises error, second succeeds
        mock_text_extractor.extract_with_retry.side_effect = [
            Exception("OCR failed"),
            "Success text",
        ]
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        mock_metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            label_price="29.99",
            full_text="Success text",
            selected_text="Success text",
            source_images=["/path.jpg"],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            images = [
                Image(path="/fail.jpg", category="BOTTLE"),
                Image(path="/success.jpg", category="BOTTLE"),
            ]

            # Should continue despite OCR error
            bottle_dict, metadata = analyzer.process_single_bottle(images)
            assert metadata.full_text == "Success text"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_all_ocr_fails_uses_vision_only(self, mock_ollama):
        """Test when all OCR fails, uses vision model response only."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.side_effect = [Exception("Failed"), Exception("Failed")]
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        mock_metadata = ExtractedMetadata(
            name="Vision Only",
            distillery="Distillery",
            label_price="39.99",
            full_text="",  # No OCR text
            selected_text="",
            source_images=["/path.jpg"],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            images = [Image(path="/img.jpg", category="BOTTLE")]

            bottle_dict, metadata = analyzer.process_single_bottle(images)
            assert bottle_dict["name"] == "Vision Only"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_no_primary_image(self, mock_ollama):
        """Test behavior when no primary image is available."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "OCR text"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        # No primary image set
        images = [
            Image(path="/img1.jpg", category="BOTTLE", is_primary=False),
            Image(path="/img2.jpg", category="BOTTLE", is_primary=False),
        ]

        # Should fall back to OCR-only extraction
        with patch.object(analyzer, "analyze_with_vision_model") as mock_analyze:
            # Should still call analyze but use first image
            mock_analyze.return_value = ExtractedMetadata(
                name="Fallback",
                distillery="Test",
                label_price="29.99",
                full_text="OCR text",
                selected_text="OCR text",
                source_images=["/img1.jpg"],
            )

            bottle_dict, metadata = analyzer.process_single_bottle(images)
            assert bottle_dict["name"] == "Fallback"
            # Should have called analyze with first image
            mock_analyze.assert_called_once()

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_process_single_bottle_empty_images(self, mock_ollama):
        """Test processing with empty image list."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
        )

        images = []

        mock_metadata = ExtractedMetadata(
            name="Empty",
            distillery="Test",
            label_price="0.00",
            full_text="",
            selected_text="",
            source_images=[],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            bottle_dict, metadata = analyzer.process_single_bottle(images)
            # Should still produce a bottle dict with unknown name
            assert bottle_dict["name"] == "Empty"
            assert len(bottle_dict["images"]) == 0


class TestBottleAnalyzerDetectQuantity:
    """Test detect_quantity method."""

    def test_detect_quantity_no_images_returns_zero(self):
        """Test detect_quantity returns 0 for empty image list."""
        analyzer = BottleAnalyzer()
        quantity = analyzer.detect_quantity([])
        assert quantity == 0

    def test_detect_quantity_single_image_returns_one(self):
        """Test single image returns quantity of 1."""
        analyzer = BottleAnalyzer()
        images = [Image(path="/img.jpg", category="BOTTLE")]

        mock_detector = Mock(spec=DuplicateDetector)
        mock_detector.generate_hash.return_value = "hash1"
        assert analyzer.detect_quantity(images, duplicate_detector=mock_detector) == 1

    def test_detect_quantity_duplicate_images_returns_count(self):
        """Test duplicate hashes detected correctly."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)

        # Simulate 3 identical images
        mock_detector.generate_hash.return_value = "hash123"

        images = [
            Image(path="/img1.jpg", category="BOTTLE"),
            Image(path="/img2.jpg", category="BOTTLE"),
            Image(path="/img3.jpg", category="BOTTLE"),
        ]

        quantity = analyzer.detect_quantity(images, duplicate_detector=mock_detector)

        # Should detect 3 duplicates
        assert quantity == 3

    def test_detect_quantity_different_hashes_returns_one(self):
        """Test different hashes return quantity of 1."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)

        # Simulate different hashes (multi-angle shots)
        mock_detector.generate_hash.side_effect = ["hash1", "hash2", "hash3"]

        images = [
            Image(path="/img1.jpg", category="BOTTLE"),
            Image(path="/img2.jpg", category="BOTTLE"),
            Image(path="/img3.jpg", category="BOTTLE"),
        ]

        quantity = analyzer.detect_quantity(images, duplicate_detector=mock_detector)
        assert quantity == 1

    def test_detect_quantity_hash_generation_error_logs_warning(self):
        """Test hash generation errors are logged and don't crash."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)
        mock_detector.generate_hash.side_effect = [Exception("Hash error"), "hash2"]

        images = [
            Image(path="/img1.jpg", category="BOTTLE"),
            Image(path="/img2.jpg", category="BOTTLE"),
        ]

        # Should handle errors gracefully
        quantity = analyzer.detect_quantity(images, duplicate_detector=mock_detector)
        assert quantity == 1  # Only one successful hash

    def test_detect_quantity_is_largest_identical_set(self):
        """Quantity is the largest set of byte-identical photos (no visual heuristic)."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)
        mock_detector.generate_hash.side_effect = ["hashA", "hashA", "hashA", "hashB", "hashB", "hashB"]
        images = [Image(path=f"/img{i}.jpg", category="BOTTLE") for i in range(6)]

        assert analyzer.detect_quantity(images, duplicate_detector=mock_detector) == 3
        assert not hasattr(BottleAnalyzer, "_estimate_quantity_from_images")


class TestBottleAnalyzerCalculateConfidence:
    """Test calculate_confidence method."""

    def test_calculate_confidence_high_complete_metadata(self):
        """Test HIGH confidence when most fields are filled."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Test",
            distillery="Distillery",
            label_price="49.99",
            specific_release="Release",
            proof="90 Proof",
            age="10 Years",
            full_text="",
            selected_text="",
            source_images=[],
        )

        confidence = analyzer.calculate_confidence(metadata)
        assert confidence == "HIGH"

    def test_calculate_confidence_medium_partial_metadata(self):
        """Test MEDIUM confidence when some fields filled."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Test",
            distillery="Distillery",
            label_price="49.99",
            specific_release="",  # Empty
            proof="90 Proof",
            age="",  # Empty
            full_text="",
            selected_text="",
            source_images=[],
        )

        confidence = analyzer.calculate_confidence(metadata)
        assert confidence == "MEDIUM"

    def test_calculate_confidence_low_minimal_metadata(self):
        """Test LOW confidence when few fields filled."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="",  # Empty
            distillery="",  # Empty
            label_price="",
            specific_release="",  # Empty
            proof="",  # Empty
            age="5 Years",  # Only one field
            full_text="",
            selected_text="",
            source_images=[],
        )

        confidence = analyzer.calculate_confidence(metadata)
        assert confidence == "LOW"

    def test_calculate_confidence_missing_attributes(self):
        """Test handling of missing attributes gracefully."""
        analyzer = BottleAnalyzer()

        # Create minimal metadata
        metadata = ExtractedMetadata(
            full_text="test",
            selected_text="test",
            source_images=[],
        )

        confidence = analyzer.calculate_confidence(metadata)
        # With no optional fields filled
        assert confidence == "LOW"


class TestBottleAnalyzerAnalyzeWithVisionModel:
    """Test analyze_with_vision_model method."""

    def test_analyze_with_vision_model_success(self):
        """A schema JSON reply is normalised to ExtractedMetadata's string fields."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(
            name="Blanton's",
            distillery="Buffalo Trace",
            label_price=59.99,
            proof=93,
            age="No Age Statement",
        )

        metadata = analyzer.analyze_with_vision_model(
            [Path("/test.jpg")], ["OCR text"], model="test-model"
        )

        assert metadata.name == "Blanton's"
        assert metadata.distillery == "Buffalo Trace"
        assert metadata.label_price == "59.99"
        assert metadata.proof == "93 Proof"
        assert metadata.age == "No Age Statement"
        assert metadata.full_text == "OCR text"
        assert metadata.source_images == ["/test.jpg"]
        assert metadata.vision_model == "test-model"
        assert metadata.vision_confidence == "high"
        assert metadata.warnings == []

    def test_analyze_with_vision_model_sends_standard_request(self):
        """Schema, system prompt, think=False, fixed options and downscaling are always sent."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="X")

        analyzer.analyze_with_vision_model(
            [Path("/a.jpg"), Path("/b.jpg")], ["front text", ""], model="m"
        )

        kwargs = client.chat_with_images.call_args.kwargs
        assert kwargs["image_paths"] == [Path("/a.jpg"), Path("/b.jpg")]
        assert kwargs["model"] == "m"
        assert kwargs["system"] == VISION_SYSTEM_PROMPT
        assert kwargs["format"] == VISION_RESPONSE_SCHEMA
        assert kwargs["think"] is False
        assert kwargs["options"] == VISION_OPTIONS
        assert kwargs["max_image_edge"] == VISION_MAX_IMAGE_EDGE
        assert '<ocr photo="1">\nfront text\n</ocr>' in kwargs["prompt"]
        assert '<ocr photo="2">' not in kwargs["prompt"]
        assert "2 attached photos" in kwargs["prompt"]

    def test_analyze_with_vision_model_legacy_lines(self):
        """Old 'key: value' replies still parse (backends without schema support)."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(
            content=(
                "name: Blanton's\n"
                "distillery: Buffalo Trace\n"
                "price: $59.99\n"
                "proof: 93 Proof\n"
                "age: No Age Statement"
            )
        )

        metadata = analyzer.analyze_with_vision_model(Path("/test.jpg"), "OCR text", model="m")

        assert metadata.name == "Blanton's"
        assert metadata.label_price == "59.99"
        assert metadata.proof == "93 Proof"
        assert metadata.full_text == "OCR text"

    def test_analyze_with_vision_model_raises_on_client_error(self):
        """Non-fatal client errors become VisionAnalysisError (never an OCR-only result)."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.side_effect = AIModelError("boom")

        with pytest.raises(VisionAnalysisError, match="boom") as exc_info:
            analyzer.analyze_with_vision_model([Path("/test.jpg")], ["Fallback OCR text"], model="m")

        assert "test.jpg" in str(exc_info.value)
        assert "'m'" in str(exc_info.value)

    def test_analyze_with_vision_model_wraps_invalid_image(self):
        """An unreadable photo fails that bottle with a VisionAnalysisError."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.side_effect = ValueError("Invalid image file /x.jpg: truncated")

        with pytest.raises(VisionAnalysisError, match="Invalid image file"):
            analyzer.analyze_with_vision_model([Path("/x.jpg")], "", model="m")

    @pytest.mark.parametrize(
        "error",
        [
            VisionModelUnavailableError("model 'm' not found"),
            ModelServerError("not running"),
        ],
    )
    def test_unusable_model_errors_propagate_unchanged(self, error):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.side_effect = error

        with pytest.raises(type(error)) as exc_info:
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert exc_info.value is error
        client.chat_with_images.assert_called_once()  # never retried

    def test_connection_error_retried_then_propagates_unchanged(self):
        """A refused connection may pass; after the policy's attempts it is still fatal."""
        error = OllamaConnectionError("Cannot reach Ollama")
        analyzer, client = _vision_analyzer()
        client.chat_with_images.side_effect = error

        with pytest.raises(OllamaConnectionError) as exc_info:
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert exc_info.value is error
        assert exc_info.value.attempts == 3
        assert client.chat_with_images.call_count == 3

    def test_timeout_retried_once_only(self):
        analyzer, client = _vision_analyzer(retry_policy=RetryPolicy(max_attempts=5))
        client.chat_with_images.side_effect = VisionTimeoutError("too slow")

        with pytest.raises(VisionTimeoutError) as excinfo:
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert client.chat_with_images.call_count == 2
        assert excinfo.value.attempts == 2

    def test_empty_content_retried_then_raises(self):
        """Empty content (e.g. thinking used the whole budget) is retried up to the policy."""
        analyzer, client = _vision_analyzer(retry_policy=RetryPolicy(max_attempts=2))
        client.chat_with_images.return_value = ChatResult(
            content="", thinking="hmm" * 100, done_reason="length"
        )

        with pytest.raises(VisionResponseError, match="no content.*done_reason=length.*thinking_chars=300"):
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert client.chat_with_images.call_count == 2

    def test_retry_succeeds_on_second_attempt(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.side_effect = [_reply(content=""), _reply(name="Second Try")]

        metadata = analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert metadata.name == "Second Try"
        assert client.chat_with_images.call_count == 2
        assert metadata.vision_attempts == 2
        assert LATE_ANSWER_WARNING.format(attempt=2) in metadata.warnings

    def test_all_null_reply_raises(self):
        """A reply without name, distillery or release is unusable."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(proof=90, confidence="low")

        with pytest.raises(VisionResponseError, match="no identifying fields"):
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert client.chat_with_images.call_count == 3  # the default policy

    def test_truncated_non_json_reply_raises(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(
            content='name: Partial\n{"distillery": "Cut', done_reason="length"
        )

        with pytest.raises(VisionResponseError, match="cut off"):
            analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

    def test_same_bottle_false_adds_warning(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Four Roses", same_bottle=False)

        metadata = analyzer.analyze_with_vision_model([Path("/a.jpg"), Path("/b.jpg")], "", model="m")

        assert metadata.name == "Four Roses"
        assert metadata.warnings == [MIXED_GROUP_WARNING]
        assert metadata.same_bottle is False  # read by the batch split check

    def test_same_bottle_true_exposed(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Four Roses", same_bottle=True)

        metadata = analyzer.analyze_with_vision_model([Path("/a.jpg"), Path("/b.jpg")], "", model="m")

        assert metadata.same_bottle is True

    def test_same_bottle_false_on_single_image_adds_no_warning(self):
        """One photo cannot show different products: background bottles are not a mixed group."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Stagg", same_bottle=False)

        metadata = analyzer.analyze_with_vision_model([Path("/a.jpg")], "", model="m")

        assert metadata.warnings == []
        assert metadata.same_bottle is None
        kwargs = client.chat_with_images.call_args.kwargs
        assert "same_bottle" not in kwargs["format"]["required"]
        assert "same_bottle" not in kwargs["prompt"]

    def test_analyze_with_vision_model_custom_model(self):
        """Custom vision model is used."""
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Test Model")

        analyzer.analyze_with_vision_model([Path("/test.jpg")], "OCR", model="qwen3-vl:32b")

        client.chat_with_images.assert_called_once()
        assert client.chat_with_images.call_args.kwargs["model"] == "qwen3-vl:32b"

    def test_default_model_when_none_given(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="X")

        metadata = analyzer.analyze_with_vision_model([Path("/test.jpg")], "")

        assert client.chat_with_images.call_args.kwargs["model"] == DEFAULT_VISION_MODEL
        assert metadata.vision_model == DEFAULT_VISION_MODEL

    def test_no_images_returns_ocr_only_without_calling_model(self):
        analyzer, client = _vision_analyzer()

        metadata = analyzer.analyze_with_vision_model([], ["a", "", "b"], model="m")

        client.chat_with_images.assert_not_called()
        assert metadata.full_text == "a\nb"
        assert metadata.source_images == []


class TestBottleAnalyzerVisionSelection:
    """Photo selection, model precedence and failure handling in process_single_bottle."""

    def test_selects_four_non_bill_images_primary_first(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Bottle")
        images = [
            Image(path="/1.jpg", category=ImageCategory.ANGLE),
            Image(path="/bill.jpg", category=ImageCategory.BILL),
            Image(path="/primary.jpg", category=ImageCategory.ANGLE, is_primary=True),
            Image(path="/3.jpg", category=ImageCategory.BOX),
            Image(path="/4.jpg", category=ImageCategory.DETAIL),
            Image(path="/5.jpg", category=ImageCategory.ANGLE),
        ]

        bottle_dict, metadata = analyzer.process_single_bottle(images, vision_model="m")

        sent = client.chat_with_images.call_args.kwargs["image_paths"]
        assert sent == [Path("/primary.jpg"), Path("/1.jpg"), Path("/3.jpg"), Path("/4.jpg")]
        # Every photo of the group is kept, primary first.
        expected = ["/primary.jpg", "/1.jpg", "/bill.jpg", "/3.jpg", "/4.jpg", "/5.jpg"]
        assert metadata.source_images == expected
        assert bottle_dict["images"] == expected

    def test_bill_string_category_skipped(self):
        analyzer, _ = _vision_analyzer()
        images = [
            Image(path="/bill.jpg", category="BILL"),
            Image(path="/bottle.jpg", category="BOTTLE"),
        ]
        assert analyzer.select_vision_images(images) == [images[1]]

    def test_duplicates_and_burst_retakes_skipped(self):
        analyzer, _ = _vision_analyzer()
        front = Image(path="/front.jpg")
        retake = Image(path="/retake.jpg", near_duplicate_of="/front.jpg")
        copy = Image(path="/copy.jpg", is_duplicate=True, duplicate_of_id="/front.jpg")
        back = Image(path="/back.jpg")
        assert analyzer.select_vision_images([front, retake, copy, back]) == [front, back]

    def test_only_retakes_left_are_still_used(self):
        analyzer, _ = _vision_analyzer()
        retake = Image(path="/retake.jpg", near_duplicate_of="/gone.jpg")
        assert analyzer.select_vision_images([retake]) == [retake]

    def test_only_bills_are_used_when_nothing_else(self):
        analyzer, _ = _vision_analyzer()
        images = [Image(path="/bill.jpg", category=ImageCategory.BILL)]
        assert analyzer.select_vision_images(images) == images

    def test_ocr_hints_aligned_with_selected_paths(self):
        extractor = Mock(spec=TextExtractor)
        extractor.extract_with_retry.side_effect = ["TEXT A", "", "TEXT C"]
        analyzer, client = _vision_analyzer(text_extractor=extractor)
        client.chat_with_images.return_value = _reply(name="Bottle")
        images = [
            Image(path="/a.jpg", category=ImageCategory.ANGLE),
            Image(path="/b.jpg", category=ImageCategory.ANGLE),
            Image(path="/c.jpg", category=ImageCategory.ANGLE, is_primary=True),
        ]

        with patch.object(analyzer, "analyze_with_vision_model", wraps=analyzer.analyze_with_vision_model) as spy:
            _, metadata = analyzer.process_single_bottle(images, vision_model="m")

        paths, hints, model = spy.call_args.args
        assert paths == [Path("/c.jpg"), Path("/a.jpg"), Path("/b.jpg")]
        assert hints == ["TEXT C", "TEXT A", ""]
        assert model == "m"
        prompt = client.chat_with_images.call_args.kwargs["prompt"]
        assert '<ocr photo="1">\nTEXT C\n</ocr>' in prompt
        assert '<ocr photo="2">\nTEXT A\n</ocr>' in prompt
        # OCR of all photos, in input order, is kept in the metadata.
        assert metadata.full_text == "TEXT A\nTEXT C"
        assert metadata.selected_text == "TEXT A\nTEXT C"

    def test_bottle_dict_has_model_used_and_upper_confidence(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Bottle", confidence="medium")

        bottle_dict, metadata = analyzer.process_single_bottle(
            [Image(path="/a.jpg", category=ImageCategory.ANGLE)], vision_model="m"
        )

        assert bottle_dict["model_used"] == "m"
        assert bottle_dict["confidence"] == "MEDIUM"
        assert bottle_dict["quantity"] == 1
        assert metadata.vision_model == "m"

    def test_confidence_falls_back_to_completeness(self):
        analyzer, _ = _vision_analyzer()
        metadata = ExtractedMetadata(name="X")
        with patch.object(analyzer, "analyze_with_vision_model", return_value=metadata):
            bottle_dict, _ = analyzer.process_single_bottle([Image(path="/a.jpg")], vision_model="m")
        assert bottle_dict["confidence"] == "LOW"

    def test_vision_failure_propagates_and_records_failed_usage(self):
        selector = Mock()
        selector.get_model.return_value = "selector-model"
        analyzer, client = _vision_analyzer(model_selector=selector)
        client.chat_with_images.side_effect = VisionTimeoutError("too slow")

        with pytest.raises(VisionTimeoutError):
            analyzer.process_single_bottle([Image(path="/a.jpg")], vision_model="m")

        selector.record_usage.assert_called_once()
        kwargs = selector.record_usage.call_args.kwargs
        assert kwargs["stage"] is ModelStage.VISION_ANALYSIS
        assert kwargs["model"] == "m"
        assert kwargs["success"] is False
        assert "VisionTimeoutError: too slow" in kwargs["error"]

    def test_success_records_usage_with_tokens(self):
        selector = Mock()
        analyzer, client = _vision_analyzer(model_selector=selector)
        client.chat_with_images.return_value = _reply(name="Bottle")

        analyzer.process_single_bottle([Image(path="/a.jpg")], vision_model="m")

        kwargs = selector.record_usage.call_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["error"] is None
        assert kwargs["input_tokens"] == 10
        assert kwargs["output_tokens"] == 5
        assert "fallback_used" not in kwargs

    def test_explicit_model_beats_selector(self):
        selector = Mock()
        selector.get_model.return_value = "selector-model"
        analyzer, client = _vision_analyzer(model_selector=selector)
        client.chat_with_images.return_value = _reply(name="Bottle")

        analyzer.process_single_bottle([Image(path="/a.jpg")], vision_model="explicit")

        assert client.chat_with_images.call_args.kwargs["model"] == "explicit"

    def test_selector_used_when_no_explicit_model(self):
        selector = Mock()
        selector.get_model.return_value = "selector-model"
        analyzer, client = _vision_analyzer(model_selector=selector)
        client.chat_with_images.return_value = _reply(name="Bottle")

        analyzer.process_single_bottle([Image(path="/a.jpg")])

        selector.get_model.assert_called_with(ModelStage.VISION_ANALYSIS)
        assert client.chat_with_images.call_args.kwargs["model"] == "selector-model"

    def test_default_model_without_selector(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="Bottle")

        analyzer.process_single_bottle([Image(path="/a.jpg")])

        assert client.chat_with_images.call_args.kwargs["model"] == DEFAULT_VISION_MODEL

    def test_selector_returning_non_string_ignored(self):
        selector = Mock()  # get_model returns a Mock, not a model name
        analyzer, _ = _vision_analyzer(model_selector=selector)
        assert analyzer._resolve_vision_model(None) == DEFAULT_VISION_MODEL


class TestBottleAnalyzerBuildBottleDict:
    """Test _build_bottle_dict method."""

    def test_build_bottle_dict_complete_metadata(self):
        """Test building dict with complete metadata."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Blanton's",
            distillery="Buffalo Trace",
            label_price="59.99",
            specific_release="Single Barrel",
            proof="93 Proof",
            age="No Age Statement",
            batch_number="B123",
            tasting_notes="Caramel and vanilla",
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = [Image(path="/img1.jpg", category="BOTTLE")]

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        assert bottle_dict["name"] == "Blanton's"
        assert bottle_dict["distillery"] == "Buffalo Trace"
        assert bottle_dict["label_price"] == "59.99"
        assert bottle_dict["msrp"] is None
        assert bottle_dict["specific_release"] == "Single Barrel"
        assert bottle_dict["proof"] == "93 Proof"
        assert bottle_dict["age"] == "No Age Statement"
        assert bottle_dict["batch_number"] == "B123"
        assert bottle_dict["tasting_notes"] == "Caramel and vanilla"
        assert bottle_dict["quantity"] == 1
        assert len(bottle_dict["images"]) == 1

    def test_build_bottle_dict_missing_name_uses_fallback(self):
        """Test missing name uses distillery + release fallback."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="",  # Empty
            distillery="Buffalo Trace",
            label_price="59.99",
            specific_release="Single Barrel",
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = [Image(path="/img.jpg", category="BOTTLE")]

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        assert bottle_dict["name"] == "Buffalo Trace Single Barrel"

    def test_build_bottle_dict_no_fallback_available(self):
        """Test when no fallback available, uses 'Unknown Bottle'."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="",  # Empty
            distillery="",  # Empty
            label_price="",
            specific_release="",  # Empty
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = [Image(path="/img.jpg", category="BOTTLE")]

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        assert bottle_dict["name"] == "Unknown Bottle"

    def test_build_bottle_dict_partial_fallback(self):
        """Test partial fallback (distillery only)."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="",  # Empty
            distillery="Test Distillery",
            label_price="29.99",
            specific_release="",  # Empty
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = [Image(path="/img.jpg", category="BOTTLE")]

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        assert bottle_dict["name"] == "Unknown Bottle"  # Needs both distillery AND release

    def test_build_bottle_dict_defaults_for_missing_fields(self):
        """Test default values for missing required fields."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = []

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        assert bottle_dict["name"] == "Unknown Bottle"
        assert bottle_dict["distillery"] == "Unknown Distillery"
        assert bottle_dict["msrp"] is None  # unknown, not $0.00
        assert bottle_dict["label_price"] is None
        assert bottle_dict["specific_release"] == ""
        assert bottle_dict["quantity"] == 1

    def test_build_bottle_dict_empty_optional_fields_omitted(self):
        """Test empty optional fields are omitted from dict."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            label_price="49.99",
            specific_release="Test",
            proof="",  # Empty
            age=None,  # None
            batch_number="",  # Empty
            tasting_notes="",  # Empty
            full_text="",
            selected_text="",
            source_images=[],
        )

        images = [Image(path="/img.jpg", category="BOTTLE")]

        bottle_dict = analyzer._build_bottle_dict(metadata, images)

        # Optional fields with empty values should not be in dict
        assert "proof" not in bottle_dict
        assert "age" not in bottle_dict
        assert "batch_number" not in bottle_dict
        assert "tasting_notes" not in bottle_dict


class TestBottleAnalyzerOcrSemantics:
    """OCR call contract between BottleAnalyzer and TextExtractor (B5)."""

    @staticmethod
    def _analyzer(text_extractor):
        analyzer = BottleAnalyzer(
            text_extractor=text_extractor,
            ollama_client=Mock(spec=OllamaClient),
        )
        metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            label_price="29.99",
            full_text="",
            selected_text="",
            source_images=["/a.jpg"],
        )
        return analyzer, metadata

    @patch("image_processor.bottle_analyzer.OllamaClient")
    def test_default_text_extractor_gets_the_retry_policy(self, mock_ollama):
        """OCR learns only whether retry is on (it keeps its own fixed 2 attempts)."""
        with patch("image_processor.bottle_analyzer.TextExtractor") as mock_extractor:
            BottleAnalyzer(retry_policy=RetryPolicy.DISABLED)
        mock_extractor.assert_called_once_with(RetryPolicy.DISABLED)

    def test_ocr_called_with_path_only(self):
        """is_primary is not an orientation hint; OCR decides rotation itself."""
        extractor = Mock(spec=TextExtractor)
        extractor.extract_with_retry.return_value = "BOURBON"
        analyzer, metadata = self._analyzer(extractor)

        with patch.object(analyzer, "analyze_with_vision_model", return_value=metadata) as vision:
            analyzer.process_single_bottle(
                [Image(path="/a.jpg", category="BOTTLE", is_primary=True)]
            )

        extractor.extract_with_retry.assert_called_once_with(Path("/a.jpg"))
        assert vision.call_args.args[1] == ["BOURBON"]

    def test_configuration_error_stops_ocr_for_bottle(self, caplog):
        """A missing tesseract fails every image the same way: log once, stop OCR."""
        from lib.exceptions import ConfigurationError

        extractor = Mock(spec=TextExtractor)
        extractor.extract_with_retry.side_effect = ConfigurationError("no tesseract")
        analyzer, metadata = self._analyzer(extractor)
        images = [
            Image(path="/a.jpg", category="BOTTLE"),
            Image(path="/b.jpg", category="BOTTLE"),
            Image(path="/c.jpg", category="BOTTLE"),
        ]

        with (
            patch.object(analyzer, "analyze_with_vision_model", return_value=metadata) as vision,
            caplog.at_level("ERROR", logger="image_processor.bottle_analyzer"),
        ):
            analyzer.process_single_bottle(images)

        assert extractor.extract_with_retry.call_count == 1
        assert sum("OCR unavailable" in r.message for r in caplog.records) == 1
        vision.assert_called_once()  # the vision model still runs without OCR
        assert vision.call_args.args[1] == ["", "", ""]

    def test_other_ocr_errors_logged_and_continue(self, caplog):
        extractor = Mock(spec=TextExtractor)
        extractor.extract_with_retry.side_effect = [OSError("broken"), "KENTUCKY"]
        analyzer, metadata = self._analyzer(extractor)

        with (
            patch.object(analyzer, "analyze_with_vision_model", return_value=metadata) as vision,
            caplog.at_level("WARNING", logger="image_processor.bottle_analyzer"),
        ):
            analyzer.process_single_bottle(
                [Image(path="/a.jpg", category="BOTTLE"), Image(path="/b.jpg", category="BOTTLE")]
            )

        assert extractor.extract_with_retry.call_count == 2
        assert any("OCR failed for /a.jpg: OSError: broken" in r.message for r in caplog.records)
        assert vision.call_args.args[1] == ["", "KENTUCKY"]


class TestBottleAnalyzerRetryPolicy:
    """The vision request runs under the run's RetryPolicy; OCR runs once per bottle."""

    @staticmethod
    def _images():
        return [Image(path="/p/a.jpg", category="BOTTLE", is_primary=True), Image(path="/p/b.jpg", category="BOTTLE")]

    def test_two_unusable_answers_then_success_with_policy_of_3(self):
        analyzer, client = _vision_analyzer(retry_policy=RetryPolicy(max_attempts=3))
        client.chat_with_images.side_effect = [
            _reply(content=""), _reply(proof=90), _reply(name="Third Time", same_bottle=True),
        ]

        bottle_dict, metadata = analyzer.process_single_bottle(self._images(), vision_model="m")

        assert client.chat_with_images.call_count == 3
        assert metadata.vision_attempts == 3
        assert "Vision answer was usable only on attempt 3; review the fields" in metadata.warnings
        assert bottle_dict["name"] == "Third Time"
        assert metadata.to_dict()["vision_attempts"] == 3

    def test_no_retry_makes_one_call_then_raises(self):
        analyzer, client = _vision_analyzer(retry_policy=RetryPolicy.DISABLED)
        client.chat_with_images.side_effect = [_reply(content=""), _reply(name="never reached")]

        with pytest.raises(VisionResponseError) as excinfo:
            analyzer.process_single_bottle(self._images(), vision_model="m")

        assert client.chat_with_images.call_count == 1
        assert excinfo.value.attempts == 1

    def test_ocr_runs_once_across_vision_retries(self):
        extractor = Mock(spec=TextExtractor)
        extractor.extract_with_retry.side_effect = lambda path: f"OCR {Path(path).name}"
        analyzer, client = _vision_analyzer(text_extractor=extractor, retry_policy=RetryPolicy(max_attempts=3))
        client.chat_with_images.side_effect = [_reply(content=""), _reply(content=""), _reply(name="X")]

        _, metadata = analyzer.process_single_bottle(self._images(), vision_model="m")

        assert extractor.extract_with_retry.call_count == 2  # once per photo, not per attempt
        prompts = [c.kwargs["prompt"] for c in client.chat_with_images.call_args_list]
        assert len(prompts) == 3 and len(set(prompts)) == 1  # the same OCR hints every time
        assert "OCR a.jpg" in prompts[0]
        assert metadata.ocr_by_image == {"a.jpg": "OCR a.jpg", "b.jpg": "OCR b.jpg"}

    def test_reviewer_note_reaches_the_prompt(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="X")

        analyzer.process_single_bottle(self._images(), vision_model="m", reviewer_note="Batch is on the neck tag")

        prompt = client.chat_with_images.call_args.kwargs["prompt"]
        assert "Reviewer note" in prompt
        assert "Batch is on the neck tag" in prompt

    def test_no_note_section_without_a_note(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="X")
        analyzer.process_single_bottle(self._images(), vision_model="m")
        assert "Reviewer note" not in client.chat_with_images.call_args.kwargs["prompt"]

    def test_label_price_kept_and_msrp_unknown(self):
        analyzer, client = _vision_analyzer()
        client.chat_with_images.return_value = _reply(name="X", label_price=59.99)

        bottle_dict, metadata = analyzer.process_single_bottle(self._images(), vision_model="m")

        assert metadata.label_price == "59.99"
        assert bottle_dict["label_price"] == "59.99"
        assert bottle_dict["msrp"] is None

    def test_cancelled_run_starts_no_vision_request(self):
        from image_processor.bottle_analyzer import AnalysisCancelledError

        analyzer, client = _vision_analyzer()
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(AnalysisCancelledError):
            analyzer.process_single_bottle(self._images(), vision_model="m", cancel=cancel)
        client.chat_with_images.assert_not_called()

    def test_last_vision_result_is_per_thread(self):
        analyzer, _ = _vision_analyzer()
        analyzer.last_vision_result = "main"
        seen = {}
        set_in_worker = threading.Event()

        def worker():
            seen["before"] = analyzer.last_vision_result
            analyzer.last_vision_result = "worker"
            seen["after"] = analyzer.last_vision_result
            set_in_worker.set()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert set_in_worker.is_set()
        assert seen == {"before": None, "after": "worker"}
        assert analyzer.last_vision_result == "main"
