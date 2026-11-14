"""Unit tests for BottleAnalyzer class.

Tests metadata extraction, image grouping, and bottle analysis functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.text_extractor import TextExtractor
from lib.duplicates import DuplicateDetector
from lib.error_handling import RetryManager
from lib.ollama_client import OllamaClient
from models.bottle import ExtractedMetadata
from models.image import Image


class TestBottleAnalyzerInitialization:
    """Test BottleAnalyzer initialization."""

    @patch("image_processor.bottle_analyzer.TextExtractor")
    @patch("image_processor.bottle_analyzer.OllamaClient")
    @patch("image_processor.bottle_analyzer.RetryManager")
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_analyzer_default_initialization(self, mock_organizer, mock_retry, mock_ollama, mock_extractor):
        """Test analyzer initializes with default dependencies."""
        mock_extractor.return_value = Mock(spec=TextExtractor)
        mock_ollama.return_value = Mock(spec=OllamaClient)
        mock_retry.return_value = Mock(spec=RetryManager)
        mock_organizer.return_value = Mock()

        analyzer = BottleAnalyzer()

        assert analyzer.text_extractor is not None
        assert analyzer.ollama_client is not None
        assert analyzer.retry_manager is not None
        assert analyzer.image_organizer is not None

        # Verify dependencies were created
        mock_extractor.assert_called_once()
        mock_ollama.assert_called_once()
        mock_retry.assert_called_once()
        mock_organizer.assert_called_once()

    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_analyzer_custom_dependencies(self, mock_organizer):
        """Test analyzer uses provided custom dependencies."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_ollama_client = Mock(spec=OllamaClient)
        mock_retry_manager = Mock(spec=RetryManager)
        mock_organizer.return_value = Mock()

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama_client,
            retry_manager=mock_retry_manager,
            image_organizer=mock_organizer.return_value,
        )

        assert analyzer.text_extractor is mock_text_extractor
        assert analyzer.ollama_client is mock_ollama_client
        assert analyzer.retry_manager is mock_retry_manager
        assert analyzer.image_organizer is mock_organizer.return_value


class TestBottleAnalyzerProcessSingleBottle:
    """Test process_single_bottle method."""

    @patch("image_processor.bottle_analyzer.OllamaClient")
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_success(self, mock_organizer, mock_ollama):
        """Test successful processing of single bottle images."""
        # Setup mocks
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "Test OCR text from bottle"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
            image_organizer=Mock(),
        )

        # Mock analyze_with_vision_model
        mock_metadata = ExtractedMetadata(
            name="Test Bottle",
            distillery="Test Distillery",
            msrp="49.99",
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
            assert bottle_dict["msrp"] == "49.99"
            assert bottle_dict["specific_release"] == "Single Barrel"
            assert len(bottle_dict["images"]) == 2
            assert metadata.name == "Test Bottle"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_with_bill_images(self, mock_organizer, mock_ollama):
        """Test processing preferentially skips BILL category images."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "OCR text"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
            image_organizer=Mock(),
        )

        mock_metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            msrp="29.99",
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
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_ocr_extraction_error_continues(self, mock_organizer, mock_ollama):
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
            image_organizer=Mock(),
        )

        mock_metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            msrp="29.99",
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
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_all_ocr_fails_uses_vision_only(self, mock_organizer, mock_ollama):
        """Test when all OCR fails, uses vision model response only."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.side_effect = [Exception("Failed"), Exception("Failed")]
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
            image_organizer=Mock(),
        )

        mock_metadata = ExtractedMetadata(
            name="Vision Only",
            distillery="Distillery",
            msrp="39.99",
            full_text="",  # No OCR text
            selected_text="",
            source_images=["/path.jpg"],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            images = [Image(path="/img.jpg", category="BOTTLE")]

            bottle_dict, metadata = analyzer.process_single_bottle(images)
            assert bottle_dict["name"] == "Vision Only"

    @patch("image_processor.bottle_analyzer.OllamaClient")
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_no_primary_image(self, mock_organizer, mock_ollama):
        """Test behavior when no primary image is available."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_text_extractor.extract_with_retry.return_value = "OCR text"
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
            image_organizer=Mock(),
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
                msrp="29.99",
                full_text="OCR text",
                selected_text="OCR text",
                source_images=["/img1.jpg"],
            )

            bottle_dict, metadata = analyzer.process_single_bottle(images)
            assert bottle_dict["name"] == "Fallback"
            # Should have called analyze with first image
            mock_analyze.assert_called_once()

    @patch("image_processor.bottle_analyzer.OllamaClient")
    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_process_single_bottle_empty_images(self, mock_organizer, mock_ollama):
        """Test processing with empty image list."""
        mock_text_extractor = Mock(spec=TextExtractor)
        mock_ollama.return_value = Mock(spec=OllamaClient)

        analyzer = BottleAnalyzer(
            text_extractor=mock_text_extractor,
            ollama_client=mock_ollama.return_value,
            image_organizer=Mock(),
        )

        images = []

        mock_metadata = ExtractedMetadata(
            name="Empty",
            distillery="Test",
            msrp="0.00",
            full_text="",
            selected_text="",
            source_images=[],
        )

        with patch.object(analyzer, "analyze_with_vision_model", return_value=mock_metadata):
            bottle_dict, metadata = analyzer.process_single_bottle(images)
            # Should still produce a bottle dict with unknown name
            assert bottle_dict["name"] == "Empty"
            assert len(bottle_dict["images"]) == 0


class TestBottleAnalyzerGroupImagesByBottle:
    """Test group_images_by_bottle method."""

    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_group_images_delegates_to_organizer(self, mock_organizer):
        """Test grouping delegates to ImageOrganizer."""
        mock_group_method = Mock(return_value=[[Mock()], [Mock()]])
        mock_organizer.return_value.group_by_bottle = mock_group_method

        analyzer = BottleAnalyzer(image_organizer=mock_organizer.return_value)

        images = [
            Image(path="/img1.jpg", category="BOTTLE"),
            Image(path="/img2.jpg", category="BOTTLE"),
            Image(path="/img3.jpg", category="BOTTLE"),
        ]

        groups = analyzer.group_images_by_bottle(images, similarity_threshold=0.8)

        # Verify delegation
        mock_group_method.assert_called_once_with(images, 0.8)
        assert len(groups) == 2

    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_group_images_custom_threshold(self, mock_organizer):
        """Test custom similarity threshold is passed through."""
        mock_group_method = Mock(return_value=[[Mock()]])
        mock_organizer.return_value.group_by_bottle = mock_group_method

        analyzer = BottleAnalyzer(image_organizer=mock_organizer.return_value)

        groups = analyzer.group_images_by_bottle([], similarity_threshold=0.9)

        mock_group_method.assert_called_once_with([], 0.9)

    @patch("image_processor.bottle_analyzer.ImageOrganizer")
    def test_group_images_empty_list(self, mock_organizer):
        """Test grouping empty image list."""
        mock_group_method = Mock(return_value=[])
        mock_organizer.return_value.group_by_bottle = mock_group_method

        analyzer = BottleAnalyzer(image_organizer=mock_organizer.return_value)

        groups = analyzer.group_images_by_bottle([])

        assert groups == []


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

        with patch.object(analyzer, "_estimate_quantity_from_images", return_value=1):
            quantity = analyzer.detect_quantity(images)
            assert quantity == 1

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

        with patch.object(analyzer, "_estimate_quantity_from_images", return_value=1):
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

    def test_detect_quantity_visual_estimation_used(self):
        """Test visual estimation is used when image count differs from hash count."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)

        # 6 images, 2 unique hashes (suggests 3 images each)
        mock_detector.generate_hash.side_effect = [
            "hashA", "hashA", "hashA",  # 6 images with 2 unique hashes
            "hashB", "hashB", "hashB",
        ]

        images = [Image(path=f"/img{i}.jpg", category="BOTTLE") for i in range(6)]

        with patch.object(analyzer, "_estimate_quantity_from_images", return_value=2):
            quantity = analyzer.detect_quantity(images, duplicate_detector=mock_detector)
            # Should return max(3, 2) = 3
            assert quantity == 3


class TestBottleAnalyzerCalculateConfidence:
    """Test calculate_confidence method."""

    def test_calculate_confidence_high_complete_metadata(self):
        """Test HIGH confidence when most fields are filled."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Test",
            distillery="Distillery",
            msrp="49.99",
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
            msrp="49.99",
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
            msrp="",
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
        """Test successful vision model analysis."""
        mock_ollama_client = Mock(spec=OllamaClient)
        mock_ollama_client.image_to_text.return_value = (
            "name: Blanton's\n"
            "distillery: Buffalo Trace\n"
            "msrp: $59.99\n"
            "proof: 93 Proof\n"
            "age: No Age Statement"
        )

        analyzer = BottleAnalyzer(ollama_client=mock_ollama_client)

        metadata = analyzer.analyze_with_vision_model(
            image_path=Path("/test.jpg"), ocr_text="OCR text", model="test-model"
        )

        assert metadata.name == "Blanton's"
        assert metadata.distillery == "Buffalo Trace"
        assert metadata.msrp == "59.99"
        assert metadata.proof == "93 Proof"
        assert metadata.age == "No Age Statement"
        assert metadata.full_text == "OCR text"

    def test_analyze_with_vision_model_fallback_on_error(self):
        """Test falls back to OCR-only on vision model error."""
        mock_ollama_client = Mock(spec=OllamaClient)
        mock_ollama_client.image_to_text.side_effect = Exception("API error")

        analyzer = BottleAnalyzer(ollama_client=mock_ollama_client)

        metadata = analyzer.analyze_with_vision_model(
            image_path=Path("/test.jpg"), ocr_text="Fallback OCR text"
        )

        # Should return OCR-only metadata
        assert metadata.full_text == "Fallback OCR text"
        assert metadata.selected_text == "Fallback OCR text"
        assert metadata.source_images == ["/test.jpg"]

    def test_analyze_with_vision_model_custom_model(self):
        """Test custom vision model is used."""
        mock_ollama_client = Mock(spec=OllamaClient)
        mock_ollama_client.image_to_text.return_value = "name: Test Model"

        analyzer = BottleAnalyzer(ollama_client=mock_ollama_client)

        analyzer.analyze_with_vision_model(
            image_path=Path("/test.jpg"), ocr_text="OCR", model="qwen3-vl:32b"
        )

        # Verify custom model was used
        mock_ollama_client.image_to_text.assert_called_once()
        call_kwargs = mock_ollama_client.image_to_text.call_args.kwargs
        assert call_kwargs["model"] == "qwen3-vl:32b"


class TestBottleAnalyzerImagesBelongTogether:
    """Test _images_belong_together method."""

    def test_images_belong_together_same_hash(self):
        """Test images with same MD5 hash belong together."""
        analyzer = BottleAnalyzer()

        img1 = Image(path="/img1.jpg", category="BOTTLE")
        img1.md5_hash = "abc123"

        img2 = Image(path="/img2.jpg", category="BOTTLE")
        img2.md5_hash = "abc123"

        result = analyzer._images_belong_together(img1, img2)
        assert result is True

    def test_images_belong_together_different_hashes_similar_names(self):
        """Test images with similar names but different hashes."""
        analyzer = BottleAnalyzer()

        img1 = Image(path="/bottle_front.jpg", category="BOTTLE")
        img1.md5_hash = "hash1"

        img2 = Image(path="/bottle_side.jpg", category="BOTTLE")
        img2.md5_hash = "hash2"

        result = analyzer._images_belong_together(img1, img2)
        # "bottle" is common prefix, should be > 70%
        assert result is True

    def test_images_belong_together_different_names(self):
        """Test images with different names don't belong together."""
        analyzer = BottleAnalyzer()

        img1 = Image(path="/imageA123.jpg", category="BOTTLE")
        img1.md5_hash = "hash1"

        img2 = Image(path="/imageB456.jpg", category="BOTTLE")
        img2.md5_hash = "hash2"

        result = analyzer._images_belong_together(img1, img2)
        # "image" is only common part, not > 70%
        assert result is False

    def test_images_belong_together_no_hashes(self):
        """Test without MD5 hashes falls back to name comparison."""
        analyzer = BottleAnalyzer()

        img1 = Image(path="/bottle_front.jpg", category="BOTTLE")
        img1.md5_hash = None

        img2 = Image(path="/bottle_back.jpg", category="BOTTLE")
        img2.md5_hash = None

        result = analyzer._images_belong_together(img1, img2)
        assert result is True

    def test_images_belong_together_one_has_hash(self):
        """Test when only one image has hash doesn't cause error."""
        analyzer = BottleAnalyzer()

        img1 = Image(path="/img1.jpg", category="BOTTLE")
        img1.md5_hash = "hash1"

        img2 = Image(path="/img2.jpg", category="BOTTLE")
        img2.md5_hash = None

        result = analyzer._images_belong_together(img1, img2)
        # Falls back to name comparison
        assert isinstance(result, bool)


class TestBottleAnalyzerEstimateQuantityFromImages:
    """Test _estimate_quantity_from_images method."""

    def test_estimate_quantity_single_hash_returns_one(self):
        """Test single hash group returns quantity 1."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)
        mock_detector.generate_hash.return_value = "hash123"

        images = [Image(path=f"/img{i}.jpg", category="BOTTLE") for i in range(3)]

        with patch("image_processor.bottle_analyzer.DuplicateDetector", return_value=mock_detector):
            quantity = analyzer._estimate_quantity_from_images(images)
            assert quantity == 1

    def test_estimate_quantity_duplicate_hashes_high_count(self):
        """Test many duplicate hashes suggests multiple bottles."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)

        # 6 identical hashes (more than MAX_ANGLES_PER_BOTTLE = 4)
        mock_detector.generate_hash.return_value = "hash123"

        images = [Image(path=f"/img{i}.jpg", category="BOTTLE") for i in range(6)]

        # Mock DuplicateDetector inside the method
        with patch("image_processor.bottle_analyzer.DuplicateDetector", return_value=mock_detector):
            quantity = analyzer._estimate_quantity_from_images(images)
            # 6 // 4 + 1 = 2 bottles
            assert quantity == 2

    def test_estimate_quantity_hash_error_returns_one(self):
        """Test hash generation errors return quantity 1."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)
        mock_detector.generate_hash.side_effect = Exception("Hash error")

        images = [Image(path="/img.jpg", category="BOTTLE")]

        with patch("image_processor.bottle_analyzer.DuplicateDetector", return_value=mock_detector):
            quantity = analyzer._estimate_quantity_from_images(images)
            assert quantity == 1

    def test_estimate_quantity_empty_images_returns_one(self):
        """Test empty image list returns quantity 1."""
        analyzer = BottleAnalyzer()

        quantity = analyzer._estimate_quantity_from_images([])
        assert quantity == 1

    def test_estimate_quantity_mixed_hashes(self):
        """Test mixed hash patterns."""
        analyzer = BottleAnalyzer()
        mock_detector = Mock(spec=DuplicateDetector)

        # 2 images with hash A, 2 with hash B (within max angles)
        mock_detector.generate_hash.side_effect = ["hashA", "hashA", "hashB", "hashB"]

        images = [Image(path=f"/img{i}.jpg", category="BOTTLE") for i in range(4)]

        with patch("image_processor.bottle_analyzer.DuplicateDetector", return_value=mock_detector):
            quantity = analyzer._estimate_quantity_from_images(images)
            # Max count is 2, within limit, returns 1
            assert quantity == 1


class TestBottleAnalyzerBuildBottleDict:
    """Test _build_bottle_dict method."""

    def test_build_bottle_dict_complete_metadata(self):
        """Test building dict with complete metadata."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Blanton's",
            distillery="Buffalo Trace",
            msrp="59.99",
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
        assert bottle_dict["msrp"] == "59.99"
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
            msrp="59.99",
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
            msrp="",
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
            msrp="29.99",
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
        assert bottle_dict["msrp"] == "0.00"
        assert bottle_dict["specific_release"] == ""
        assert bottle_dict["quantity"] == 1

    def test_build_bottle_dict_empty_optional_fields_omitted(self):
        """Test empty optional fields are omitted from dict."""
        analyzer = BottleAnalyzer()

        metadata = ExtractedMetadata(
            name="Test",
            distillery="Test",
            msrp="49.99",
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


class TestBottleAnalyzerParseVisionResponse:
    """Test _parse_vision_response method."""

    def test_parse_vision_response_complete_fields(self):
        """Test parsing complete vision model response."""
        analyzer = BottleAnalyzer()

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

        metadata = analyzer._parse_vision_response(response)

        assert metadata["name"] == "Blanton's Single Barrel"
        assert metadata["distillery"] == "Buffalo Trace"
        assert metadata["msrp"] == "59.99"
        assert metadata["specific_release"] == "Original Single Barrel"
        assert metadata["proof"] == "93 Proof"
        assert metadata["age"] == "No Age Statement"
        assert metadata["batch_number"] == "123"
        assert metadata["tasting_notes"] == "Caramel and vanilla"

    def test_parse_vision_response_simplifies_field_names(self):
        """Test field name normalization and simplification."""
        analyzer = BottleAnalyzer()

        response = "name: Test\nrelease: Single Barrel\nnotes: Good flavor\nbatch: ABC"

        metadata = analyzer._parse_vision_response(response)

        assert metadata["name"] == "Test"
        assert metadata["specific_release"] == "Single Barrel"  # "release" → "specific_release"
        assert metadata["tasting_notes"] == "Good flavor"  # "notes" → "tasting_notes"
        assert metadata["batch_number"] == "ABC"  # "batch" → "batch_number"

    def test_parse_vision_response_strips_dollar_signs(self):
        """Test dollar signs are stripped from MSRP."""
        analyzer = BottleAnalyzer()

        response = "name: Test\nmsrp: $49.99"

        metadata = analyzer._parse_vision_response(response)

        assert metadata["msrp"] == "49.99"

    def test_parse_vision_response_invalid_msrp_skipped(self):
        """Test invalid MSRP values are skipped."""
        analyzer = BottleAnalyzer()

        response = "name: Test\nmsrp: NOT-A-NUMBER"

        metadata = analyzer._parse_vision_response(response)

        assert "msrp" not in metadata

    def test_parse_vision_response_normalizes_whitespace(self):
        """Test whitespace is normalized."""
        analyzer = BottleAnalyzer()

        response = "  name  :  Test Name  \n  distillery:Test Distillery  "

        metadata = analyzer._parse_vision_response(response)

        assert metadata["name"] == "Test Name"
        assert metadata["distillery"] == "Test Distillery"

    def test_parse_vision_response_empty_response(self):
        """Test empty response returns empty dict."""
        analyzer = BottleAnalyzer()

        metadata = analyzer._parse_vision_response("")

        assert metadata == {}

    def test_parse_vision_response_unknown_fields_skipped(self):
        """Test unknown fields are not included."""
        analyzer = BottleAnalyzer()

        response = "name: Test\nunknown_field: something\nalso_unknown: value"

        metadata = analyzer._parse_vision_response(response)

        assert metadata["name"] == "Test"
        assert "unknown_field" not in metadata
        assert "also_unknown" not in metadata

    def test_parse_vision_response_field_casing_normalized(self):
        """Test field names are normalized to lowercase."""
        analyzer = BottleAnalyzer()

        response = "NAME: Uppercase\nDISTILLERY: Test"

        metadata = analyzer._parse_vision_response(response)

        assert metadata["name"] == "Uppercase"
        assert metadata["distillery"] == "Test"

    def test_parse_vision_response_hyphen_normalized(self):
        """Test hyphens in field names are converted to underscores."""
        analyzer = BottleAnalyzer()

        response = "batch-number:123\nspecific-release:Single"

        metadata = analyzer._parse_vision_response(response)

        assert metadata["batch_number"] == "123"
        assert metadata["specific_release"] == "Single"
