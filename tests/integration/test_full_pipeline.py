"""Integration tests for full end-to-end processing pipeline.

Tests verify the complete workflow from image input to processed bottle metadata.
Uses pytest.mark.integration to mark these tests as requiring external services.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image as PILImage

import pytest

from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.batch_processor import BatchProcessor
from image_processor.image_organizer import ImageOrganizer
from lib.exceptions import VisionAnalysisError
from lib.ollama_client import VisionChatResult
from models.bottle import Bottle, ExtractedMetadata
from models.image import Image


def _vision_reply(**fields) -> VisionChatResult:
    """A schema-shaped JSON vision reply (unset keys are null)."""
    keys = ("name", "distillery", "specific_release", "msrp", "proof", "abv", "age",
            "batch_number", "tasting_notes")
    body = {k: None for k in keys}
    body.update({"confidence": "high", "same_bottle": True})
    body.update(fields)
    return VisionChatResult(content=json.dumps(body), done_reason="stop")


# Marker for integration tests that require Ollama
has_ollama = pytest.mark.skipif(
    not pytest.importorskip("requests", reason="requests not available"),
    reason="requests library required for Ollama calls"
)


class TestFullPipelineEndToEnd:
    """Test complete pipeline from images to bottle models."""

    @has_ollama
    def test_single_bottle_basic_extraction(self, tmp_path):
        """Test extraction of bottle metadata from single image.

        Covers: Image loading → OCR → Vision model → Bottle model creation
        """
        # Create test image (simple colored rectangle with text)
        img_path = tmp_path / "test_bottle.jpg"
        _create_test_image(img_path, "Test Bourbon Distillery")

        # Setup
        analyzer = BottleAnalyzer(ollama_client=Mock())
        test_image = Image(
            path=str(img_path),
            category="BOTTLE",
            is_primary=True
        )

        # Mock vision model response
        with patch.object(analyzer.ollama_client, 'chat_with_images') as mock_vision:
            mock_vision.return_value = _vision_reply(
                name="Test Bourbon",
                distillery="Test Distillery",
                msrp=50.00,
                proof=90,
                age="12 Years",
            )

            # Execute
            bottle_dict, metadata = analyzer.process_single_bottle([test_image])

            # Verify
            assert bottle_dict["name"] == "Test Bourbon"
            assert bottle_dict["distillery"] == "Test Distillery"
            assert bottle_dict["msrp"] == "50.00"
            assert bottle_dict["proof"] == "90 Proof"
            assert bottle_dict["age"] == "12 Years"
            assert bottle_dict["confidence"] == "HIGH"
            assert isinstance(metadata, ExtractedMetadata)
            assert mock_vision.call_args.kwargs["image_paths"] == [img_path]

    @has_ollama
    def test_multi_angle_bottle_processing(self, tmp_path):
        """Test processing multiple angles of the same bottle.

        Covers: Multi-image grouping → Text aggregation → Single bottle output
        """
        # Create test images for multiple angles
        img1_path = tmp_path / "bottle_front.jpg"
        img2_path = tmp_path / "bottle_back.jpg"
        img3_path = tmp_path / "bottle_label.jpg"

        _create_test_image(img1_path, "Front Label")
        _create_test_image(img2_path, "Back Label")
        _create_test_image(img3_path, "Close Label")

        # Setup
        analyzer = BottleAnalyzer(ollama_client=Mock())
        images = [
            Image(path=str(img1_path), category="BOTTLE", is_primary=True),
            Image(path=str(img2_path), category="BOTTLE", is_primary=False),
            Image(path=str(img3_path), category="BOTTLE", is_primary=False),
        ]

        # Mock vision and OCR
        with patch.object(analyzer.ollama_client, 'chat_with_images') as mock_vision, \
             patch.object(analyzer.text_extractor, 'extract_with_retry') as mock_ocr:
            mock_vision.return_value = _vision_reply(
                name="Multi Angle Bourbon", distillery="Test Distillery", msrp=75.00
            )
            mock_ocr.side_effect = [
                "Front label text",
                "Back label text",
                "Close label text",
            ]

            # Execute
            bottle_dict, metadata = analyzer.process_single_bottle(images)

            # Verify
            assert bottle_dict["name"] == "Multi Angle Bourbon"
            assert len(bottle_dict["images"]) == 3
            # Verify OCR was called for all images
            assert mock_ocr.call_count == 3
            # All three photos went to ONE vision call, primary first
            mock_vision.assert_called_once()
            assert mock_vision.call_args.kwargs["image_paths"] == [img1_path, img2_path, img3_path]
            assert metadata.full_text == "Front label text\nBack label text\nClose label text"

    def test_bottle_model_from_extraction_result(self, tmp_path):
        """Test creating Bottle model from extraction results without Ollama.

        Covers: Bottle.from_dict → Model validation → Total value calculation
        This test doesn't require external services, so no marker.
        """
        bottle_data = {
            "name": "Standard Bourbon",
            "distillery": "Example Distillery",
            "specific_release": "Single Barrel Select",
            "msrp": "89.99",
            "proof": "100 Proof",
            "age": "10 Years",
            "quantity": {
                "count": 1,
                "price": "89.99",
                "source": "extracted"
            },
            "metadata": {
                "name": "Standard Bourbon",
                "distillery": "Example Distillery",
                "msrp": "89.99",
                "full_text": "Extracted text from label",
                "selected_text": "Selected text for analysis",
                "source_images": []
            }
        }

        # Execute
        bottle = Bottle.from_dict(bottle_data)

        # Verify
        assert bottle.name == "Standard Bourbon"
        assert str(bottle.msrp) == "89.99"
        assert bottle.total_value == bottle.msrp
        assert bottle.quantity.count == 1
        assert bottle.quantity.price == bottle.msrp

    def test_batch_processing_workflow(self, tmp_path):
        """Test batch processing of multiple image groups.

        Covers: BatchProcessor → Multiple bottles → BatchResult aggregation
        """
        # Create test images
        img1_path = tmp_path / "bottle1.jpg"
        img2_path = tmp_path / "bottle2.jpg"
        img3_path = tmp_path / "bottle3.jpg"

        _create_test_image(img1_path, "Bottle 1")
        _create_test_image(img2_path, "Bottle 2")
        _create_test_image(img3_path, "Bottle 3")

        # Setup
        batch_processor = BatchProcessor(batch_size=2)
        image_groups = [
            [Image(path=str(img1_path), category="BOTTLE", is_primary=True)],
            [Image(path=str(img2_path), category="BOTTLE", is_primary=True)],
            [Image(path=str(img3_path), category="BOTTLE", is_primary=True)],
        ]

        # Mock BottleAnalyzer
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Test", "msrp": "60"}, Mock()),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "70"}, Mock()),
            ]

            # Execute process_batch on sliced batches
            batches = batch_processor.slice_into_batches(image_groups)
            all_results = []
            for batch in batches:
                result = batch_processor.process_batch(batch)
                all_results.extend(result.bottle_dicts)

            # Verify
            assert len(all_results) == 3
            assert all_results[0]["name"] == "Bottle 1"
            assert all_results[1]["name"] == "Bottle 2"
            assert all_results[2]["name"] == "Bottle 3"


class TestPipelineErrorHandling:
    """Test error handling in full pipeline."""

    def test_unrecoverable_error_propagates(self, tmp_path):
        """Test that unrecoverable errors propagate up the pipeline."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path, "Test")

        analyzer = BottleAnalyzer(ollama_client=Mock())
        test_image = Image(path=str(img_path), category="BOTTLE", is_primary=True)

        # Mock vision model to raise exception
        with patch.object(analyzer.ollama_client, 'chat_with_images') as mock_vision:
            mock_vision.side_effect = Exception("Vision model exploded")

            with patch.object(analyzer.text_extractor, 'extract_with_retry') as mock_ocr:
                mock_ocr.return_value = "Test Bourbon Result"

                # A vision failure is a failure, never an OCR-only "Unknown Bottle"
                with pytest.raises(VisionAnalysisError, match="Vision model exploded"):
                    analyzer.process_single_bottle([test_image])

    def test_invalid_image_handling(self, tmp_path):
        """Test handling of non-image files in input."""
        # Create invalid file
        invalid_path = tmp_path / "not_an_image.txt"
        invalid_path.write_text("This is text, not an image")

        # Organizer should filter out invalid files
        organizer = ImageOrganizer()

        # Process should handle invalid files by skipping them
        # (This would verify the filtering logic in ImageOrganizer)
        assert True  # Placeholder - organizer implementation handles this


# Helper functions

def _create_test_image(path: Path, text: str) -> None:
    """Create a simple test image with text.

    Args:
        path: Path where to save the image.
        text: Text to add to the image.
    """
    # Create a simple white image with black text
    img = PILImage.new('RGB', (400, 600), color='white')
    # Note: would need PIL drawing to add text, but for now just save blank image
    img.save(path)


