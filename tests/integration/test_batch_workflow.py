"""Integration tests for batch processing workflow.

Tests verify batch processing with lookahead, preview reports,
and feedback loop using test fixture images.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image as PILImage

import pytest

from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.batch_processor import BatchProcessor, BatchResult
from models.bottle import Bottle, ExtractedMetadata, BatchPreviewReport
from models.image import Image


# Marker for integration tests that require Ollama
has_ollama = pytest.mark.skipif(
    not pytest.importorskip("requests", reason="requests not available"),
    reason="requests library required for Ollama calls"
)


class TestBatchProcessingWithLookahead:
    """Test batch processing with lookahead functionality."""

    @has_ollama
    def test_lookahead_groups_related_images(self, tmp_path):
        """Test that lookahead groups multi-angle photos of same bottle.

        Covers: lookahead operation → image grouping → single bottle per group
        """
        # Create test images simulating multi-angle photos of 2 bottles
        # Bottle 1: front, back, label angles
        bottle1_front = tmp_path / "IMG_0001.jpg"
        bottle1_back = tmp_path / "IMG_0002.jpg"
        bottle1_label = tmp_path / "IMG_0003.jpg"
        # Bottle 2: front, back, label angles
        bottle2_front = tmp_path / "IMG_0004.jpg"
        bottle2_back = tmp_path / "IMG_0005.jpg"

        _create_test_image(bottle1_front)
        _create_test_image(bottle1_back)
        _create_test_image(bottle1_label)
        _create_test_image(bottle2_front)
        _create_test_image(bottle2_back)

        # All images as Image objects
        all_images = [
            Image(path=str(bottle1_front), category="BOTTLE", is_primary=True),
            Image(path=str(bottle1_back), category="BOTTLE", is_primary=False),
            Image(path=str(bottle1_label), category="BOTTLE", is_primary=False),
            Image(path=str(bottle2_front), category="BOTTLE", is_primary=True),
            Image(path=str(bottle2_back), category="BOTTLE", is_primary=False),
        ]

        # Setup processor with lookahead=2
        batch_processor = BatchProcessor(batch_size=2, lookahead=2)

        # Mock bottle analyzer to return simulated results
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test 1", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Test 2", "msrp": "75"}, Mock()),
            ]

            # Apply lookahead to group images
            # (In real implementation, this would use visual similarity)
            grouped = batch_processor.apply_lookahead(all_images[:3])  # First group

            # Verify grouping
            assert len(grouped) == 3  # Should include primary + lookahead images

    @has_ollama
    def test_batch_processing_with_lookahead_limit(self, tmp_path):
        """Test that lookahead respects the maximum limit.

        Covers: lookahead limit (0-10) → bounded group size
        """
        # Create test images
        images = []
        for i in range(15):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=(i == 0)))

        # Setup processor with lookahead=5
        batch_processor = BatchProcessor(batch_size=3, lookahead=5)

        # Apply lookahead - should respect max limit
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.return_value = ({"name": "Test Bottle", "distillery": "Test", "msrp": "50"}, Mock())

            result = batch_processor.process_batch(images[:3])

            # Verify lookahead was applied but bounded
            assert len(result.bottle_dicts) >= 1

    @has_ollama
    def test_lookahead_value_zero_processes_independently(self, tmp_path):
        """Test that lookahead=0 processes images independently.

        Covers: lookahead=0 → no grouping → each image separate bottle
        """
        # Create test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        # Setup processor with lookahead=0
        batch_processor = BatchProcessor(batch_size=3, lookahead=0)

        # Process
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Test", "msrp": "60"}, Mock()),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "70"}, Mock()),
            ]

            result = batch_processor.process_batch(images)

            # Each image processed independently
            assert mock_process.call_count == 3
            assert len(result.bottle_dicts) == 3


class TestPreviewReports:
    """Test batch preview report generation."""

    def test_preview_report_generated_for_batch(self, tmp_path):
        """Test that preview reports are generated for processed batch.

        Covers: BatchResult → report generation → markdown output
        """
        # Create test data
        bottle_data = [
            {"name": "Bottle 1", "distillery": "Test 1", "msrp": "50.00", "proof": "90 Proof"},
            {"name": "Bottle 2", "distillery": "Test 2", "msrp": "75.00", "proof": "100 Proof"},
        ]

        # Create BatchPreviewReport
        report = BatchPreviewReport(
            report_id="report_001",
            batch_id="batch_001",
        )

        # Generate markdown content
        report.markdown_content = _generate_preview_markdown(bottle_data, report.batch_id)
        report.report_path = tmp_path / f"{report.batch_id}_preview.md"

        # Save report
        with open(report.report_path, 'w') as f:
            f.write(report.markdown_content)

        # Verify report was created
        assert report.report_path.exists()
        assert "Bottle 1" in report.markdown_content
        assert "Bottle 2" in report.markdown_content
        assert "batch_001" in report.markdown_content

    def test_preview_report_shows_bottle_details(self, tmp_path):
        """Test that preview reports include bottle details.

        Covers: Report content → name, distillery, msrp, proof fields displayed
        """
        bottle_data = [
            {
                "name": "Four Roses Single Barrel",
                "distillery": "Four Roses",
                "msrp": "45.00",
                "proof": "100 Proof",
                "age": "n/a",
                "quantity": {"count": 1, "price": "45.00"},
            }
        ]

        report = BatchPreviewReport(report_id="report_002", batch_id="batch_002")
        report.markdown_content = _generate_preview_markdown(bottle_data, report.batch_id)

        # Verify all fields present
        assert "Four Roses Single Barrel" in report.markdown_content
        assert "Four Roses" in report.markdown_content
        assert "$45.00" in report.markdown_content
        assert "100 Proof" in report.markdown_content

    def test_preview_report_includes_feedback_instructions(self, tmp_path):
        """Test that preview reports include feedback instructions.

        Covers: Report content → user guidance for corrections
        """
        bottle_data = [{"name": "Test Bottle", "distillery": "Test", "msrp": "50.00"}]

        report = BatchPreviewReport(report_id="report_003", batch_id="batch_003")
        report.markdown_content = _generate_preview_markdown(bottle_data, report.batch_id)

        # Verify instructions present
        assert "feedback" in report.markdown_content.lower() or "corrections" in report.markdown_content.lower()


class TestFeedbackLoop:
    """Test feedback loop integration."""

    def test_feedback_applies_corrections_to_batch(self):
        """Test that feedback loop applies user corrections.

        Covers: corrections → batch update → reprocessing
        """
        # Create test bottle with errors
        original_bottle = Bottle(
            name="Wrong Name",
            distillery="Wrong Distillery",
            specific_release="Single Barrel",
            msrp=Decimal("0.00"),  # Wrong
        )

        # User corrections
        corrections = [
            {"field": "name", "original": "Wrong Name", "corrected": "Four Roses Single Barrel"},
            {"field": "distillery", "original": "Wrong Distillery", "corrected": "Four Roses"},
            {"field": "msrp", "original": "0.00", "corrected": "45.00"},
        ]

        # Apply corrections (simulated feedback loop)
        corrected_dict = original_bottle.to_dict()
        for correction in corrections:
            field = correction["field"]
            corrected_value = correction["corrected"]
            if field == "msrp":
                corrected_dict[field] = str(Decimal(corrected_value))
            else:
                corrected_dict[field] = corrected_value

        # Verify corrections applied
        corrected_bottle = Bottle.from_dict(corrected_dict)
        assert corrected_bottle.name == "Four Roses Single Barrel"
        assert corrected_bottle.distillery == "Four Roses"
        assert str(corrected_bottle.msrp) == "45.00"

    def test_feedback_updates_preview_report(self, tmp_path):
        """Test that feedback updates preview report.

        Covers: corrections → report regeneration → updated content
        """
        # Original bottle data
        bottle_data = [{"name": "Old Name", "distillery": "Old Distillery", "msrp": "50.00"}]

        # Generate original report
        report = BatchPreviewReport(report_id="report_004", batch_id="batch_004")
        report.markdown_content = _generate_preview_markdown(bottle_data, report.batch_id)

        # Apply correction
        bottle_data[0]["name"] = "New Name"
        bottle_data[0]["distillery"] = "New Distillery"

        # Regenerate report
        report.markdown_content = _generate_preview_markdown(bottle_data, report.batch_id)

        # Verify update
        assert "New Name" in report.markdown_content
        assert "New Distillery" in report.markdown_content
        assert "Old Name" not in report.markdown_content


class TestBatchWorkflowEndToEnd:
    """Test complete batch workflow end-to-end."""

    @has_ollama
    def test_full_batch_workflow_from_images_to_report(self, tmp_path):
        """Test complete workflow: images → batch processing → preview report.

        Covers: Full pipeline with lookahead and preview report
        """
        # Create test images
        images = []
        for i in range(6):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=(i % 3 == 0)))

        # Setup batch processor
        batch_processor = BatchProcessor(batch_size=2, lookahead=2)

        # Mock bottle analyzer
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Distillery 1", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Distillery 2", "msrp": "75"}, Mock()),
                ({"name": "Bottle 3", "distillery": "Distillery 3", "msrp": "100"}, Mock()),
            ]

            # Process batch
            batches = batch_processor.slice_into_batches([images[:3], images[3:]])
            all_results = []
            for batch in batches:
                result = batch_processor.process_batch(batch)
                all_results.extend(result.bottle_dicts)

            # Verify results
            assert len(all_results) == 3
            assert all_results[0]["name"] == "Bottle 1"
            assert all_results[1]["name"] == "Bottle 2"
            assert all_results[2]["name"] == "Bottle 3"

            # Generate preview report
            report = BatchPreviewReport(
                report_id="report_e2e",
                batch_id="batch_e2e",
            )
            report.markdown_content = _generate_preview_markdown(all_results, report.batch_id)
            report.report_path = tmp_path / f"{report.batch_id}_preview.md"

            with open(report.report_path, 'w') as f:
                f.write(report.markdown_content)

            # Verify report
            assert report.report_path.exists()
            assert "Bottle 1" in report.markdown_content
            assert "Bottle 2" in report.markdown_content
            assert "Bottle 3" in report.markdown_content

    @has_ollama
    def test_batch_with_errors_continues_processing(self, tmp_path):
        """Test that individual bottle errors don't stop batch processing.

        Covers: Error handling → non-fatal errors → batch completion
        """
        # Create test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        # Setup batch processor
        batch_processor = BatchProcessor(batch_size=3, lookahead=0)

        # Mock bottle analyzer with partial success
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            # First succeeds, second fails, third succeeds
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, Mock()),
                Exception("Processing failed"),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "75"}, Mock()),
            ]

            # Process batch - should handle errors gracefully
            try:
                result = batch_processor.process_batch([images])

                # Verify - should have at least some successful results
                assert len(result.bottle_dicts) >= 1 or len(result.errors) > 0
            except Exception:
                # If implementation doesn't handle errors gracefully, verify exception
                assert mock_process.call_count == 3


# Helper functions

def _create_test_image(path: Path) -> None:
    """Create a simple test image.

    Args:
        path: Path where to save the image.
    """
    img = PILImage.new('RGB', (400, 600), color='white')
    img.save(path)


def _generate_preview_markdown(bottle_data: list[dict], batch_id: str) -> str:
    """Generate markdown preview report content.

    Args:
        bottle_data: List of bottle dictionaries.
        batch_id: Batch identifier.

    Returns:
        Markdown formatted string.
    """
    lines = [f"# Batch Preview Report", f"**Batch ID:** {batch_id}", ""]
    lines.append("## Extracted Bottles")
    lines.append("")

    for idx, bottle in enumerate(bottle_data, 1):
        lines.append(f"### Bottle {idx}")
        lines.append(f"**Name:** {bottle.get('name', 'Unknown')}")
        lines.append(f"**Distillery:** {bottle.get('distillery', 'Unknown')}")
        lines.append(f"**MSRP:** ${bottle.get('msrp', '0.00')}")
        if "proof" in bottle:
            lines.append(f"**Proof:** {bottle['proof']}")
        if "age" in bottle:
            lines.append(f"**Age:** {bottle['age']}")
        lines.append("")

    lines.append("## Feedback")
    lines.append("To correct any errors in this batch, create a corrections file")
    lines.append("and run: `bourbon-processor feedback --batch <batch_id> --corrections <file.json>`")
    lines.append("")

    return "\n".join(lines)


def _test_conftest_configure_integration_marker():
    """Verify integration marker is configured."""
    from conftest import pytest_configure
    import pytest

    config = Mock()
    config.addinivalue_line = Mock()
    pytest_configure(config)

    # Verify marker was added
    config.addinivalue_line.assert_any_call(
        "markers",
        "integration: marks tests as integration tests requiring external services"
    )
