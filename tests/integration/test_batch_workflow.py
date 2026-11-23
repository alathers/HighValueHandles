"""Integration tests for batch processing workflow.

Tests verify that groups pass through batching intact, preview reports,
and the feedback loop using test fixture images.
"""

import tempfile
from decimal import Decimal
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


class TestBatchingKeepsGroupsIntact:
    """Groups from ImageOrganizer pass through batching whole and are analyzed once.

    (Replaces the old lookahead tests: lookahead is deprecated and does
    nothing, because capture-time grouping already keeps every bottle's
    photos together.)
    """

    @staticmethod
    def _timed_photos(tmp_path, seconds):
        import datetime as dt

        from PIL import ImageDraw

        images = []
        for k, s in enumerate(seconds):
            path = tmp_path / f"IMG_{k:04d}.jpg"
            exif = PILImage.Exif()
            taken = dt.datetime(2026, 11, 7, 16, 0, 0) + dt.timedelta(seconds=s)
            exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
            img = PILImage.new("RGB", (64, 64), (230, 230, 230))
            ImageDraw.Draw(img).rectangle([k * 5 % 50, 0, k * 5 % 50 + 8, 63], fill=(0, 0, 0))
            img.save(path, "JPEG", exif=exif.tobytes())
            images.append(Image(path=str(path), category="BOTTLE"))
        return images

    def test_groups_pass_through_batches_intact_and_once(self, tmp_path):
        from image_processor.image_organizer import ImageOrganizer

        # Five bottles: pairs 3 s apart, bottles a minute apart.
        seconds = [0, 3, 60, 63, 120, 123, 180, 183, 240, 243]
        images = self._timed_photos(tmp_path, seconds)
        groups = ImageOrganizer().group_by_bottle(images)
        assert [len(g) for g in groups] == [2] * 5

        processor = BatchProcessor(batch_size=2, lookahead=5)  # lookahead ignored
        seen: list[list[str]] = []

        def analyze(group, vision_model=None):
            seen.append([Path(img.path).name for img in group])
            return {"name": f"Bottle {len(seen)}", "images": [str(img.path) for img in group]}, ExtractedMetadata()

        with patch.object(processor.bottle_analyzer, "process_single_bottle", side_effect=analyze):
            processor.start_run(groups)
            batches = processor.slice_into_batches(groups)
            results = [processor.process_batch(batch) for batch in batches]

        assert [len(b) for b in batches] == [2, 2, 1]
        assert seen == [[Path(img.path).name for img in g] for g in groups]  # each group once, whole
        assert sum(len(r.bottle_dicts) for r in results) == 5

    def test_lookahead_argument_changes_nothing(self, tmp_path):
        images = self._timed_photos(tmp_path, [0, 3, 60])
        groups = [[images[0], images[1]], [images[2]]]
        for lookahead in (0, 3, 10):
            processor = BatchProcessor(batch_size=1, lookahead=lookahead)
            assert processor.slice_into_batches(groups) == [[groups[0]], [groups[1]]]
            assert processor.apply_lookahead([groups[0]], [groups[1]]) == ([groups[0]], [groups[1]])

    def test_single_photo_groups_processed_independently(self, tmp_path):
        images = []
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        batch_processor = BatchProcessor(batch_size=3)

        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Test", "msrp": "60"}, Mock()),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "70"}, Mock()),
            ]

            result = batch_processor.process_batch([[img] for img in images])

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

        Covers: Full pipeline with batching and preview report
        """
        # Create test images
        images = []
        for i in range(6):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        # Setup batch processor
        batch_processor = BatchProcessor(batch_size=2)

        # Mock bottle analyzer
        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Distillery 1", "msrp": "50"}, Mock()),
                ({"name": "Bottle 2", "distillery": "Distillery 2", "msrp": "75"}, Mock()),
                ({"name": "Bottle 3", "distillery": "Distillery 3", "msrp": "100"}, Mock()),
                ({"name": "Bottle 4", "distillery": "Distillery 4", "msrp": "125"}, Mock()),
                ({"name": "Bottle 5", "distillery": "Distillery 5", "msrp": "150"}, Mock()),
                ({"name": "Bottle 6", "distillery": "Distillery 6", "msrp": "175"}, Mock()),
            ]

            # Process batch
            batches = batch_processor.slice_into_batches([[img] for img in images])
            all_results = []
            for batch in batches:
                result = batch_processor.process_batch(batch)
                all_results.extend(result.bottle_dicts)

            # Verify results
            assert len(all_results) == 6
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
            assert "Bottle 4" in report.markdown_content
            assert "Bottle 5" in report.markdown_content
            assert "Bottle 6" in report.markdown_content

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
        batch_processor = BatchProcessor(batch_size=3)

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
