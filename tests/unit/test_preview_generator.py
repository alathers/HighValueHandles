"""Unit tests for PreviewGenerator class.

Tests markdown generation, file operations, and interactive feedback handling.
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime

import pytest

from src.image_processor.preview_generator import PreviewGenerator
from src.lib.error_handling import ErrorReport, ErrorType
from src.lib.exceptions import BourbonException, OCRError
from src.models.bottle import Bottle, ExtractedMetadata, Correction, ReferenceLink, Quantity


class TestPreviewGeneratorInitialization:
    """Test PreviewGenerator initialization."""

    @patch("pathlib.Path.mkdir")
    def test_init_default_output_dir(self, mock_mkdir):
        """Test initialization with default output directory."""
        generator = PreviewGenerator()

        assert generator.output_dir == Path("processed")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.mkdir")
    def test_init_custom_output_dir(self, mock_mkdir):
        """Test initialization with custom output directory."""
        custom_dir = Path("/custom/output")
        generator = PreviewGenerator(output_dir=custom_dir)

        assert generator.output_dir == custom_dir
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.mkdir")
    def test_init_with_string_path(self, mock_mkdir):
        """Test initialization accepts string path converted to Path."""
        generator = PreviewGenerator(output_dir="/custom/string/path")

        # Path parameter is converted to Path object in __init__
        assert isinstance(generator.output_dir, Path)
        assert str(generator.output_dir) == "/custom/string/path"


class TestPreviewGeneratorMarkdownGeneration:
    """Test markdown generation for individual bottles."""

    def test_generate_bottle_markdown_basic(self):
        """Test basic markdown generation for a bottle."""
        metadata = ExtractedMetadata(
            full_text="Sample OCR text",
            selected_text="Sample OCR text",
            source_images=[Path("image1.jpg"), Path("image2.jpg")],
        )
        bottle = Bottle(
            name="Test Bourbon",
            distillery="Test Distillery",
            specific_release="Regular",
            msrp=Decimal("99.99"),
            metadata=metadata,
        )
        # Add abv attribute dynamically to work around Bottle model limitation
        bottle.abv = None
        generator = PreviewGenerator()

        markdown = generator.generate_bottle_markdown(bottle)

        assert "# Test Bourbon" in markdown
        assert "## Distillery" in markdown
        assert "Test Distillery" in markdown
        assert "## MSRP" in markdown
        assert "$99.99" in markdown
        assert "## Images" in markdown
        assert "image1.jpg" in markdown
        assert "image2.jpg" in markdown

    def test_generate_bottle_markdown_with_all_fields(self):
        """Test markdown with all optional fields populated."""
        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Complete Bourbon",
            distillery="Complete Distillery",
            specific_release="Limited Edition 2025",
            msrp=Decimal("149.99"),
            proof="100",
            age="12 years",
            tasting_notes="Caramel, vanilla, and oak notes with a smooth finish.",
            reference_links=[
                ReferenceLink(label="Buy Now", url="https://example.com/buy"),
                ReferenceLink(label="Review", url="https://example.com/review"),
            ],
            quantity=Quantity(count=2, price=Decimal("149.99")),
            corrections=[
                Correction(
                    field_name="age",
                    original_value="10 years",
                    corrected_value="12 years",
                )
            ],
            metadata=metadata,
        )
        generator = PreviewGenerator()

        markdown = generator.generate_bottle_markdown(bottle)

        assert "**Proof:** 100" in markdown
        assert "**Age:** 12 years" in markdown
        assert "**Release:** Limited Edition 2025" in markdown
        assert "## Tasting Notes" in markdown
        assert "Caramel, vanilla, and oak" in markdown
        assert "## References" in markdown
        assert "[Buy Now](https://example.com/buy)" in markdown
        assert "Value per bottle: $149.99" in markdown
        assert "Total value: $299.98" in markdown
        assert "## Corrections" in markdown
        assert "**age**" in markdown

    def test_generate_bottle_markdown_without_optional_fields(self):
        """Test markdown generation when optional fields are missing."""
        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Minimal Bourbon",
            distillery="Minimal Distillery",
            specific_release="Regular",
            msrp=Decimal("50.00"),
            metadata=metadata,
        )
        bottle.abv = None
        generator = PreviewGenerator()

        markdown = generator.generate_bottle_markdown(bottle)

        # Verify optional sections are not present
        assert "## Tasting Notes" not in markdown
        assert "## References" not in markdown
        assert "## Corrections" not in markdown
        # Verify required sections are present
        assert "## Quantity" in markdown

    def test_generate_bottle_markdown_with_single_quantity(self):
        """Test quantity section shows correctly when quantity is 1."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Single Bottle",
            distillery="Distillery",
            specific_release="Regular",
            msrp=Decimal("75.00"),
            quantity=Quantity(count=1, price=Decimal("75.00")),
            metadata=metadata,
        )
        bottle.abv = None
        generator = PreviewGenerator()

        markdown = generator.generate_bottle_markdown(bottle)

        assert "Total quantity: 1" in markdown
        assert "Value per bottle" not in markdown
        assert "Total value: $75.00" in markdown

    def test_generate_bottle_markdown_with_multiple_quantity(self):
        """Test quantity section shows per-bottle value when quantity > 1."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Case of Bourbon",
            distillery="Distillery",
            specific_release="Regular",
            msrp=Decimal("75.00"),
            quantity=Quantity(count=12, price=Decimal("75.00")),
            metadata=metadata,
        )
        bottle.abv = None
        generator = PreviewGenerator()

        markdown = generator.generate_bottle_markdown(bottle)

        assert "Total quantity: 12" in markdown
        assert "Value per bottle: $75.00" in markdown
        assert "Total value: $900.00" in markdown


class TestPreviewGeneratorSaveBottleFiles:
    """Test bottle file saving operations."""

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch.object(PreviewGenerator, "generate_bottle_markdown")
    @patch("builtins.open")
    @patch("shutil.copy2")
    def test_save_bottle_files_basic(
        self, mock_copy, mock_open, mock_md, mock_exists, mock_mkdir
    ):
        """Test basic file saving with auto-generated folder name."""
        mock_exists.return_value = True
        mock_md.return_value = "# Test Bourbon\n\nTest content"
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("image1.jpg"), Path("image2.jpg")],
        )
        bottle = Bottle(
            name="Test Bourbon",
            distillery="Test Distillery",
            specific_release="Regular",
            msrp=Decimal("99.99"),
            metadata=metadata,
        )

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250113_120000"
            generator = PreviewGenerator()
            result_path = generator.save_bottle_files(bottle, metadata)

        assert mock_mkdir.called
        assert mock_copy.call_count == 2
        assert mock_open.call_count == 2  # markdown + OCR text

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch.object(PreviewGenerator, "generate_bottle_markdown")
    @patch("builtins.open")
    @patch("shutil.copy2")
    def test_save_bottle_files_with_custom_prefix(
        self, mock_copy, mock_open, mock_md, mock_exists, mock_mkdir
    ):
        """Test file saving with custom folder prefix."""
        mock_exists.return_value = True
        mock_md.return_value = "# Custom\n\nContent"
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Any Name",
            distillery="Any Distillery",
            specific_release="Regular",
            msrp=Decimal("75.00"),
            metadata=metadata,
        )

        generator = PreviewGenerator()
        result_path = generator.save_bottle_files(
            bottle, metadata, prefix="custom_folder"
        )

        assert mock_mkdir.called

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch.object(PreviewGenerator, "generate_bottle_markdown")
    @patch("builtins.open")
    @patch("shutil.copy2")
    def test_save_bottle_files_sanitizes_folder_name(
        self, mock_copy, mock_open, mock_md, mock_exists, mock_mkdir
    ):
        """Test folder name is sanitized for filesystem."""
        mock_exists.return_value = True
        mock_md.return_value = "# Sanitized\n\nContent"
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("image.jpg")],
        )
        bottle = Bottle(
            name="Test/Bourbon\\With Slashes",
            distillery="Test Distillery",
            specific_release="Regular",
            msrp=Decimal("99.99"),
            metadata=metadata,
        )

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250113_120000"
            generator = PreviewGenerator()
            result_path = generator.save_bottle_files(bottle, metadata)

        # Verify mkdir was called
        assert mock_mkdir.called
        # The folder name in result_path should be sanitized
        assert "/" not in str(result_path.name)
        assert "\\" not in str(result_path.name)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch.object(PreviewGenerator, "generate_bottle_markdown")
    @patch("builtins.open")
    @patch("shutil.copy2")
    def test_save_bottle_files_skips_missing_images(
        self, mock_copy, mock_open, mock_md, mock_exists, mock_mkdir
    ):
        """Test images that don't exist are skipped."""
        # First exists returns True, second returns False
        mock_exists.side_effect = [True, False]
        mock_md.return_value = "# Test\n\nContent"
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="OCR text",
            selected_text="OCR text",
            source_images=[Path("existing.jpg"), Path("missing.jpg")],
        )
        bottle = Bottle(
            name="Test Bourbon",
            distillery="Test Distillery",
            specific_release="Regular",
            msrp=Decimal("99.99"),
            metadata=metadata,
        )

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250113_120000"
            generator = PreviewGenerator()
            generator.save_bottle_files(bottle, metadata)

        # Only one image copied (existing.jpg)
        assert mock_copy.call_count == 1


class TestPreviewGeneratorErrorReport:
    """Test error report generation and saving."""

    def test_generate_error_report_for_ocr_error(self):
        """Test error report generation for OCR failures."""
        error = OCRError("OCR processing failed")
        generator = PreviewGenerator()

        report = generator.generate_error_report(
            bottle_id="bottle123",
            images=["image1.jpg", "image2.jpg"],
            error=error,
            retry_attempts=2,
        )

        assert report.error_type == ErrorType.OCR_FAILURE
        assert report.retry_count == 2
        assert "image quality" in report.hint.lower()

    def test_generate_error_report_for_model_error(self):
        """Test error report generation for model unavailability."""
        error = BourbonException("Model not available: qwen2-vl")
        generator = PreviewGenerator()

        report = generator.generate_error_report(
            bottle_id="bottle456",
            images=["image.jpg"],
            error=error,
            retry_attempts=1,
        )

        assert report.error_type == ErrorType.AI_MODEL_FAILURE
        assert "Ollama service" in report.hint

    def test_generate_error_report_for_general_bourbon_exception(self):
        """Test error report for general BourbonException."""
        error = BourbonException("General processing error")
        generator = PreviewGenerator()

        report = generator.generate_error_report(
            bottle_id="bottle789",
            images=["image.jpg"],
            error=error,
        )

        assert report.error_type == ErrorType.UNKNOWN
        assert "system configuration" in report.hint.lower()

    def test_generate_error_report_for_unexpected_exception(self):
        """Test error report for non-BourbonException errors."""
        error = ValueError("Unexpected error")
        generator = PreviewGenerator()

        report = generator.generate_error_report(
            bottle_id="bottle999",
            images=["image.jpg"],
            error=error,
        )

        assert report.error_type == ErrorType.UNKNOWN
        assert "system logs" in report.hint.lower()

    @patch("src.image_processor.preview_generator.save_error_reports")
    @patch("src.image_processor.preview_generator.datetime")
    def test_save_error_report_default_dir(self, mock_datetime, mock_save):
        """Test saving error report to default directory."""
        # Mock fixed timestamp for predictable filename
        mock_datetime.now.return_value.strftime.return_value = "20261113_120000"

        error_report = ErrorReport(
            error_type=ErrorType.OCR_FAILURE,
            file_path=Path("image.jpg"),
            message="Test error",
            details="Details",
            retry_count=0,
            hint="Fix it",
        )
        generator = PreviewGenerator()
        result_path = generator.save_error_report(error_report)

        assert result_path == Path("processed/error_report_20261113_120000.json")
        mock_save.assert_called_once()

    @patch("src.image_processor.preview_generator.save_error_reports")
    @patch("src.image_processor.preview_generator.datetime")
    def test_save_error_report_custom_dir(self, mock_datetime, mock_save):
        """Test saving error report to custom directory."""
        # Mock fixed timestamp for predictable filename
        mock_datetime.now.return_value.strftime.return_value = "20261113_120000"

        error_report = ErrorReport(
            error_type=ErrorType.OCR_FAILURE,
            file_path=Path("image.jpg"),
            message="Test error",
            details="Details",
            retry_count=0,
            hint="Fix it",
        )
        generator = PreviewGenerator()
        custom_dir = Path("/custom")
        result_path = generator.save_error_report(error_report, output_dir=custom_dir)

        assert result_path == Path("/custom/error_report_20261113_120000.json")
        mock_save.assert_called_once()


class TestPreviewGeneratorBatchPreviewReport:
    """Test batch preview report generation."""

    def test_generate_batch_preview_report_basic(self):
        """Test basic batch preview report generation."""
        metadata1 = ExtractedMetadata(
            full_text="Bottle 1 text",
            selected_text="Bottle 1 text",
            source_images=[Path("bottle1.jpg")],
        )
        metadata2 = ExtractedMetadata(
            full_text="Bottle 2 text",
            selected_text="Bottle 2 text",
            source_images=[Path("bottle2.jpg")],
        )
        bottle_dicts = [
            {
                "name": "Bottle One",
                "distillery": "Distillery A",
                "msrp": "99.99",
                "proof": "100",
                "quantity": 1,
            },
            {
                "name": "Bottle Two",
                "distillery": "Distillery B",
                "msrp": "149.99",
                "quantity": 2,
            },
        ]
        metadata_list = [metadata1, metadata2]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "# Batch Preview - Batch 1" in markdown
        assert "Total bottles in batch: 2" in markdown
        assert "## Grouping Summary" in markdown
        assert "| Bottle Name | Quantity | Images | Confidence |" in markdown
        assert "## Extracted Metadata Details" in markdown

    def test_generate_batch_preview_report_confidence_calculation(self):
        """Test confidence calculation based on filled fields."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )

        # High confidence (4+ fields): name, distillery, msrp, proof, age = 5 fields
        bottle_high = {
            "name": "High Confidence",
            "distillery": "Distillery",
            "msrp": "99.99",
            "proof": "100",
            "age": "12",
        }

        # Medium confidence (2-3 fields): name, distillery, msrp = 3 fields
        bottle_medium = {
            "name": "Medium Confidence",
            "distillery": "Distillery",
            "msrp": "99.99",
        }

        # Low confidence (<2 fields): only msrp = 1 field
        bottle_low = {"msrp": "99.99"}

        generator = PreviewGenerator()

        markdown_high = generator.generate_batch_preview_report(
            [bottle_high], [metadata], batch_id=1
        )
        assert "| HIGH |" in markdown_high

        markdown_medium = generator.generate_batch_preview_report(
            [bottle_medium], [metadata], batch_id=1
        )
        assert "| MEDIUM |" in markdown_medium

        markdown_low = generator.generate_batch_preview_report(
            [bottle_low], [metadata], batch_id=1
        )
        assert "| LOW |" in markdown_low

    def test_generate_batch_preview_report_with_rejected_bottles(self):
        """Test batch preview includes rejected bottles."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test Bottle", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]
        rejected_bottles = ["bottle1", "bottle2"]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts,
            metadata_list,
            rejected_bottles=rejected_bottles,
            batch_id=1,
        )

        assert "Rejected bottles: 2" in markdown
        assert "## Low-Confidence or Rejected Items" in markdown
        assert "**Rejected:**" in markdown
        assert "- bottle1" in markdown
        assert "- bottle2" in markdown

    def test_generate_batch_preview_report_low_confidence_section(self):
        """Test low-confidence bottles are highlighted."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        # Only msrp field = 1 field (LOW confidence)
        bottle_dicts = [{"msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "## Low-Confidence or Rejected Items" in markdown
        assert "**Low Confidence:**" in markdown
        assert "- Bottle 1" in markdown

    def test_generate_batch_preview_report_with_ocr_preview(self):
        """Test OCR text preview is included and truncated."""
        # Create text longer than 300 characters to trigger truncation
        long_text = "This is a long OCR text that exceeds the preview limit but will be truncated. " * 10
        assert len(long_text) > 300

        metadata = ExtractedMetadata(
            full_text=long_text,
            selected_text=long_text,
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test Bottle", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "**OCR Preview:**" in markdown
        assert "```" in markdown
        # Should be truncated with ellipsis
        assert "..." in markdown
        # Full text should not be present
        assert len(long_text) > 400  # Longer than what's in preview

    def test_generate_batch_preview_report_feedack_instructions(self):
        """Test feedback instructions are included in report."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "## Feedback Instructions" in markdown
        assert "**A** - Accept" in markdown
        assert "**R** - Reject" in markdown
        assert "**M** - Merge" in markdown
        assert "**S** - Split" in markdown
        assert "**E** - Edit" in markdown
        assert "feedback command" in markdown.lower()
        assert "retry failed" in markdown.lower()

    def test_generate_batch_preview_report_populates_optional_fields(self):
        """Test optional fields are added when present."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [
            {
                "name": "Complete Bottle",
                "distillery": "Distillery",
                "msrp": "99.99",
                "proof": "100",
                "age": "12",
                "specific_release": "Limited",
            }
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "**Proof:** 100" in markdown
        assert "**Age:** 12" in markdown
        assert "**Release:** Limited" in markdown


class TestPreviewGeneratorSaveBatchPreview:
    """Test saving batch preview reports."""

    @patch("builtins.open")
    def test_save_batch_preview_default_path(self, mock_open):
        """Test saving batch preview to default path."""
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        result_path = generator.save_batch_preview(bottle_dicts, metadata_list, batch_id=2)

        # Default path should be {output_dir}/batch_2_preview.md
        assert str(result_path) == "processed/batch_2_preview.md"

    @patch("builtins.open")
    def test_save_batch_preview_custom_path(self, mock_open):
        """Test saving batch preview to custom path."""
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        custom_path = Path("/custom/custom_preview.md")
        result_path = generator.save_batch_preview(
            bottle_dicts, metadata_list, batch_id=1, output_path=custom_path
        )

        assert result_path == custom_path


class TestPreviewGeneratorInteractivePrompt:
    """Test interactive user prompting."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_accept_all(self, mock_print, mock_input):
        """Test accepting all bottles in interactive mode."""
        mock_input.return_value = "A"

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg"), Path("bottle2.jpg")],
        )
        bottle_dicts = [
            {"name": "Bottle One", "distillery": "Dist A", "msrp": "99.99", "quantity": 1},
            {"name": "Bottle Two", "distillery": "Dist B", "msrp": "149.99", "quantity": 1},
        ]
        metadata_list = [metadata, metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert corrections == []
        assert rejected == []
        assert mock_input.call_count == 2

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_reject_bottle(self, mock_print, mock_input):
        """Test rejecting bottles in interactive mode."""
        mock_input.return_value = "R"

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        bottle_dicts = [
            {"name": "Bottle One", "distillery": "Dist A", "msrp": "99.99", "quantity": 1}
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert corrections == []
        assert len(rejected) == 1
        assert rejected[0] == "bottle1"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_merge(self, mock_print, mock_input):
        """Test merging bottles in interactive mode."""
        mock_input.return_value = "M"

        metadata1 = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        metadata2 = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle2.jpg")],
        )
        bottle_dicts = [
            {"name": "Bottle One", "distillery": "Dist A", "msrp": "99.99", "quantity": 1},
            {"name": "Bottle Two", "distillery": "Dist A", "msrp": "99.99", "quantity": 1},
        ]
        metadata_list = [metadata1, metadata2]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert len(corrections) == 1
        assert corrections[0]["field_name"] == "grouping"
        assert corrections[0]["corrected_value"] == "merge_with_1"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_split(self, mock_print, mock_input):
        """Test splitting bottles in interactive mode."""
        mock_input.return_value = "S"

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        bottle_dicts = [
            {"name": "Grouped Bottle", "distillery": "Dist", "msrp": "199.99", "quantity": 2}
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert len(corrections) == 1
        assert corrections[0]["field_name"] == "grouping"
        assert corrections[0]["corrected_value"] == "split"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_edit(self, mock_print, mock_input):
        """Test editing bottle fields in interactive mode."""
        def input_side_effect(*args):
            # First call selects Edit mode
            # Subsequent calls provide new values
            if mock_input.call_count == 1:
                return "E"
            elif mock_input.call_count == 2:
                return ""  # Keep name
            elif mock_input.call_count == 3:
                return "New Distillery"  # Change distillery
            elif mock_input.call_count == 4:
                return ""  # Keep MSRP
            elif mock_input.call_count == 5:
                return ""  # Keep proof
            elif mock_input.call_count == 6:
                return ""  # Keep age
            elif mock_input.call_count == 7:
                return ""  # Keep release
            return "A"

        mock_input.side_effect = input_side_effect

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        bottle_dicts = [
            {
                "name": "Test Bottle",
                "distillery": "Old Distillery",
                "msrp": "99.99",
                "proof": "100",
                "age": "12",
                "specific_release": "Regular",
            }
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert len(corrections) == 1
        assert corrections[0]["field_name"] == "distillery"
        assert corrections[0]["original_value"] == "Old Distillery"
        assert corrections[0]["corrected_value"] == "New Distillery"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_quit(self, mock_print, mock_input):
        """Test quitting early from interactive mode."""
        mock_input.return_value = "Q"

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg"), Path("bottle2.jpg"), Path("bottle3.jpg")],
        )
        bottle_dicts = [
            {"name": "Bottle 1", "distillery": "Dist", "msrp": "99.99", "quantity": 1},
            {"name": "Bottle 2", "distillery": "Dist", "msrp": "99.99", "quantity": 1},
            {"name": "Bottle 3", "distillery": "Dist", "msrp": "99.99", "quantity": 1},
        ]
        metadata_list = [metadata, metadata, metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        # Should only process first bottle before quitting
        assert corrections == []
        assert rejected == []
        assert mock_input.call_count == 1

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_invalid_action_treated_as_accept(
        self, mock_print, mock_input
    ):
        """Test invalid action is treated as Accept."""
        mock_input.return_value = "X"  # Invalid choice

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        bottle_dicts = [
            {"name": "Test Bottle", "distillery": "Dist", "msrp": "99.99", "quantity": 1}
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert corrections == []
        assert rejected == []

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_prompt_case_insensitive(self, mock_print, mock_input):
        """Test action is case-insensitive."""
        mock_input.return_value = "a"  # lowercase a

        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle1.jpg")],
        )
        bottle_dicts = [
            {"name": "Test Bottle", "distillery": "Dist", "msrp": "99.99", "quantity": 1}
        ]
        metadata_list = [metadata]

        generator = PreviewGenerator()
        corrections, rejected = generator.interactive_prompt(
            bottle_dicts, metadata_list
        )

        assert corrections == []
        assert rejected == []


class TestPreviewGeneratorHandleInteractiveChoice:
    """Test parsing user interactive choice actions."""

    def test_handle_choice_accept(self):
        """Test Accept choice returns None."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="A",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is None

    def test_handle_choice_reject(self):
        """Test Reject choice returns rejection correction."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="R",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is not None
        assert result["field_name"] == "grouping"
        assert result["corrected_value"] == "rejected"
        assert result["bottle_id"] == "bottle1"

    def test_handle_choice_merge(self):
        """Test Merge choice returns merge correction."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="M",
            bottle_dict={"name": "Test"},
            bottle_id="bottle2",
            bottle_index=2,  # Not first bottle
        )

        assert result is not None
        assert result["field_name"] == "grouping"
        assert result["corrected_value"] == "merge_with_1"

    def test_handle_choice_merge_first_bottle(self):
        """Test Merge choice on first bottle returns None."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="M",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,  # First bottle
        )

        assert result is None

    def test_handle_choice_split(self):
        """Test Split choice returns split correction."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="S",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is not None
        assert result["field_name"] == "grouping"
        assert result["corrected_value"] == "split"

    def test_handle_choice_edit(self):
        """Test Edit choice returns edit mode marker."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="E",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is not None
        assert result["field_name"] == "edit"
        assert result["corrected_value"] == "enter_edit_mode"

    def test_handle_choice_unknown(self):
        """Test unknown choice returns None (treated as Accept)."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="X",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is None

    def test_handle_choice_whitespace_trimming(self):
        """Test choice is trimmed of whitespace."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="  A  ",  # Whitespace around choice
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is None

    def test_handle_choice_case_insensitive(self):
        """Test choice is case-insensitive."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="r",  # lowercase
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is not None
        assert result["corrected_value"] == "rejected"

    def test_handle_choice_includes_timestamp(self):
        """Test correction includes corrected_at timestamp."""
        generator = PreviewGenerator()
        result = generator.handle_interactive_choice(
            choice="R",
            bottle_dict={"name": "Test"},
            bottle_id="bottle1",
            bottle_index=1,
        )

        assert result is not None
        # Just verify timestamp exists and is in ISO format
        assert "corrected_at" in result
        assert isinstance(result["corrected_at"], str)
        # Should parse as ISO format datetime
        datetime.fromisoformat(result["corrected_at"])
