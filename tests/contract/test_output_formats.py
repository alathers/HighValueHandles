"""Output format contract tests for bourbon image processor.

Tests ensure output file formats match the specification, validating:
- Bottle markdown file structure and content
- JSON metadata file schema and required fields
- Error report JSON schema
- Batch preview markdown format
"""

import json
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from models.bottle import (
    BatchPreviewReport,
    Bottle,
    ExtractedMetadata,
)
from lib.error_handling import ErrorReport, ErrorType


class TestBottleMarkdownOutput:
    """Test bottle markdown file output format.

    Contract: Output files follow the structure defined in
    examples/processed_sample/bottle_001/bottle_001.md
    """

    def test_bottle_markdown_has_title(self):
        """Test markdown bottle file starts with H1 title.

        Contract: First line is "# {Bottle Name}"
        """
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert markdown.startswith("# Test Bottle")

    def test_bottle_markdown_distillery_section(self):
        """Test distillery is displayed with bold label."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert "**Distillery:**" in markdown
        assert "Test Distillery" in markdown

    def test_bottle_markdown_msrp_section(self):
        """Test MSRP is displayed with dollar sign."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert "**MSRP:**" in markdown
        assert "$49.99" in markdown

    def test_bottle_markdown_proof_section(self):
        """Test proof is displayed when present."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
            proof="93 Proof",
        )
        markdown = bottle.to_markdown()

        assert "**Proof:**" in markdown
        assert "93 Proof" in markdown

    def test_bottle_markdown_age_section(self):
        """Test age is displayed when present."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
            age="12 Years",
        )
        markdown = bottle.to_markdown()

        assert "**Age:**" in markdown
        assert "12 Years" in markdown

    def test_bottle_markdown_release_information_section(self):
        """Test Release Information H2 section exists."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Single Barrel",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert "## Release Information" in markdown
        assert "Single Barrel" in markdown

    def test_bottle_markdown_tasting_notes_section(self):
        """Test Tasting Notes H2 section exists."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
            tasting_notes="Caramel, vanilla, and oak",
        )
        markdown = bottle.to_markdown()

        assert "## Tasting Notes" in markdown
        assert "Caramel, vanilla, and oak" in markdown

    def test_bottle_markdown_complete_structure(self):
        """Test complete markdown structure matches expected format."""
        bottle = Bottle(
            name="Test Bottlename",
            distillery="Test Distillery Name",
            specific_release="Single Barrel Release",
            msrp=Decimal("59.99"),
            proof="100 Proof",
            age="10 Years",
            tasting_notes="Rich amber color with notes of caramel, vanilla, and oak. Long warming finish.",
        )
        markdown = bottle.to_markdown()

        # Verify required sections
        assert markdown.startswith("# Test Bottlename")
        assert "**Distillery:** Test Distillery Name" in markdown
        assert "**MSRP:** $59.99" in markdown
        assert "**Proof:** 100 Proof" in markdown
        assert "**Age:** 10 Years" in markdown
        assert "## Release Information" in markdown
        assert "Single Barrel Release" in markdown
        assert "## Tasting Notes" in markdown
        assert "Rich amber color" in markdown


class TestBottleJSONOutput:
    """Test bottle JSON metadata file output format.

    Contract: JSON follows schema defined in
    examples/processed_sample/bottle_001/bottle_001.json
    """

    def test_bottle_json_has_required_fields(self):
        """Test JSON has all required fields."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test Release",
            msrp=Decimal("49.99"),
            proof="100 Proof",
            age="10 Years",
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        # Required top-level fields
        assert "name" in data
        assert "distillery" in data
        assert "specific_release" in data
        assert "msrp" in data

    def test_bottle_json_optional_fields_null_when_missing(self):
        """Test optional fields are null or omitted when not set."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        # Optional fields should be None/null
        assert data.get("proof") is None
        assert data.get("age") is None
        assert data.get("tasting_notes") is None

    def test_bottle_json_msrp_number_type(self):
        """Test MSRP is numeric (as string for Decimal)."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("59.99"),
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        # MSRP is stored as string representation of Decimal
        assert data["msrp"] == "59.99"

    def test_bottle_json_nested_metadata(self):
        """Test metadata is properly serialized in JSON."""
        metadata = ExtractedMetadata(
            full_text="Test text content",
            selected_text="Selected text",
            source_images=[Path("/path/to/img1.jpg"), Path("/path/to/img2.jpg")],
        )
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Single Barrel",
            msrp=Decimal("59.99"),
            metadata=metadata,
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        assert "metadata" in data
        assert data["metadata"]["full_text"] == "Test text content"
        assert data["metadata"]["selected_text"] == "Selected text"
        assert isinstance(data["metadata"].get("source_images"), list)

    def test_bottle_json_source_images_array(self):
        """Test source_images is an array of strings."""
        metadata = ExtractedMetadata(
            full_text="Test",
            selected_text="Test",
            source_images=[Path("/path/to/img1.jpg"), Path("/path/to/img2.jpg")],
        )
        bottle = Bottle(
            name="Test",
            distillery="Test",
            specific_release="Test",
            msrp=Decimal("49.99"),
            metadata=metadata,
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        assert "metadata" in data
        assert isinstance(data["metadata"].get("source_images"), list)
        assert len(data["metadata"]["source_images"]) == 2

    def test_bottle_json_extracted_at_iso_format(self):
        """Test extracted_at is ISO 8601 timestamp."""
        bottle = Bottle(
            name="Test",
            distillery="Test",
            specific_release="Test",
            msrp=Decimal("49.99"),
            extracted_at=datetime(2026, 11, 13, 10, 30, 0),
        )
        json_str = bottle.to_json()
        data = json.loads(json_str)

        assert "extracted_at" in data
        # Should parse as valid ISO 8601
        datetime.fromisoformat(data["extracted_at"])

    def test_bottle_json_parsing_roundtrip(self):
        """Test JSON can be parsed back to Bottle object."""
        original = Bottle(
            name="Blanton's Single Barrel",
            distillery="Buffalo Trace Distillery",
            specific_release="Original Single Barrel",
            msrp=Decimal("59.99"),
            proof="93 Proof",
            age="No Age Statement",
            tasting_notes="Caramel, vanilla, and oak notes",
        )

        json_str = original.to_json()
        parsed = Bottle.from_json(json_str)

        assert parsed.name == original.name
        assert parsed.distillery == original.distillery
        assert str(parsed.msrp) == str(original.msrp)
        assert parsed.proof == original.proof
        assert parsed.age == original.age


class TestErrorReportJSONSchema:
    """Test error report JSON schema.

    Contract: Error reports follow ErrorReport dataclass structure
    defined in src/lib/error_handling.py
    """

    def test_error_report_required_fields(self):
        """Test error report has required fields."""
        report = ErrorReport(
            error_type=ErrorType.OCR_FAILURE,
            file_path="/path/to/image.jpg",
            message="Failed to extract text",
            details="OCR call returned empty results"
        )
        data = report.to_dict()

        assert "error_type" in data
        assert "file_path" in data
        assert "message" in data
        assert "timestamp" in data

    def test_error_report_optional_fields(self):
        """Test error report optional fields."""
        report = ErrorReport(
            error_type=ErrorType.OCR_FAILURE,
            file_path="/path/to/image.jpg",
            message="Failed to extract text",
            details="Traceback (most recent call last):\n  File \"extract.py\", line 42, in extract_text",
            retry_count=3,
            hint="Try adjusting image contrast",
        )
        data = report.to_dict()

        assert "Traceback" in data["details"]
        assert "extract_text" in data["details"]
        assert data["retry_count"] == 3
        assert data["hint"] == "Try adjusting image contrast"

    def test_error_report_timestamp_iso_format(self):
        """Test error report timestamp is ISO format."""
        now_dt = datetime.now()
        now_iso = now_dt.isoformat()
        report = ErrorReport(
            error_type=ErrorType.UNKNOWN,
            file_path="/path/to/file.jpg",
            message="Test message",
            details="Test details",
            timestamp=now_iso,
        )
        data = report.to_dict()

        # Should parse as valid ISO 8601
        parsed_time = datetime.fromisoformat(data["timestamp"])
        assert parsed_time.year == now_dt.year

    def test_error_report_roundtrip(self):
        """Test error report can serialize and deserialize."""
        original = ErrorReport(
            error_type=ErrorType.NETWORK_ERROR,
            file_path=Path("/tmp/bottle.jpg"),
            message="Connection refused",
            details="Connection refused at http://localhost:11434",
            retry_count=2,
            timestamp=datetime(2026, 11, 13, 10, 30, 0).isoformat(),
        )

        data = original.to_dict()
        parsed = ErrorReport.from_dict(data)

        assert parsed.error_type == original.error_type
        assert parsed.file_path == original.file_path
        assert parsed.message == original.message
        assert parsed.retry_count == original.retry_count


class TestBatchPreviewMarkdown:
    """Test batch preview markdown report format.

    Contract: Batch preview reports follow the markdown structure
    described in the specification.
    """

    def test_batch_preview_report_structure(self):
        """Test batch preview report has proper markdown structure."""
        report = BatchPreviewReport(
            report_id="preview_001",
            batch_id="batch_001",
            generated_timestamp=datetime(2026, 11, 13, 10, 30, 0),
            markdown_content="# Batch Preview\n\nProcessing 3 bottles...",
        )

        # Verify report has markdown content
        assert report.markdown_content.startswith("#")
        assert "Batch Preview" in report.markdown_content

    def test_batch_preview_report_serialization(self):
        """Test batch preview report can be serialized."""
        report = BatchPreviewReport(
            report_id="preview_001",
            batch_id="batch_001",
            generated_timestamp=datetime(2026, 11, 13, 10, 30, 0),
            markdown_content="# Batch Preview\n\nBottle 1: Test\n\nBottle 2: Another",
            report_path=Path("/tmp/preview.md"),
        )

        data = report.to_dict()
        assert data["report_id"] == "preview_001"
        assert data["batch_id"] == "batch_001"
        assert "markdown_content" in data
        assert "generated_timestamp" in data

    def test_batch_preview_report_deserialization(self):
        """Test batch preview report can be deserialized."""
        data = {
            "report_id": "preview_002",
            "batch_id": "batch_002",
            "generated_timestamp": "2026-11-13T10:30:00",
            "markdown_content": "# Preview Report",
            "report_path": "/tmp/preview_002.md",
        }

        report = BatchPreviewReport.from_dict(data)
        assert report.report_id == "preview_002"
        assert report.batch_id == "batch_002"
        assert report.markdown_content == "# Preview Report"
        assert str(report.report_path) == "/tmp/preview_002.md"


class TestOutputFilePersistence:
    """Test output files can be written and read correctly."""

    def test_markdown_file_write_and_read(self):
        """Test markdown file can be written and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bottle = Bottle(
                name="Test Bottle",
                distillery="Test Distillery",
                specific_release="Test",
                msrp=Decimal("49.99"),
            )

            # Write to file
            output_file = Path(tmpdir) / "bottle.md"
            output_file.write_text(bottle.to_markdown())

            # Read and verify
            content = output_file.read_text()
            assert "# Test Bottle" in content
            assert "**Distillery:** Test Distillery" in content

    def test_json_file_write_and_read(self):
        """Test JSON file can be written and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bottle = Bottle(
                name="Test Bottle",
                distillery="Test Distillery",
                specific_release="Test",
                msrp=Decimal("49.99"),
            )

            # Write to file
            output_file = Path(tmpdir) / "bottle.json"
            output_file.write_text(bottle.to_json())

            # Read and verify
            data = json.loads(output_file.read_text())
            assert data["name"] == "Test Bottle"
            assert data["distillery"] == "Test Distillery"
            assert data["msrp"] == "49.99"

    def test_error_report_file_write_and_read(self):
        """Test error report can be written and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ErrorReport(
                error_type=ErrorType.UNKNOWN,
                file_path="/test/path.jpg",
                message="Test error message",
                details="Test error details"
            )

            # Write to file
            output_file = Path(tmpdir) / "error_report.json"
            json.dump(report.to_dict(), output_file.open("w"))

            # Read and verify
            data = json.loads(output_file.read_text())
            assert data["error_type"] == "UNKNOWN"
            assert data["file_path"] == "/test/path.jpg"
            assert data["message"] == "Test error message"
