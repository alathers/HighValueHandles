"""Output format contract tests for bourbon image processor.

Tests ensure output file formats match the specification, validating:
- Bottle markdown file structure and content
- JSON metadata file schema and required fields
- Failure report JSON schema (what ``retry --report`` reads)
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from image_processor.preview_generator import OUTPUT_SCHEMA_VERSION, PreviewGenerator
from lib.exceptions import OCRError
from models.bottle import (
    Bottle,
    ExtractedMetadata,
)


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

    def test_bottle_markdown_distillery_row(self):
        """The distillery is a row of the Facts table."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert "| Distillery | Test Distillery |" in markdown

    def test_bottle_markdown_msrp_line(self):
        """MSRP is shown in the Prices section with a dollar sign."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )
        markdown = bottle.to_markdown()

        assert "- **MSRP:** $49.99" in markdown

    def test_bottle_markdown_optional_rows(self):
        """Proof, age, release and tasting notes are Facts rows when present."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Single Barrel",
            msrp=Decimal("49.99"),
            proof="93 Proof",
            age="12 Years",
            tasting_notes="Caramel, vanilla, and oak",
        )
        markdown = bottle.to_markdown()

        assert "| Proof | 93 Proof |" in markdown
        assert "| Age | 12 Years |" in markdown
        assert "| Release | Single Barrel |" in markdown
        assert "| Tasting notes | Caramel, vanilla, and oak |" in markdown

    def test_bottle_markdown_complete_structure(self):
        """Sections appear in the documented order; an unchecked bottle is never publish-ready."""
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

        assert markdown.startswith("# Test Bottlename\n\n**Status:** NOT publish-ready: not fact-checked")
        order = [
            "## Needs Review", "## Facts", "## Prices", "## Description (draft)", "## Sources",
            "## Verification", "## References", "## Web Research Log", "## Quantity", "## Images",
            "## Processing Details",
        ]
        positions = [markdown.index(h) for h in order]
        assert positions == sorted(positions)
        assert "Rich amber color" in markdown
        assert "Web research was not run for this bottle." in markdown

    def test_to_markdown_matches_generate_bottle_markdown(self, tmp_path):
        bottle = Bottle(name="T", distillery="D", specific_release="", msrp=None)
        assert bottle.to_markdown() == PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)


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


class TestBottleJSONSchemaV2:
    """The saved bottle JSON (``save_bottle_files``) has a fixed v2 key set."""

    KEYS = {
        "schema_version", "bottle_id", "name", "distillery", "specific_release", "msrp", "proof", "abv", "age",
        "tasting_notes", "label_price", "quantity", "extracted_at", "run_id", "reference_links", "corrections",
        "facts", "web_research", "vendor_image", "review_required", "metadata", "batch_number", "model_used",
        "confidence", "source_images", "images", "missing_images", "total_value",
    }

    def test_key_set_and_version(self, tmp_path):
        from image_processor.fact_checker import FactChecker
        from models.facts import FactInput

        image = tmp_path / "IMG_1.jpg"
        image.write_bytes(b"x")
        sheet = FactChecker.deterministic(
            FactInput("IMG_1", {"name": "Test"}, "high", ["IMG_1.jpg"], {}), "no text model"
        )
        bottle = Bottle.from_analysis(
            {"name": "Test", "images": [str(image)], "facts": sheet.to_dict(), "run_id": "r1",
             "web_research": {"status": "skipped", "reason": "web lookups disabled"}},
            ExtractedMetadata(source_images=[str(image)]),
        )
        folder = PreviewGenerator(tmp_path / "out").save_bottle_files(bottle)
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert set(data) == self.KEYS
        assert data["schema_version"] == OUTPUT_SCHEMA_VERSION == 2
        assert data["reference_links"] == []
        assert data["corrections"] == []
        assert data["facts"]["publish"]["ready"] is False
        assert data["web_research"]["status"] == "skipped"
        assert data["run_id"] == "r1"

    def test_batch_json_has_schema_version(self, tmp_path):
        PreviewGenerator(tmp_path).save_batch_preview([], [], batch_id="b1")
        assert json.loads((tmp_path / "batch_b1.json").read_text())["schema_version"] == 2


class TestFailureReportJSONSchema:
    """Per-bottle failure report written by ``PreviewGenerator.save_failure_report``.

    Contract: this is the file ``retry --report`` reads.
    """

    KEYS = {
        "schema_version", "bottle_id", "images", "error_message", "error_type", "stage",
        "hint", "details", "timestamp", "retry_count", "suggested_fix",
    }

    def _write(self, tmp_path, error, **kwargs):
        image = tmp_path / "IMG_1.jpg"
        image.write_bytes(b"x")
        path = PreviewGenerator(tmp_path / "out").save_failure_report(
            "IMG_1", [str(image), "relative.jpg"], error, **kwargs
        )
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_exact_key_set(self, tmp_path):
        path, data = self._write(tmp_path, OCRError("no text"), stage="analysis")
        assert set(data) == self.KEYS
        assert path.name == "error_report_img_1.json"

    def test_values_for_an_exception(self, tmp_path):
        try:
            raise OCRError("no text")
        except OCRError as error:
            _, data = self._write(tmp_path, error, stage="analysis", retry_count=2)

        assert data["schema_version"] == OUTPUT_SCHEMA_VERSION == 2
        assert data["bottle_id"] == "IMG_1"
        assert all(Path(p).is_absolute() for p in data["images"])
        assert data["error_message"].startswith("OCRError: no text")
        assert data["error_type"] == "OCR_FAILURE"
        assert data["stage"] == "analysis"
        assert data["hint"]
        assert "Traceback" in data["details"]
        assert data["retry_count"] == 2
        assert data["suggested_fix"] == {}
        assert datetime.fromisoformat(data["timestamp"]).tzinfo is not None

    def test_values_for_a_plain_message(self, tmp_path):
        _, data = self._write(tmp_path, "Rejected during interactive review", stage="rejected")

        assert data["error_message"] == "Rejected during interactive review"
        assert data["error_type"] == "UNKNOWN"
        assert data["hint"] is None
        assert data["details"] is None

    def test_existing_report_is_never_overwritten(self, tmp_path):
        first, _ = self._write(tmp_path, "one", stage="save")
        second, _ = self._write(tmp_path, "two", stage="save")

        assert first != second
        assert json.loads(first.read_text())["error_message"] == "one"


class TestOutputFilePersistence:
    """Test output files can be written and read correctly."""

    def test_markdown_file_write_and_read(self, tmp_path):
        """Test markdown file can be written and read back."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )

        output_file = tmp_path / "bottle.md"
        output_file.write_text(bottle.to_markdown())

        content = output_file.read_text()
        assert "# Test Bottle" in content
        assert "| Distillery | Test Distillery |" in content

    def test_json_file_write_and_read(self, tmp_path):
        """Test JSON file can be written and read back."""
        bottle = Bottle(
            name="Test Bottle",
            distillery="Test Distillery",
            specific_release="Test",
            msrp=Decimal("49.99"),
        )

        output_file = tmp_path / "bottle.json"
        output_file.write_text(bottle.to_json())

        data = json.loads(output_file.read_text())
        assert data["name"] == "Test Bottle"
        assert data["distillery"] == "Test Distillery"
        assert data["msrp"] == "49.99"
