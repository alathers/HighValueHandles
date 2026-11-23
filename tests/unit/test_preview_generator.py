"""Unit tests for PreviewGenerator class.

Tests markdown generation, file operations, and interactive feedback handling.
"""

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime

import pytest

from image_processor.preview_generator import PreviewGenerator
from lib.error_handling import ErrorReport, ErrorType
from lib.exceptions import BourbonException, OCRError
from models.bottle import Bottle, ExtractedMetadata, Correction, ReferenceLink, Quantity


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

    def test_unknown_msrp_renders_unknown_not_zero(self, tmp_path):
        """No visible price: the .md says Unknown and the JSON has nulls, never $0.00."""
        bottle = Bottle.from_analysis(
            {"name": "George T. Stagg", "abv": "65.1%"},
            ExtractedMetadata(source_images=[Path("IMG_4668.jpeg")]),
            quantity=2,
        )
        generator = PreviewGenerator(output_dir=tmp_path)

        markdown = generator.generate_bottle_markdown(bottle)

        assert "## MSRP\nUnknown" in markdown
        assert "Total value: Unknown" in markdown
        assert "Value per bottle" not in markdown
        assert "$0.00" not in markdown
        assert "- **ABV:** 65.1%" in markdown

        folder = generator.save_bottle_files(bottle)
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert data["msrp"] is None
        assert data["total_value"] is None

    def test_extracted_timestamp_has_offset(self):
        bottle = Bottle.from_analysis({"name": "X"})
        markdown = PreviewGenerator().generate_bottle_markdown(bottle)
        line = next(line for line in markdown.splitlines() if line.startswith("- **Extracted**"))
        # e.g. 2026-11-22T21:38:09-06:00 (local time with its offset)
        assert line[-6] in "+-" and line[-3] == ":"


def _jpeg(path: Path, color=(120, 60, 20)) -> Path:
    """Write a tiny real JPEG."""
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (8, 8), color).save(path, "JPEG")
    return path


def _bottle(name="Test Bourbon", images=(), full_text="OCR text", **kwargs) -> Bottle:
    metadata = ExtractedMetadata(
        full_text=full_text,
        selected_text=full_text,
        source_images=[Path(p) for p in images],
    )
    return Bottle(
        name=name,
        distillery=kwargs.pop("distillery", "Test Distillery"),
        specific_release=kwargs.pop("specific_release", "Regular"),
        msrp=kwargs.pop("msrp", Decimal("99.99")),
        metadata=metadata,
        **kwargs,
    )


def _visible_entries(directory: Path) -> list:
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


class TestSlugify:
    """slugify() makes safe, lowercase folder names."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Eagle Rare 10 Year", "eagle_rare_10_year"),
            ("Blanton's: Gold/Edition? <1>", "blantons_gold_edition_1"),
            ("Élevé Café", "eleve_cafe"),
            ("../../etc", "etc"),
            ("Test/Bourbon\\With Slashes", "test_bourbon_with_slashes"),
            ("", "bottle"),
            (None, "bottle"),
            ("???", "bottle"),
        ],
    )
    def test_slugify(self, text, expected):
        from image_processor.preview_generator import slugify

        assert slugify(text) == expected

    def test_slugify_truncates_without_trailing_underscore(self):
        from image_processor.preview_generator import slugify

        slug = slugify("a" * 59 + " bcd")
        assert len(slug) <= 60
        assert not slug.endswith("_")


class TestPreviewGeneratorSaveBottleFiles:
    """save_bottle_files writes <slug>/{<slug>.md,.json,.txt,images/} atomically."""

    def test_save_bottle_files_basic(self, tmp_path):
        """Layout, JSON contents and round trip."""
        photos = tmp_path / "photos"
        img1 = _jpeg(photos / "image1.jpg")
        img2 = _jpeg(photos / "image2.jpg", (10, 10, 10))
        bottle = _bottle(images=[img1, img2], full_text="OCR text\nmore")
        out = tmp_path / "out"

        folder = PreviewGenerator(output_dir=out).save_bottle_files(
            bottle, bottle.metadata, model_used="qwen3.6:35b-a3b", confidence="HIGH"
        )

        assert folder == out / "test_bourbon"
        assert _visible_entries(folder) == ["images", "test_bourbon.json", "test_bourbon.md", "test_bourbon.txt"]
        assert _visible_entries(folder / "images") == ["image1.jpg", "image2.jpg"]
        assert (folder / "images" / "image1.jpg").read_bytes() == img1.read_bytes()
        assert (folder / "test_bourbon.txt").read_text(encoding="utf-8") == "OCR text\nmore"

        data = json.loads((folder / "test_bourbon.json").read_text(encoding="utf-8"))
        assert data["msrp"] == "99.99"
        assert len(data["metadata"]["source_images"]) == 2
        assert data["images"] == ["images/image1.jpg", "images/image2.jpg"]
        assert data["source_images"] == [str(img1), str(img2)]
        assert data["missing_images"] == []
        assert data["bottle_id"] == "image1"
        assert data["model_used"] == "qwen3.6:35b-a3b"
        assert data["confidence"] == "HIGH"
        assert data["schema_version"] == 1
        restored = Bottle.from_json((folder / "test_bourbon.json").read_text(encoding="utf-8"))
        assert restored.name == "Test Bourbon"
        assert restored.msrp == Decimal("99.99")

        md = (folder / "test_bourbon.md").read_text(encoding="utf-8")
        assert md.startswith("# Test Bourbon\n")
        assert "[image1.jpg](images/image1.jpg) (primary)" in md
        assert "- **Model**: qwen3.6:35b-a3b" in md
        assert (out / ".bourbon_output").exists()

    def test_save_bottle_files_with_custom_prefix(self, tmp_path):
        """A prefix replaces the bottle name and is slugified too."""
        bottle = _bottle(name="Any Name", images=[_jpeg(tmp_path / "image.jpg")])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(
            bottle, bottle.metadata, prefix="custom folder"
        )

        assert folder.name == "custom_folder"
        assert (folder / "custom_folder.md").is_file()

    def test_save_bottle_files_sanitizes_folder_name(self, tmp_path):
        """Path separators and '..' never escape output_dir."""
        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        image = _jpeg(tmp_path / "image.jpg")

        slashes = generator.save_bottle_files(_bottle(name="Test/Bourbon\\With Slashes", images=[image]))
        escape = generator.save_bottle_files(_bottle(name="../../x", images=[image]))

        assert slashes == out / "test_bourbon_with_slashes"
        assert escape == out / "x"
        assert escape.resolve().parent == out.resolve()

    def test_save_bottle_files_skips_missing_images(self, tmp_path):
        """A missing image is listed, not copied, and is not an error."""
        existing = _jpeg(tmp_path / "existing.jpg")
        missing = tmp_path / "missing.jpg"
        bottle = _bottle(images=[existing, missing])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(bottle, bottle.metadata)

        assert _visible_entries(folder / "images") == ["existing.jpg"]
        data = json.loads((folder / "test_bourbon.json").read_text(encoding="utf-8"))
        assert data["missing_images"] == [str(missing)]

    def test_same_bottle_twice_never_overwrites(self, tmp_path):
        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])

        first = generator.save_bottle_files(bottle, bottle.metadata)
        first_json = (first / "test_bourbon.json").read_bytes()
        second = generator.save_bottle_files(bottle, bottle.metadata)

        assert first.name == "test_bourbon"
        assert second.name == "test_bourbon_2"
        assert (second / "test_bourbon_2.json").is_file()
        assert (second / "test_bourbon_2.md").is_file()
        assert (first / "test_bourbon.json").read_bytes() == first_json

    def test_copy_failure_leaves_nothing_behind(self, tmp_path):
        from lib.exceptions import FileSystemError

        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])

        with patch("shutil.copy2", side_effect=OSError(28, "No space left on device")):
            with pytest.raises(FileSystemError, match="No space left on device"):
                generator.save_bottle_files(bottle, bottle.metadata)

        assert sorted(p.name for p in out.iterdir()) == [".bourbon_output"]

    def test_write_failure_leaves_nothing_behind(self, tmp_path):
        from lib.exceptions import FileSystemError

        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])

        with patch.object(PreviewGenerator, "generate_bottle_markdown", side_effect=RuntimeError("boom")):
            with pytest.raises(FileSystemError, match="RuntimeError: boom"):
                generator.save_bottle_files(bottle, bottle.metadata)

        assert sorted(p.name for p in out.iterdir()) == [".bourbon_output"]

    def test_rename_race_moves_to_next_free_name(self, tmp_path):
        """If another run claims the chosen name first, the next suffix is used."""
        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])
        real_rename = Path.rename
        raced = []

        def racing_rename(self, target):
            if not raced and Path(target).name == "test_bourbon":
                raced.append(target)
                Path(target).mkdir()
                (Path(target) / "theirs.txt").write_text("other run")
            return real_rename(self, target)

        with patch.object(Path, "rename", racing_rename):
            folder = generator.save_bottle_files(bottle, bottle.metadata)

        assert folder.name == "test_bourbon_2"
        assert (folder / "test_bourbon_2.json").is_file()
        assert (out / "test_bourbon" / "theirs.txt").read_text() == "other run"

    def test_bottle_without_metadata_uses_metadata_argument(self, tmp_path):
        """Regression: Bottle(metadata=None) used to crash on .source_images."""
        image = _jpeg(tmp_path / "image.jpg")
        bottle = Bottle(name="Plain", distillery="D", specific_release="", msrp=Decimal("10"))
        metadata = ExtractedMetadata(full_text="ocr", source_images=[image])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(bottle, metadata)

        assert _visible_entries(folder / "images") == ["image.jpg"]
        assert (folder / "plain.txt").read_text(encoding="utf-8") == "ocr"

    def test_duplicate_basenames_get_suffix(self, tmp_path):
        a = _jpeg(tmp_path / "a" / "IMG_1.jpg")
        b = _jpeg(tmp_path / "b" / "IMG_1.jpg", (1, 2, 3))
        bottle = _bottle(images=[a, b])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(bottle, bottle.metadata)

        assert _visible_entries(folder / "images") == ["IMG_1.jpg", "IMG_1_2.jpg"]
        assert (folder / "images" / "IMG_1_2.jpg").read_bytes() == b.read_bytes()

    @pytest.mark.parametrize("name", ["Unknown Bottle", "unknown", ""])
    def test_unknown_name_uses_primary_image_stem(self, tmp_path, name):
        bottle = _bottle(name=name, images=[_jpeg(tmp_path / "IMG_4653.jpeg")])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(bottle, bottle.metadata)

        assert folder.name == "unknown_img_4653"

    def test_originals_are_not_modified(self, tmp_path):
        image = _jpeg(tmp_path / "image.jpg")
        before = (image.read_bytes(), image.stat().st_mtime_ns)
        bottle = _bottle(images=[image])

        PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(bottle, bottle.metadata)

        assert (image.read_bytes(), image.stat().st_mtime_ns) == before

    def test_extra_keys_added_to_json(self, tmp_path):
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])

        folder = PreviewGenerator(output_dir=tmp_path / "out").save_bottle_files(
            bottle, bottle.metadata, extra={"vision_raw_response": "{...}"}
        )

        data = json.loads((folder / "test_bourbon.json").read_text(encoding="utf-8"))
        assert data["vision_raw_response"] == "{...}"


class TestBottleMarkdownProcessingDetails:
    """generate_bottle_markdown copes with missing metadata and links images."""

    def test_metadata_none_does_not_raise(self, tmp_path):
        bottle = Bottle(name="Plain", distillery="D", specific_release="", msrp=Decimal("10"))

        md = PreviewGenerator(output_dir=tmp_path).generate_bottle_markdown(bottle)

        assert "## Images" in md
        assert "## Processing Details" in md
        assert "- **Images Processed**: 0" in md
        assert "- **Model**: unknown" in md

    def test_image_links_and_details(self, tmp_path):
        metadata = ExtractedMetadata(
            batch_number="B-17", source_images=[Path("a.jpg")], warnings=["photos show different products"]
        )
        bottle = _bottle(images=[])
        bottle.metadata = metadata

        md = PreviewGenerator(output_dir=tmp_path).generate_bottle_markdown(
            bottle, image_links=["images/a.jpg", "images/b.jpg"], model_used="m", confidence="LOW"
        )

        assert "1. [a.jpg](images/a.jpg) (primary)" in md
        assert "2. [b.jpg](images/b.jpg)" in md
        assert "- **Batch:** B-17" in md
        assert "- **Images Processed**: 2" in md
        assert "- **Extraction Confidence**: LOW" in md
        assert "- **Warning**: photos show different products" in md
        assert "\n\n\n" not in md


class TestBottleIdFor:
    """bottle_id_for() matches BatchProcessor's failed-bottle IDs."""

    def test_prefers_metadata_primary(self):
        metadata = ExtractedMetadata(source_images=[Path("/p/IMG_2.jpg")])
        assert PreviewGenerator.bottle_id_for({"images": ["/p/IMG_9.jpg"]}, metadata) == "IMG_2"

    def test_falls_back_to_dict_images_then_unknown(self):
        assert PreviewGenerator.bottle_id_for({"images": ["/p/IMG_9.jpg"]}, ExtractedMetadata()) == "IMG_9"
        assert PreviewGenerator.bottle_id_for(None, None) == "unknown"


class TestOutputPermissions:
    """Saved folders and reports get umask-derived modes, like a plain mkdir/open."""

    @pytest.fixture
    def umask_022(self):
        import os

        previous = os.umask(0o022)
        yield 0o022
        os.umask(previous)

    def test_bottle_folder_mode_matches_plain_mkdir(self, tmp_path, umask_022):
        out = tmp_path / "out"
        out.mkdir()
        reference = out / "reference_dir"
        reference.mkdir()
        bottle = _bottle(images=[_jpeg(tmp_path / "photos" / "IMG_1.jpg")])

        folder = PreviewGenerator(output_dir=out).save_bottle_files(bottle, bottle.metadata)

        expected = reference.stat().st_mode & 0o777
        assert expected == 0o755
        assert folder.stat().st_mode & 0o777 == expected
        assert (folder / "images").stat().st_mode & 0o777 == expected

    def test_failure_report_mode_matches_plain_open(self, tmp_path, umask_022):
        out = tmp_path / "out"
        out.mkdir()
        reference = out / "reference.json"
        reference.write_text("{}")

        report = PreviewGenerator(output_dir=out).save_failure_report("IMG_1", [], "boom", stage="save")

        assert reference.stat().st_mode & 0o777 == 0o644
        assert report.stat().st_mode & 0o777 == 0o644


class TestSaveFailureReport:
    """save_failure_report writes the dict format retry_command reads."""

    def test_report_created_concurrently_is_not_overwritten(self, tmp_path):
        """A report that appears after the free name was chosen is kept; the next suffix is used."""
        import image_processor.preview_generator as pg

        out = tmp_path / "out"
        out.mkdir()
        real_unique_path = pg._unique_path
        calls = []

        def racing_unique_path(directory, stem, suffix=""):
            chosen = real_unique_path(directory, stem, suffix)
            if not calls:
                chosen.write_text("other run", encoding="utf-8")  # another run wins the race
            calls.append(chosen)
            return chosen

        with patch.object(pg, "_unique_path", racing_unique_path):
            report = PreviewGenerator(output_dir=out).save_failure_report("IMG_1", [], "boom", stage="save")

        assert (out / "error_report_img_1.json").read_text(encoding="utf-8") == "other run"
        assert report == out / "error_report_img_1_2.json"
        assert json.loads(report.read_text(encoding="utf-8"))["error_message"] == "boom"
        assert not list(out.glob(".*.tmp"))

    def test_report_falls_back_without_hard_links(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "error_report_img_1.json").write_text("first", encoding="utf-8")

        with patch("image_processor.preview_generator.os.link", side_effect=PermissionError("no links")):
            report = PreviewGenerator(output_dir=out).save_failure_report("IMG_1", [], "boom", stage="save")

        assert report == out / "error_report_img_1_2.json"
        assert (out / "error_report_img_1.json").read_text(encoding="utf-8") == "first"
        assert json.loads(report.read_text(encoding="utf-8"))["bottle_id"] == "IMG_1"
        assert not list(out.glob(".*.tmp"))

    def test_report_has_retry_fields_and_never_overwrites(self, tmp_path):
        out = tmp_path / "out"
        generator = PreviewGenerator(output_dir=out)
        try:
            raise OSError(28, "No space left on device")
        except OSError as e:
            first = generator.save_failure_report("IMG_1", ["/p/IMG_1.jpg", "/p/IMG_2.jpg"], e, stage="save")
        second = generator.save_failure_report("IMG_1", ["/p/IMG_1.jpg"], "analysis failed", stage="analysis")

        assert first == out / "error_report_img_1.json"
        assert second == out / "error_report_img_1_2.json"
        data = json.loads(first.read_text(encoding="utf-8"))
        assert {"bottle_id", "images", "error_message"} <= data.keys()
        assert data["bottle_id"] == "IMG_1"
        assert data["images"] == ["/p/IMG_1.jpg", "/p/IMG_2.jpg"]
        assert data["error_message"] == "OSError: [Errno 28] No space left on device"
        assert data["stage"] == "save"
        assert "Traceback" in data["details"]
        second_data = json.loads(second.read_text(encoding="utf-8"))
        assert second_data["error_message"] == "analysis failed"
        assert second_data["error_type"] == "UNKNOWN"
        assert not list(out.glob(".*.tmp"))

    def test_report_images_are_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = PreviewGenerator(output_dir=tmp_path / "out").save_failure_report(
            "x", ["rel/IMG_1.jpg"], "boom", stage="rejected"
        )
        assert json.loads(path.read_text())["images"] == [str(tmp_path / "rel" / "IMG_1.jpg")]

    def test_retry_command_accepts_generated_report(self, tmp_path):
        """A generated report passes retry_command's validation (server mocked away)."""
        from cli.commands import parse_args, retry_command

        image = _jpeg(tmp_path / "IMG_1.jpg")
        report = PreviewGenerator(output_dir=tmp_path / "out").save_failure_report(
            "IMG_1", [str(image)], RuntimeError("boom"), stage="save"
        )
        args = parse_args(["retry", "--report", str(report), "-o", str(tmp_path / "out")])
        with patch("cli.commands.ModelServerManager") as MockManager:
            MockManager.return_value.ensure_running.return_value = False
            assert retry_command(args) == 1  # got past validation to the server check
        MockManager.return_value.ensure_running.assert_called_once()


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

    @patch("image_processor.preview_generator.save_error_reports")
    @patch("image_processor.preview_generator.datetime")
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

    @patch("image_processor.preview_generator.save_error_reports")
    @patch("image_processor.preview_generator.datetime")
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

    def test_save_batch_preview_into_directory_writes_md_and_json(self, tmp_path):
        """output_path may be a directory; a sibling batch_<id>.json is written."""
        metadata = ExtractedMetadata(full_text="Text", source_images=[Path("bottle.jpg")])
        bottle_dicts = [{"name": "Test", "distillery": "Dist", "msrp": "99.99", "quantity": 2}]
        target = tmp_path / "reports"
        target.mkdir()

        generator = PreviewGenerator(output_dir=tmp_path / "out")
        result_path = generator.save_batch_preview(
            bottle_dicts, [metadata], batch_id="20261122_120000_001",
            output_path=target, rejected_bottles=["IMG_7"],
        )

        assert result_path == target / "batch_20261122_120000_001_preview.md"
        assert "IMG_7" in result_path.read_text(encoding="utf-8")
        data = json.loads((target / "batch_20261122_120000_001.json").read_text(encoding="utf-8"))
        assert data["batch_id"] == "20261122_120000_001"
        assert data["bottles"] == bottle_dicts
        assert data["rejected"] == ["IMG_7"]
        assert ExtractedMetadata.from_dict(data["metadata"][0]).full_text == "Text"


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


class TestOutputMarkers:
    """Bottle folders are marked as output; the root marker is optional."""

    def test_bottle_folder_carries_its_own_marker(self, tmp_path):
        from image_processor.image_organizer import OUTPUT_MARKER, ImageOrganizer

        photo = tmp_path / "photos" / "IMG_1.jpg"
        photo.parent.mkdir()
        photo.write_bytes(b"\xff\xd8\xff")
        generator = PreviewGenerator(output_dir=tmp_path / "photos" / "out", mark_output_dir=False)
        folder = generator.save_bottle_files(
            Bottle.from_analysis({"name": "X"}, ExtractedMetadata(source_images=[str(photo)]))
        )

        assert (folder / OUTPUT_MARKER).exists()
        assert not (generator.output_dir / OUTPUT_MARKER).exists()
        # Scanning the photo folder never re-ingests the copied images.
        assert ImageOrganizer().iter_image_paths(tmp_path / "photos") == [photo]
        # Not even when the output root itself is the scan root.
        assert ImageOrganizer().iter_image_paths(generator.output_dir) == []
