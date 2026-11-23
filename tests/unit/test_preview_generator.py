"""Unit tests for PreviewGenerator class.

Tests markdown generation, file operations, and interactive feedback handling.
"""

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from image_processor.preview_generator import PreviewGenerator
from models.bottle import Bottle, Correction, ExtractedMetadata, Quantity, ReferenceLink
from models.facts import FactSheet
from tests.unit.test_enrichment import checked_bottle


class TestPreviewGeneratorInitialization:
    """Test PreviewGenerator initialization."""

    def test_output_dir_is_required(self):
        with pytest.raises(TypeError):
            PreviewGenerator()  # type: ignore[call-arg]

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
    """Markdown of bottles without a fact sheet (built directly, not by the pipeline)."""

    def test_generate_bottle_markdown_basic(self, tmp_path):
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
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)

        assert markdown.startswith("# Test Bourbon\n")
        assert "**Status:** NOT publish-ready: not fact-checked (no fact sheet)" in markdown
        assert "| Distillery | Test Distillery | - | not fact-checked |" in markdown
        assert "- **MSRP:** $99.99" in markdown
        assert "## Images" in markdown
        assert "1. image1.jpg (primary)" in markdown
        assert "2. image2.jpg" in markdown

    def test_generate_bottle_markdown_with_all_fields(self, tmp_path):
        metadata = ExtractedMetadata(full_text="OCR text", selected_text="OCR text", source_images=[Path("image.jpg")])
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
                ReferenceLink(label="Review", url="https://example.com/review", source_class="reference",
                              retrieved_at="2026-11-23T01:02:03+00:00", used_as_evidence=True),
            ],
            quantity=Quantity(count=2, price=Decimal("149.99")),
            corrections=[
                Correction(bottle_id="IMG_1", field_name="age", original_value="10 years", corrected_value="12 years")
            ],
            metadata=metadata,
        )
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)

        assert "| Proof | 100 |" in markdown
        assert "| Age | 12 years |" in markdown
        assert "| Release | Limited Edition 2025 |" in markdown
        assert "| Tasting notes | Caramel, vanilla, and oak notes with a smooth finish. |" in markdown
        assert "- [Buy Now](https://example.com/buy) (unclassified, retrieved -, not cited)" in markdown
        assert "- [Review](https://example.com/review) (reference, retrieved 2026-11-23, cited)" in markdown
        assert "Value per bottle: $149.99" in markdown
        assert "Total value: $299.98" in markdown
        assert '- **age**: "10 years" -> "12 years" (interactive)' in markdown

    def test_generate_bottle_markdown_without_optional_fields(self, tmp_path):
        metadata = ExtractedMetadata(full_text="OCR text", selected_text="OCR text", source_images=[Path("image.jpg")])
        bottle = Bottle(
            name="Minimal Bourbon", distillery="Minimal Distillery", specific_release="Regular",
            msrp=Decimal("50.00"), metadata=metadata,
        )
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)

        assert "| Tasting notes | Unknown: not fact-checked |" in markdown
        assert "No web pages were used." in markdown
        assert "## Corrections" not in markdown
        assert "## Quantity" in markdown

    def test_generate_bottle_markdown_with_single_quantity(self, tmp_path):
        bottle = Bottle(
            name="Single Bottle", distillery="Distillery", specific_release="Regular", msrp=Decimal("75.00"),
            quantity=Quantity(count=1, price=Decimal("75.00")),
            metadata=ExtractedMetadata(source_images=[Path("image.jpg")]),
        )
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)

        assert "Total quantity: 1" in markdown
        assert "Value per bottle" not in markdown
        assert "Total value: $75.00" in markdown

    def test_generate_bottle_markdown_with_multiple_quantity(self, tmp_path):
        bottle = Bottle(
            name="Case of Bourbon", distillery="Distillery", specific_release="Regular", msrp=Decimal("75.00"),
            quantity=Quantity(count=12, price=Decimal("75.00")),
            metadata=ExtractedMetadata(source_images=[Path("image.jpg")]),
        )
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)

        assert "Total quantity: 12" in markdown
        assert "Value per bottle: $75.00" in markdown
        assert "Total value: $900.00" in markdown

    def test_unknown_msrp_renders_unknown_not_zero(self, tmp_path):
        """No MSRP: the .md says Unknown and the JSON has nulls, never $0.00."""
        bottle = Bottle.from_analysis(
            {"name": "George T. Stagg", "abv": "65.1%", "label_price": "99.99"},
            ExtractedMetadata(source_images=[Path("IMG_4668.jpeg")]),
            quantity=2,
        )
        generator = PreviewGenerator(output_dir=tmp_path)

        markdown = generator.generate_bottle_markdown(bottle)

        assert "- **MSRP:** Unknown: not fact-checked" in markdown
        assert "- **Retail:** $99.99: Price visible on label (retail, not MSRP)" in markdown
        assert "Total value: Unknown" in markdown
        assert "Value per bottle" not in markdown
        assert "$0.00" not in markdown
        assert "| ABV | 65.1% |" in markdown

        folder = generator.save_bottle_files(bottle)
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert data["msrp"] is None
        assert data["total_value"] is None
        assert data["label_price"] == "99.99"

    def test_extracted_timestamp_has_offset(self, tmp_path):
        bottle = Bottle.from_analysis({"name": "X"})
        markdown = PreviewGenerator(tmp_path).generate_bottle_markdown(bottle)
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
        assert (folder / "test_bourbon.txt").read_text(encoding="utf-8") == "OCR text\nmore\n\nweb: not researched\n"

        data = json.loads((folder / "test_bourbon.json").read_text(encoding="utf-8"))
        assert data["msrp"] == "99.99"  # a Bottle built directly keeps its msrp
        assert len(data["metadata"]["source_images"]) == 2
        assert data["images"] == ["images/image1.jpg", "images/image2.jpg"]
        assert data["source_images"] == [str(img1), str(img2)]
        assert data["missing_images"] == []
        assert data["bottle_id"] == "image1"
        assert data["model_used"] == "qwen3.6:35b-a3b"
        assert data["confidence"] == "HIGH"
        assert data["schema_version"] == 2
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
        assert (folder / "plain.txt").read_text(encoding="utf-8") == "ocr\n\nweb: not researched\n"

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
        assert "| Batch | B-17 |" in md
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


class TestPreviewGeneratorBatchReport:
    """Test batch preview report generation."""

    def test_generate_batch_preview_report_basic(self, tmp_path):
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

        generator = PreviewGenerator(tmp_path)
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "# Batch Preview - Batch 1" in markdown
        assert "Total bottles in batch: 2" in markdown
        assert "## Grouping Summary" in markdown
        assert "| Bottle Name | Quantity | Images | Confidence |" in markdown
        assert "## Extracted Metadata Details" in markdown

    def test_generate_batch_preview_report_confidence_calculation(self, tmp_path):
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

        generator = PreviewGenerator(tmp_path)

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

    def test_generate_batch_preview_report_with_rejected_bottles(self, tmp_path):
        """Test batch preview includes rejected bottles."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test Bottle", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]
        rejected_bottles = ["bottle1", "bottle2"]

        generator = PreviewGenerator(tmp_path)
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

    def test_generate_batch_preview_report_low_confidence_section(self, tmp_path):
        """Test low-confidence bottles are highlighted."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        # Only msrp field = 1 field (LOW confidence)
        bottle_dicts = [{"msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator(tmp_path)
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "## Low-Confidence or Rejected Items" in markdown
        assert "**Low Confidence:**" in markdown
        assert "- Bottle 1" in markdown

    def test_generate_batch_preview_report_with_ocr_preview(self, tmp_path):
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

        generator = PreviewGenerator(tmp_path)
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "**OCR Preview:**" in markdown
        assert "```" in markdown
        # Should be truncated with ellipsis
        assert "..." in markdown
        # Full text should not be present
        assert len(long_text) > 400  # Longer than what's in preview

    def test_generate_batch_preview_report_feedack_instructions(self, tmp_path):
        """Test feedback instructions are included in report."""
        metadata = ExtractedMetadata(
            full_text="Text",
            selected_text="Text",
            source_images=[Path("bottle.jpg")],
        )
        bottle_dicts = [{"name": "Test", "distillery": "Dist", "msrp": "99.99"}]
        metadata_list = [metadata]

        generator = PreviewGenerator(tmp_path)
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "## Feedback Instructions" in markdown
        assert "**A** - Approve" in markdown
        assert "**R** - Reject" in markdown
        assert "**E** - Edit" in markdown
        assert "**C** - Settle conflicts" in markdown
        assert "**N** - Keep" in markdown
        assert "**Q** - Quit" in markdown
        assert "--approve <bottle_id>" in markdown
        assert "Merge" not in markdown and "Split" not in markdown
        assert "bourbon-process feedback --batch 1" in markdown
        assert "bottle_id" in markdown
        assert "feedback command" in markdown.lower()
        assert "retry failed" in markdown.lower()

    def test_generate_batch_preview_report_populates_optional_fields(self, tmp_path):
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

        generator = PreviewGenerator(tmp_path)
        markdown = generator.generate_batch_preview_report(
            bottle_dicts, metadata_list, batch_id=1
        )

        assert "**Proof:** 100" in markdown
        assert "**Age:** 12" in markdown
        assert "**Release:** Limited" in markdown


class TestPreviewGeneratorSaveBatchPreview:
    """Test saving batch preview reports."""

    @patch("builtins.open")
    def test_save_batch_preview_default_path(self, mock_open, tmp_path):
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

        generator = PreviewGenerator(tmp_path)
        result_path = generator.save_batch_preview(bottle_dicts, metadata_list, batch_id=2)

        # Default path should be {output_dir}/batch_2_preview.md
        assert result_path == tmp_path / "batch_2_preview.md"

    @patch("builtins.open")
    def test_save_batch_preview_custom_path(self, mock_open, tmp_path):
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

        generator = PreviewGenerator(tmp_path)
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
        assert data["schema_version"] == 2
        assert data["bottles"] == bottle_dicts
        assert data["rejected"] == ["IMG_7"]
        assert ExtractedMetadata.from_dict(data["metadata"][0]).full_text == "Text"


class TestPreviewGeneratorInteractivePrompt:
    """Interactive review: A approves, R rejects, E/C record corrections, N keeps, Q quits."""

    @staticmethod
    def _plain(n=1):
        metadata = ExtractedMetadata(full_text="Text", selected_text="Text", source_images=[Path("bottle1.jpg")])
        return [{"name": f"Bottle {i}", "distillery": "Dist", "quantity": 1} for i in range(1, n + 1)], [metadata] * n

    @patch("builtins.input")
    @patch("builtins.print")
    def test_accept_approves_every_bottle(self, mock_print, mock_input, tmp_path):
        mock_input.return_value = "A"
        bottles, metas = self._plain(2)
        bottles[1]["bottle_id"] = "bottle2"

        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(bottles, metas)

        assert (corrections, rejected, approvals) == ([], [], ["bottle1", "bottle2"])
        assert mock_input.call_count == 2

    @patch("builtins.input")
    @patch("builtins.print")
    def test_reject_bottle(self, mock_print, mock_input, tmp_path):
        mock_input.return_value = "R"
        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(*self._plain())
        assert (corrections, rejected, approvals) == ([], ["bottle1"], [])

    @patch("builtins.input")
    @patch("builtins.print")
    def test_next_keeps_without_approval(self, mock_print, mock_input, tmp_path):
        mock_input.return_value = "n"
        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(*self._plain(2))
        assert (corrections, rejected, approvals) == ([], [], [])
        assert mock_input.call_count == 2

    @patch("builtins.input")
    @patch("builtins.print")
    def test_edit_produces_a_correction_with_bottle_id(self, mock_print, mock_input, tmp_path):
        mock_input.side_effect = ["E", "nmae", "distillery", "New Distillery", "", "N"]
        bottles = [{"name": "Test Bottle", "distillery": "Old Distillery", "proof": "100", "age": "12"}]
        metas = [ExtractedMetadata(source_images=[Path("bottle1.jpg")])]

        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(bottles, metas)

        (correction,) = corrections
        assert correction["bottle_id"] == "bottle1"
        assert correction["field_name"] == "distillery"
        assert correction["original_value"] == "Old Distillery"
        assert correction["corrected_value"] == "New Distillery"
        assert (rejected, approvals) == ([], [])
        # The prompt only records the edit; apply_corrections changes the bottle.
        assert bottles[0]["distillery"] == "Old Distillery"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_second_edit_of_a_field_starts_from_the_first(self, mock_print, mock_input, tmp_path):
        mock_input.side_effect = ["E", "age", "10", "age", "11", "", "A"]
        corrections, _rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(
            [{"bottle_id": "B1", "name": "X", "age": "12"}], [ExtractedMetadata(source_images=[Path("b.jpg")])]
        )
        assert [(c["original_value"], c["corrected_value"]) for c in corrections] == [("12", "10"), ("10", "11")]
        assert approvals == ["B1"]  # approved after the edits

    @patch("builtins.input")
    @patch("builtins.print")
    def test_quit(self, mock_print, mock_input, tmp_path):
        mock_input.return_value = "Q"
        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(*self._plain(3))
        assert (corrections, rejected, approvals) == ([], [], [])
        assert mock_input.call_count == 1

    @patch("builtins.input")
    @patch("builtins.print")
    def test_invalid_action_asks_again(self, mock_print, mock_input, tmp_path):
        """An unknown key or a bare Enter is never taken as Accept: the prompt repeats."""
        mock_input.side_effect = ["X", "", "R"]
        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(*self._plain())
        assert (corrections, rejected, approvals) == ([], ["bottle1"], [])
        assert mock_input.call_count == 3

    @patch("builtins.input")
    @patch("builtins.print")
    def test_case_insensitive(self, mock_print, mock_input, tmp_path):
        mock_input.return_value = "a"
        _c, _r, approvals = PreviewGenerator(tmp_path).interactive_prompt(*self._plain())
        assert approvals == ["bottle1"]


class TestInteractivePromptActions:
    """The review shows facts, conflicts and blockers, and offers A/R/E/C/N/Q."""

    @patch("builtins.input", return_value="N")
    def test_shows_recorded_confidence_and_actions(self, mock_input, tmp_path, capsys):
        bottle = {
            "bottle_id": "IMG_1", "name": "Stagg", "distillery": "Buffalo Trace", "msrp": "99.99",
            "proof": "130", "age": "15", "confidence": "LOW",
        }
        PreviewGenerator(tmp_path).interactive_prompt([bottle], [ExtractedMetadata(source_images=[Path("IMG_1.jpg")])])
        out = capsys.readouterr().out
        assert "Confidence:  LOW" in out
        assert "[A]ccept (approve for publishing) | [R]eject | [E]dit | [C]onflicts | " in out
        assert "[N]ext (keep, not approved) | [Q]uit" in out
        assert "NOT publish-ready: not fact-checked" in out
        assert "[M]erge" not in out and "[S]plit" not in out

    @patch("builtins.input", return_value="N")
    def test_missing_confidence_is_unknown(self, mock_input, tmp_path, capsys):
        PreviewGenerator(tmp_path).interactive_prompt([{"name": "X"}], [ExtractedMetadata(source_images=[Path("a.jpg")])])
        assert "Confidence:  Unknown" in capsys.readouterr().out

    @pytest.mark.parametrize("action", ["M", "S"])
    def test_merge_and_split_are_not_actions(self, action, tmp_path):
        """M/S are unknown keys: the prompt asks again and records nothing for them."""
        with patch("builtins.input", side_effect=[action, "N", action, "Q"]) as mock_input, \
                patch("builtins.print"):
            corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt(
                [{"name": "A"}, {"name": "B"}],
                [ExtractedMetadata(source_images=[Path("a.jpg")]), ExtractedMetadata(source_images=[Path("b.jpg")])],
            )
        assert (corrections, rejected, approvals) == ([], [], [])
        assert mock_input.call_count == 4

    def test_handle_interactive_choice_removed(self):
        assert not hasattr(PreviewGenerator, "handle_interactive_choice")

    def test_shows_facts_conflicts_and_blockers(self, tmp_path, capsys):
        bottle, metadata, _sheet = checked_bottle()
        with patch("builtins.input", return_value="N"):
            PreviewGenerator(tmp_path).interactive_prompt([bottle], [metadata])
        out = capsys.readouterr().out
        assert "NOT publish-ready: unresolved conflict proof-1 (proof)" in out
        assert "Distillery     high    Example Creek Distillery" in out
        assert "Proof          -       CONFLICT (see below)" in out
        assert "proof-1 (proof): L1 100 vs W1 101" in out
        assert "MSRP           medium  $59.99" in out
        assert "Label price: $64.99 (retail, not MSRP)" in out
        assert "web: 3 pages consulted, 1 cited" in out


def _review_and_apply(bottle, metadata, inputs, tmp_path):
    """Run the prompt with ``inputs``, then apply its results the way the CLI does."""
    from cli.commands import _apply_approvals, _apply_review_corrections, _corrections_from_prompt

    with patch("builtins.input", side_effect=inputs), patch("builtins.print"):
        corrections, rejected, approvals = PreviewGenerator(tmp_path).interactive_prompt([bottle], [metadata])
    if corrections:
        _apply_review_corrections([bottle], [metadata], _corrections_from_prompt(corrections), "test")
    _apply_approvals([bottle], [metadata], approvals, "interactive")
    return corrections, rejected, approvals, FactSheet.from_dict(bottle["facts"])


class TestInteractiveFactsReview:
    """[C]onflicts, [A]ccept and [E]dit end in re-reconciled facts and a recomputed gate."""

    def test_conflict_resolved_by_picking_a_candidate(self, tmp_path):
        bottle, metadata, sheet = checked_bottle()
        assert any("unresolved conflict proof-1" in b for b in sheet.publish["blockers"])

        corrections, _r, approvals, new = _review_and_apply(bottle, metadata, ["C", "2", "N"], tmp_path)

        (correction,) = corrections
        assert correction["field_name"] == "proof"
        assert correction["corrected_value"] == "101 Proof"
        assert correction["original_value"] == ""
        assert approvals == []
        assert new.fields["proof"].status == "user"
        assert new.fields["proof"].value == 101.0
        conflict = next(c for c in new.conflicts if c.id == "proof-1")
        assert conflict.resolved
        assert not any("unresolved conflict" in b for b in new.publish["blockers"])
        assert new.publish["blockers"] == ["not approved by a reviewer"]
        assert bottle["proof"] == "101 Proof"
        assert bottle["corrections"][0]["source"] == "interactive"

    def test_conflict_resolved_by_typing_a_value_then_approved(self, tmp_path):
        bottle, metadata, _sheet = checked_bottle()

        _c, _r, approvals, new = _review_and_apply(bottle, metadata, ["C", "=100", "A"], tmp_path)

        assert approvals == ["IMG_1"]
        assert new.fields["proof"].value == 100.0
        assert new.review["approved"] is True and new.review["by"] == "interactive"
        assert new.publish["ready"] is True  # every check passes once a person settled and approved it
        assert new.publish["blockers"] == []

    def test_conflict_skip_changes_nothing(self, tmp_path):
        bottle, metadata, sheet = checked_bottle()
        corrections, _r, _a, new = _review_and_apply(bottle, metadata, ["C", "", "N"], tmp_path)
        assert corrections == []
        assert new.publish["blockers"] == sheet.publish["blockers"]

    def test_accept_approves_but_gate_still_blocks(self, tmp_path):
        bottle, metadata, _sheet = checked_bottle()
        _c, _r, approvals, new = _review_and_apply(bottle, metadata, ["A"], tmp_path)
        assert approvals == ["IMG_1"]
        assert new.review["approved"] is True
        assert "not approved by a reviewer" not in new.publish["blockers"]
        assert new.publish["ready"] is False  # the conflict still blocks

    def test_edit_of_a_field_the_copy_cites_marks_it_stale(self, tmp_path):
        bottle, metadata, sheet = checked_bottle()
        assert "age" in sheet.copy.cited_fields()

        corrections, _r, _a, new = _review_and_apply(bottle, metadata, ["E", "age", "9 years", "", "N"], tmp_path)

        (correction,) = corrections
        assert (correction["bottle_id"], correction["original_value"]) == ("IMG_1", "8 years")
        assert new.fields["age"].status == "user"
        assert new.copy.stale is True
        assert any(b.startswith("description is stale") for b in new.publish["blockers"])
        assert bottle["age"] == "9 years"

    def test_conflicts_action_without_conflicts(self, tmp_path, capsys):
        with patch("builtins.input", side_effect=["C", "N"]):
            corrections, _r, _a = PreviewGenerator(tmp_path).interactive_prompt(
                [{"name": "X"}], [ExtractedMetadata(source_images=[Path("a.jpg")])]
            )
        assert corrections == []
        assert "No unresolved conflicts." in capsys.readouterr().out


class TestBottleMarkdownGolden:
    """The full document for a fact-checked bottle (everything before the volatile tail)."""

    def test_golden(self):
        bottle_dict, metadata, _ = checked_bottle()
        md = Bottle.from_analysis(bottle_dict, metadata).to_markdown()
        assert md[: md.index("## Quantity")] == GOLDEN_MARKDOWN

    def test_untrusted_text_is_escaped(self):
        from image_processor.preview_generator import md_escape, md_link

        assert md_escape("a | b ] c [d] `e` *f* <script>") == r"a \| b \] c \[d\] \`e\` \*f\* &lt;script&gt;"
        assert md_escape("line\nbreak") == "line break"
        assert md_link("x", "javascript:alert(1)") == "x"
        assert md_link("t", "https://a.test/p (1)") == "[t](https://a.test/p%20%281%29)"

    def test_model_written_description_is_escaped(self):
        """A sentence can never smuggle a link, table pipe or HTML into the document; markers stay plain."""
        bottle_dict, metadata, sheet = checked_bottle()
        sheet.copy.sentences[1] = {**sheet.copy.sentences[1], "text": "Aged [8](http://evil.test) | <b>years</b>."}
        bottle_dict["facts"] = sheet.to_dict()
        md = Bottle.from_analysis(bottle_dict, metadata).to_markdown()
        desc = md[md.index("## Description"):md.index("## Sources")]
        assert r"Aged \[8\](http://evil.test) \| &lt;b&gt;years&lt;/b&gt;. [3]" in desc
        assert "](http://evil.test)" not in desc.replace(r"\]", "")

    def test_review_prints_strip_terminal_escape_sequences(self):
        from image_processor.preview_generator import _term

        assert _term("ok\x1b[2J\x07 quote\nnext") == "ok[2J quote next"

    def test_web_log_renders_cited_not_cited_and_skipped_with_reasons(self):
        bottle_dict, metadata, _ = checked_bottle()
        md = Bottle.from_analysis(bottle_dict, metadata).to_markdown()
        log = md[md.index("## Web Research Log"):]
        assert "| 1 | [examplecreek.test](https://examplecreek.test/single-barrel) | official | 2026-11-23 | cited | cited for distillery, age, msrp |" in log
        assert "| 2 | [shop.test](https://shop.test/ec-sb) | retailer | 2026-11-23 | fetched_not_cited | W2: no verifiable quote accepted (quote_not_found) |" in log
        assert "| 3 | forum.test | - | - | skipped | deny-listed |" in log  # never fetched: no link

    def test_skipped_web_run_renders_the_section_with_the_reason(self):
        from models.evidence import WebResearch

        bottle = Bottle.from_analysis({
            "name": "X",
            "web_research": WebResearch(status="skipped", reason="identity too uncertain to search").to_dict(),
        })
        md = bottle.to_markdown()
        log = md[md.index("## Web Research Log"):md.index("## Quantity")]
        assert "- Status: skipped (identity too uncertain to search)" in log
        assert "- Provider: none" in log
        assert "- Queries: none sent" in log
        assert "web: 0 pages consulted, 0 cited (skipped: identity too uncertain to search)" in log
        assert "No pages were consulted." in log

    def test_user_asserted_msrp(self):
        from image_processor.fact_checker import FactChecker
        from models.facts import FactInput, UserFact

        sheet = FactChecker.deterministic(
            FactInput("B", {"name": "X"}, "high", ["a.jpg"], {}, (UserFact("msrp", "49.99", "feedback:c.json"),)),
            "no text model",
        )
        md = Bottle.from_analysis({"name": "X", "facts": sheet.to_dict()}).to_markdown()
        assert "- **MSRP:** $49.99: user-asserted (feedback:c.json)" in md
        assert "| MSRP | $49.99 (user-asserted) | high |" in md


class TestSaveV2Files:
    def test_evidence_files_written_with_matching_sha256(self, tmp_path):
        import hashlib

        bottle_dict, metadata, _ = checked_bottle()
        image = _jpeg(tmp_path / "IMG_1.jpg")
        bottle_dict["images"] = [str(image)]
        metadata.source_images = [image]
        folder = PreviewGenerator(tmp_path / "out").save_bottle_files(Bottle.from_analysis(bottle_dict, metadata))

        data = json.loads((folder / f"{folder.name}.json").read_text())
        for ev in data["web_research"]["evidence"]:
            path = folder / "evidence" / f"{ev['source_id']}.txt"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == ev["sha256"]
        sources = {s["id"]: s for s in data["facts"]["sources"]}
        assert sources["W1"]["evidence_path"] == "evidence/W1.txt"
        assert (folder / sources["W1"]["evidence_path"]).is_file()
        txt = (folder / f"{folder.name}.txt").read_text()
        assert txt.endswith("web: 3 pages consulted, 1 cited\n")
        assert data["batch_number"] is None  # from the facts (unknown), not a guess

    def test_extra_files_are_written_inside_the_atomic_save(self, tmp_path):
        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])
        out = tmp_path / "out"
        folder = PreviewGenerator(out).save_bottle_files(
            bottle, bottle.metadata, extra_files=[("notes/a.txt", "héllo"), ("raw.bin", b"\x00\x01")]
        )
        assert (folder / "notes" / "a.txt").read_bytes() == "héllo".encode()
        assert (folder / "raw.bin").read_bytes() == b"\x00\x01"

    @pytest.mark.parametrize("bad", ["../escape.txt", "/abs.txt", "a/../../b.txt"])
    def test_extra_file_outside_the_folder_fails_and_leaves_nothing(self, tmp_path, bad):
        from lib.exceptions import FileSystemError

        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])
        out = tmp_path / "out"
        with pytest.raises(FileSystemError):
            PreviewGenerator(out).save_bottle_files(bottle, bottle.metadata, extra_files=[(bad, "x")])
        assert _visible_entries(out) == []
        assert not list(out.glob(".*.partial"))
        assert not (tmp_path / "escape.txt").exists()

    def test_failure_writing_extra_files_leaves_nothing(self, tmp_path):
        from lib.exceptions import FileSystemError

        bottle = _bottle(images=[_jpeg(tmp_path / "image.jpg")])
        out = tmp_path / "out"
        with pytest.raises(FileSystemError, match="written twice"):
            PreviewGenerator(out).save_bottle_files(bottle, bottle.metadata, extra_files=[("a.txt", "1"), ("a.txt", "2")])
        assert _visible_entries(out) == []
        assert not list(out.glob(".*.partial"))

    def test_vendor_ref_saved_with_vendor_category(self, tmp_path):
        from models.evidence import VendorImage

        cache = _jpeg(tmp_path / "cache" / "abc123.jpg", (1, 1, 1))
        vendor = VendorImage(cache_path=str(cache), image_url="https://examplecreek.test/i.jpg",
                             page_url="https://examplecreek.test/sb", retrieved_at="2026-11-23T00:00:00+00:00",
                             verdict={"same_product": True, "confidence": "high", "same_release": "yes"})
        image = _jpeg(tmp_path / "IMG_1.jpg")
        bottle = Bottle.from_analysis(
            {"name": "Example Creek", "images": [str(image)], "vendor_image": vendor.to_dict()},
            ExtractedMetadata(source_images=[image]),
        )
        folder = PreviewGenerator(tmp_path / "out").save_bottle_files(bottle)

        assert (folder / "images" / "vendor_ref.jpg").read_bytes() == cache.read_bytes()
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert data["vendor_image"]["path"] == "images/vendor_ref.jpg"
        assert data["vendor_image"]["category"] == "VENDOR"
        assert data["vendor_image"]["usage"] == "reference only - not licensed for publication"
        assert data["vendor_image"]["image_url"] == "https://examplecreek.test/i.jpg"
        assert data["images"] == ["images/IMG_1.jpg"]  # the photos; the vendor image is separate
        md = (folder / f"{folder.name}.md").read_text()
        assert "2. [vendor_ref.jpg](images/vendor_ref.jpg) (vendor reference, not for publication)" in md
        assert "- **Images Processed**: 1" in md

    def test_missing_vendor_cache_is_recorded_not_fatal(self, tmp_path):
        bottle = Bottle.from_analysis({"name": "X", "vendor_image": {"cache_path": str(tmp_path / "gone.jpg")}})
        folder = PreviewGenerator(tmp_path / "out").save_bottle_files(bottle)
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert data["vendor_image"]["path"] is None
        assert not (folder / "images" / "vendor_ref.jpg").exists()


class TestBatchPreviewV2Columns:
    def test_ready_conflicts_unknowns_verification_msrp_columns(self, tmp_path):
        bottle_dict, metadata, sheet = checked_bottle()
        markdown = PreviewGenerator(tmp_path).generate_batch_preview_report([bottle_dict], [metadata], batch_id="b1")

        assert "| Bottle Name | Quantity | Images | Confidence | Ready | Conflicts | Unknowns | Verification | MSRP (source) |" in markdown
        unknowns = sum(1 for f in sheet.fields.values() if f.status == "unknown")
        assert (f"| Example Creek Single Barrel | 1 | 1 | HIGH | N | 1 | {unknowns} | verified | "
                "$59.99 (W1 examplecreek.test) |") in markdown
        assert "**web: 3 pages consulted, 1 cited**" in markdown
        assert "**Label price:** $64.99 (retail, not MSRP)" in markdown
        assert "**Status:** NOT publish-ready: unresolved conflict proof-1" in markdown

    def test_bottle_without_facts(self, tmp_path):
        markdown = PreviewGenerator(tmp_path).generate_batch_preview_report(
            [{"name": "Plain"}], [ExtractedMetadata(source_images=[Path("a.jpg")])]
        )
        assert "| Plain | 1 | 1 | LOW | N | - | - | not run | Unknown (not fact-checked) |" in markdown
        assert "**web: not researched**" in markdown


GOLDEN_MARKDOWN = (
    '# Example Creek Single Barrel\n'
    '\n'
    '**Status:** NOT publish-ready: unresolved conflict proof-1 (proof): L1 100 vs W1 101; core field proof or abv not supported at medium confidence or better; not approved by a reviewer\n'
    '\n'
    '## Needs Review\n'
    '\n'
    '- **proof** (proof-1): L1 100 vs W1 101\n'
    '  1. 100 (label): [1] L1 "100 Proof"\n'
    '  2. 101 (web): [2] W1 "bottled at 101 proof"\n'
    '\n'
    '## Facts\n'
    '\n'
    '| Field | Value | Confidence | Sources |\n'
    '|---|---|---|---|\n'
    '| Name | Example Creek Single Barrel | medium | [1] label "Example Creek Single Barrel" |\n'
    '| Distillery | Example Creek Distillery | high | [1] label "Example Creek Distillery"; [2] W1 "Distilled by Example Creek Distillery" |\n'
    '| Bottler | Unknown: Bottler not part of the label reading; not stated in 2 fetched page(s) | - | - |\n'
    '| Release | Unknown: Specific release not read from label; not stated in 2 fetched page(s) | - | - |\n'
    '| Batch | Unknown: Batch number not read from label; not stated in 2 fetched page(s) | - | - |\n'
    '| Proof | CONFLICT: see Needs Review | - | - |\n'
    '| ABV | Unknown: Abv not read from label; not stated in 2 fetched page(s) | - | - |\n'
    '| Age | 8 years | medium | [2] W1 "Aged 8 years \\| small \\`batch\\`" |\n'
    '| Release year | Unknown: Release year not part of the label reading; not stated in 2 fetched page(s) | - | - |\n'
    '| Mash bill | Unknown: Mash bill not part of the label reading; not stated in 2 fetched page(s); 1 claim(s) rejected (W2 quote_not_found) | - | - |\n'
    '| Tasting notes | Unknown: Tasting notes not read from label; not stated in 2 fetched page(s) | - | - |\n'
    '| MSRP | $59.99 | medium | [2] W1 "Suggested retail price (MSRP) of $59.99" |\n'
    '\n'
    '## Prices\n'
    '\n'
    '- **MSRP:** $59.99: [examplecreek.test](https://examplecreek.test/single-barrel), "Suggested retail price (MSRP) of $59.99", retrieved 2026-11-23 (confidence medium)\n'
    '- **Retail:** $64.99: Price visible on label (retail, not MSRP)\n'
    '\n'
    '## Description (draft)\n'
    '\n'
    'Example Creek Single Barrel is distilled by Example Creek Distillery. [1][2] It is aged 8 years. [3]\n'
    '\n'
    '_Markers number the facts the sentence relies on (not the Sources list): [1] Name, [2] Distillery, [3] Age, [4] MSRP._\n'
    '\n'
    '## Sources\n'
    '\n'
    '1. **L1** label reading (vision model) of IMG_1.jpg\n'
    '2. **W1** [Single Barrel \\| Example Creek](https://examplecreek.test/single-barrel) (official, examplecreek.test, retrieved 2026-11-23); excerpt: `evidence/W1.txt`\n'
    '   - "Distilled by Example Creek Distillery"\n'
    '   - "bottled at 101 proof"\n'
    '   - "Aged 8 years \\| small \\`batch\\`"\n'
    '   - "Suggested retail price (MSRP) of $59.99"\n'
    '3. **W2** [Example Creek SB](https://shop.test/ec-sb) (retailer, shop.test, retrieved 2026-11-23); excerpt: `evidence/W2.txt`\n'
    '\n'
    '## Verification\n'
    '\n'
    '- Status: verified\n'
    '- Model: gpt-oss:120b\n'
    '- Think: extract high\n'
    '- Prompt version: facts-1\n'
    '- Claims accepted: 3; prices accepted: 1; proposals rejected: 1 (listed in the JSON)\n'
    '- Duration: 12.5 s; model calls: 2\n'
    '\n'
    '## References\n'
    '\n'
    '- [Single Barrel \\| Example Creek](https://examplecreek.test/single-barrel) (official, retrieved 2026-11-23, cited)\n'
    '- [Example Creek SB](https://shop.test/ec-sb) (retailer, retrieved 2026-11-23, not cited)\n'
    '\n'
    '## Web Research Log\n'
    '\n'
    '- Provider: duckduckgo\n'
    '- Status: ok\n'
    '- Searched: 2026-11-23T10:00:00+00:00\n'
    '- Queries: "example creek single barrel"\n'
    '- web: 3 pages consulted, 1 cited\n'
    '\n'
    '| # | Site | Class | Retrieved | Outcome | Used for / Reason |\n'
    '|---|---|---|---|---|---|\n'
    '| 1 | [examplecreek.test](https://examplecreek.test/single-barrel) | official | 2026-11-23 | cited | cited for distillery, age, msrp |\n'
    '| 2 | [shop.test](https://shop.test/ec-sb) | retailer | 2026-11-23 | fetched_not_cited | W2: no verifiable quote accepted (quote_not_found) |\n'
    '| 3 | forum.test | - | - | skipped | deny-listed |\n'
    '\n'
)


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
