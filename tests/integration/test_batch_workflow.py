"""Integration tests for batch processing workflow.

Tests verify that groups pass through batching intact, the real batch preview
report, and the feedback loop (corrections applied by bottle ID) using
generated test images. The model is always mocked.
"""

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image as PILImage

from image_processor.batch_processor import BatchProcessor
from image_processor.corrections import apply_corrections
from image_processor.preview_generator import PreviewGenerator
from models.bottle import Bottle, Correction, ExtractedMetadata
from models.image import Image


class TestBatchingKeepsGroupsIntact:
    """Groups from ImageOrganizer pass through batching whole and are analyzed once.

    Capture-time grouping already keeps every bottle's photos together, so a
    batch never splits a bottle.
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

        processor = BatchProcessor(batch_size=2)
        seen: list[list[str]] = []

        def analyze(group, vision_model=None, **_kwargs):
            seen.append([Path(img.path).name for img in group])
            return {"name": f"Bottle {len(seen)}", "images": [str(img.path) for img in group]}, ExtractedMetadata()

        with patch.object(processor.bottle_analyzer, "process_single_bottle", side_effect=analyze):
            processor.start_run(groups)
            batches = processor.slice_into_batches(groups)
            results = [processor.process_batch(batch) for batch in batches]

        assert [len(b) for b in batches] == [2, 2, 1]
        assert seen == [[Path(img.path).name for img in g] for g in groups]  # each group once, whole
        assert sum(len(r.bottle_dicts) for r in results) == 5

    def test_single_photo_groups_processed_independently(self, tmp_path):
        images = []
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        batch_processor = BatchProcessor(batch_size=3)

        with patch.object(batch_processor.bottle_analyzer, 'process_single_bottle') as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, ExtractedMetadata()),
                ({"name": "Bottle 2", "distillery": "Test", "msrp": "60"}, ExtractedMetadata()),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "70"}, ExtractedMetadata()),
            ]

            result = batch_processor.process_batch([[img] for img in images])

            assert mock_process.call_count == 3
            assert len(result.bottle_dicts) == 3


class TestFeedbackLoop:
    """Corrections from review or a corrections file reach only the bottle they name."""

    def test_feedback_applies_corrections_to_the_named_bottle(self):
        bottle_dicts = [
            {"bottle_id": "IMG_1", "name": "Wrong Name", "distillery": "Four Roses", "msrp": None},
            {"bottle_id": "IMG_2", "name": "Wrong Name", "distillery": "Four Roses", "msrp": None},
        ]
        metadata_list = [ExtractedMetadata(), ExtractedMetadata()]
        corrections = [
            Correction(bottle_id="IMG_2", field_name="name", original_value="Wrong Name",
                       corrected_value="Four Roses Single Barrel"),
            Correction(bottle_id="IMG_2", field_name="msrp", original_value="", corrected_value="45.00"),
        ]

        _, unapplied = apply_corrections(bottle_dicts, metadata_list, corrections)

        assert unapplied == []
        assert bottle_dicts[0]["name"] == "Wrong Name"  # same old value, different bottle
        corrected = Bottle.from_analysis(bottle_dicts[1], corrections=bottle_dicts[1]["corrections"])
        assert corrected.name == "Four Roses Single Barrel"
        assert str(corrected.msrp) == "45.00"
        assert [c.bottle_id for c in corrected.corrections] == ["IMG_2", "IMG_2"]

    def test_corrected_values_reach_the_saved_batch_preview(self, tmp_path):
        bottle_dicts = [{"bottle_id": "IMG_1", "name": "Old Name", "distillery": "Old Distillery", "msrp": "50.00"}]
        metadata_list = [ExtractedMetadata(source_images=[Path("IMG_1.jpg")])]
        apply_corrections(
            bottle_dicts,
            metadata_list,
            [Correction(bottle_id="IMG_1", field_name="name", original_value="Old Name", corrected_value="New Name")],
        )

        preview = PreviewGenerator(tmp_path).save_batch_preview(bottle_dicts, metadata_list, batch_id="b1")

        markdown = preview.read_text(encoding="utf-8")
        assert "New Name" in markdown
        assert "Old Name" not in markdown
        data = json.loads((tmp_path / "batch_b1.json").read_text(encoding="utf-8"))
        assert data["bottles"][0]["corrections"][0]["bottle_id"] == "IMG_1"


class TestBatchWorkflowEndToEnd:
    """Test complete batch workflow end-to-end."""

    def test_full_batch_workflow_from_images_to_report(self, tmp_path):
        """Images -> batches -> mocked analysis -> real batch preview report."""
        images = []
        for i in range(6):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        batch_processor = BatchProcessor(batch_size=2)

        with patch.object(batch_processor.bottle_analyzer, "process_single_bottle") as mock_process:
            mock_process.side_effect = [
                (
                    {"name": f"Bottle {k}", "distillery": f"Distillery {k}", "msrp": str(25 * (k + 1))},
                    ExtractedMetadata(source_images=[images[k - 1].path]),
                )
                for k in range(1, 7)
            ]
            batches = batch_processor.slice_into_batches([[img] for img in images])
            results = [batch_processor.process_batch(batch) for batch in batches]

        assert [len(r.bottle_dicts) for r in results] == [2, 2, 2]
        all_bottles = [d for r in results for d in r.bottle_dicts]
        all_metadata = [m for r in results for m in r.metadata_list]
        assert [d["name"] for d in all_bottles] == [f"Bottle {k}" for k in range(1, 7)]

        out = tmp_path / "out"
        preview = PreviewGenerator(out).save_batch_preview(all_bottles, all_metadata, batch_id="e2e")

        assert preview == out / "batch_e2e_preview.md"
        markdown = preview.read_text(encoding="utf-8")
        for k in range(1, 7):
            assert f"Bottle {k}" in markdown
        assert (out / "batch_e2e.json").exists()

    def test_batch_with_errors_continues_processing(self, tmp_path):
        """One bottle failing is recorded with its reason; the others still succeed."""
        images = []
        for i in range(3):
            img_path = tmp_path / f"IMG_{i:04d}.jpg"
            _create_test_image(img_path)
            images.append(Image(path=str(img_path), category="BOTTLE", is_primary=True))

        batch_processor = BatchProcessor(batch_size=3)

        with patch.object(batch_processor.bottle_analyzer, "process_single_bottle") as mock_process:
            mock_process.side_effect = [
                ({"name": "Bottle 1", "distillery": "Test", "msrp": "50"}, ExtractedMetadata()),
                RuntimeError("Processing failed"),
                ({"name": "Bottle 3", "distillery": "Test", "msrp": "75"}, ExtractedMetadata()),
            ]
            result = batch_processor.process_batch([[img] for img in images])

        assert mock_process.call_count == 3
        assert [d["name"] for d in result.bottle_dicts] == ["Bottle 1", "Bottle 3"]
        assert result.failed_bottles == ["IMG_0001"]
        assert isinstance(result.errors["IMG_0001"], RuntimeError)
        assert str(result.errors["IMG_0001"]) == "Processing failed"


# Helper functions

def _create_test_image(path: Path) -> None:
    """Create a simple test image.

    Args:
        path: Path where to save the image.
    """
    img = PILImage.new('RGB', (400, 600), color='white')
    img.save(path)
