"""End-to-end tests of the ``process`` save path.

Runs the real CLI ``main()`` with the real ImageOrganizer, BatchProcessor,
BottleAnalyzer and PreviewGenerator. Mocked: pre-flight checks (server and
Tesseract), the model server manager, OCR and the vision call. Nothing here
talks to a model server.
"""

import hashlib
import json
import shutil
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from cli import commands
from models.bottle import Bottle, ExtractedMetadata

LABELS = {
    "evan": dict(name="Evan Williams Black Label", distillery="Heaven Hill", msrp="15.99", proof="86 Proof"),
    "weller": dict(name="Weller Special Reserve", distillery="Buffalo Trace", msrp="$29.99", proof="90 Proof"),
}


def _jpeg(path: Path, color) -> None:
    PILImage.new("RGB", (64, 96), color).save(path, "JPEG")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """main() installs SIGTERM/SIGHUP/SIGQUIT handlers; give pytest its own back."""
    names = [n for n in ("SIGTERM", "SIGHUP", "SIGQUIT") if hasattr(signal, n)]
    previous = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield
    for n, handler in previous.items():
        signal.signal(getattr(signal, n), handler)


@pytest.fixture
def photos(tmp_path):
    """Two bottles: evan (front + back) and weller (front + identical copy -> quantity 2)."""
    d = tmp_path / "photos"
    d.mkdir()
    _jpeg(d / "evan_front.jpg", (200, 10, 10))
    _jpeg(d / "evan_back.jpg", (10, 200, 10))
    _jpeg(d / "weller_front.jpg", (10, 10, 200))
    (d / "weller_front_dup.jpg").write_bytes((d / "weller_front.jpg").read_bytes())
    return d


def _fake_vision(self, image_paths, ocr_text="", model=None):
    """Stand-in for BottleAnalyzer.analyze_with_vision_model.

    Lists only the photos it was given (as the real method does), so the
    analyzer and Bottle.from_analysis must add the rest of the group.
    """
    paths = [Path(p) for p in image_paths]
    key = paths[0].name.split("_")[0]
    hints = ocr_text if isinstance(ocr_text, str) else "\n".join(ocr_text)
    return ExtractedMetadata(
        full_text=hints, selected_text=hints, source_images=[str(p) for p in paths],
        vision_model=model, vision_confidence="high", **LABELS[key],
    )


def _run(argv, vision=_fake_vision, user_input=None):
    """Run ``main(argv)`` with server, OCR and vision mocked; return (rc, vision calls)."""
    calls: list[list[Path]] = []

    def spy(self, image_paths, ocr_text="", model=None):
        calls.append([Path(p) for p in image_paths])
        return vision(self, image_paths, ocr_text, model)

    manager = MagicMock()
    with patch.object(commands, "pre_flight_checks", return_value=0), \
            patch.object(commands, "build_server_manager", return_value=manager), \
            patch("image_processor.text_extractor.TextExtractor.extract_with_retry",
                  lambda self, p, **k: f"OCR {Path(p).name}"), \
            patch("image_processor.bottle_analyzer.BottleAnalyzer.analyze_with_vision_model", spy), \
            patch("builtins.input", side_effect=user_input or []):
        rc = commands.main(argv)
    manager.close.assert_called_once()
    return rc, calls


def _bottle_dirs(out: Path):
    return sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("."))


def test_end_to_end_writes_complete_bottle_folders(photos):
    out = photos / "processed"  # -o nested under -f on purpose
    before = {p.name: _md5(p) for p in photos.glob("*.jpg")}

    rc, calls = _run(["process", "-f", str(photos), "-o", str(out), "-b", "1"])

    assert rc == 0
    names = [p.name for p in _bottle_dirs(out)]
    assert names == ["evan_williams_black_label", "weller_special_reserve"]

    evan = out / "evan_williams_black_label"
    for ext in (".md", ".json", ".txt"):
        assert (evan / f"evan_williams_black_label{ext}").is_file()
    assert sorted(p.name for p in (evan / "images").iterdir()) == ["evan_back.jpg", "evan_front.jpg"]

    data = json.loads((evan / "evan_williams_black_label.json").read_text())
    assert data["name"] == "Evan Williams Black Label"
    assert data["msrp"] == "15.99"
    assert len(data["metadata"]["source_images"]) == 2
    assert sorted(data["images"]) == ["images/evan_back.jpg", "images/evan_front.jpg"]
    assert data["confidence"] == "HIGH"
    assert data["model_used"] == commands.DEFAULT_VISION_MODEL
    assert Bottle.from_json((evan / "evan_williams_black_label.json").read_text()).name == data["name"]

    txt = (evan / "evan_williams_black_label.txt").read_text()
    assert "OCR evan_front.jpg" in txt and "OCR evan_back.jpg" in txt
    md = (evan / "evan_williams_black_label.md").read_text()
    assert md.startswith("# Evan Williams Black Label")
    assert "images/evan_back.jpg" in md

    weller = json.loads((out / "weller_special_reserve" / "weller_special_reserve.json").read_text())
    assert weller["msrp"] == "29.99"
    assert weller["quantity"]["count"] == 2
    assert weller["quantity"]["price"] == "29.99"
    assert weller["total_value"] == "59.98"

    # One preview (markdown + JSON) per batch
    assert len(list(out.glob("batch_*_preview.md"))) == 2
    assert len([p for p in out.glob("batch_*.json")]) == 2
    # Nothing half-written, originals untouched, outputs never fed to the model
    assert not list(out.glob(".*.partial"))
    assert (out / ".bourbon_output").exists()
    assert before == {p.name: _md5(p) for p in photos.glob("*.jpg")}
    assert len(calls) == 2
    assert all(out not in p.parents for call in calls for p in call)


def test_second_run_never_clobbers_and_ignores_previous_outputs(photos):
    out = photos / "processed"
    _run(["process", "-f", str(photos), "-o", str(out), "-b", "all"])
    first = (out / "evan_williams_black_label" / "evan_williams_black_label.json").read_text()

    rc, calls = _run(["process", "-f", str(photos), "-o", str(out), "-b", "all"])

    assert rc == 0
    assert len(calls) == 2  # two groups; outputs under processed/ not re-ingested
    assert [p.name for p in _bottle_dirs(out)] == [
        "evan_williams_black_label", "evan_williams_black_label_2",
        "weller_special_reserve", "weller_special_reserve_2",
    ]
    assert (out / "evan_williams_black_label_2" / "evan_williams_black_label_2.json").is_file()
    assert (out / "evan_williams_black_label" / "evan_williams_black_label.json").read_text() == first


def test_separate_output_dir_from_earlier_run_is_not_rescanned(photos):
    """A marker keeps an older output dir out of a run that writes elsewhere."""
    _run(["process", "-f", str(photos), "-o", str(photos / "processed"), "-b", "all"])

    rc, calls = _run(["process", "-f", str(photos), "-o", str(photos / "processed2"), "-b", "all"])

    assert rc == 0
    assert len(calls) == 2


def test_save_failure_is_reported_and_leaves_no_folder(photos, tmp_path):
    out = tmp_path / "out"
    real_copy2 = shutil.copy2

    def flaky_copy(src, dst, *a, **k):
        if "weller" in str(src):
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *a, **k)

    with patch("shutil.copy2", flaky_copy):
        rc, _ = _run(["process", "-f", str(photos), "-o", str(out), "-b", "all"])

    assert rc == 1  # partial failure
    assert [p.name for p in _bottle_dirs(out)] == ["evan_williams_black_label"]
    assert not list(out.glob(".*.partial"))
    reports = list(out.glob("error_report_*.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert {"bottle_id", "images", "error_message"} <= rep.keys()  # retry_command contract
    assert rep["stage"] == "save"
    assert "No space left" in rep["error_message"]
    assert len(rep["images"]) == 2


def test_analysis_failure_writes_retryable_report(photos, tmp_path):
    from lib.exceptions import VisionResponseError

    out = tmp_path / "out"

    def vision(self, image_paths, ocr_text="", model=None):
        if Path(image_paths[0]).name.startswith("weller"):
            raise VisionResponseError("model returned no content")
        return _fake_vision(self, image_paths, ocr_text, model)

    rc, _ = _run(["process", "-f", str(photos), "-o", str(out), "-b", "all"], vision=vision)

    assert rc == 1
    assert [p.name for p in _bottle_dirs(out)] == ["evan_williams_black_label"]
    reports = list(out.glob("error_report_*.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert rep["stage"] == "analysis"
    assert "VisionResponseError: model returned no content" in rep["error_message"]
    assert sorted(Path(p).name for p in rep["images"]) == ["weller_front.jpg", "weller_front_dup.jpg"]


def test_interactive_reject_saves_nothing(photos, tmp_path):
    out = tmp_path / "out"

    rc, _ = _run(
        ["process", "-f", str(photos), "-o", str(out), "-b", "all", "--interactive"],
        user_input=["R", "R"],
    )

    assert rc == 4  # every bottle rejected counts as failed
    assert _bottle_dirs(out) == []
    reports = [json.loads(p.read_text()) for p in out.glob("error_report_*.json")]
    assert sorted(r["stage"] for r in reports) == ["rejected", "rejected"]


def test_interactive_reject_is_recorded_in_the_batch_json(photos, tmp_path):
    """The batch preview/JSON are written after review, so later feedback skips the reject."""
    out = tmp_path / "out"

    rc, _ = _run(
        ["process", "-f", str(photos), "-o", str(out), "-b", "5", "--interactive"],
        user_input=["A", "R"],
    )

    assert rc == 1  # one saved, one rejected
    (batch_json,) = out.glob("batch_*.json")
    data = json.loads(batch_json.read_text())
    (report,) = [json.loads(p.read_text()) for p in out.glob("error_report_*.json")]
    assert report["stage"] == "rejected"
    assert data["rejected"] == [report["bottle_id"]]


def test_same_file_names_in_two_folders_get_distinct_ids(tmp_path):
    """IMG_1.jpg in a/ and b/ are two bottles; a failure report must list the right one."""
    root = tmp_path / "photos"
    for sub, color in (("a", (200, 10, 10)), ("b", (10, 10, 200))):
        (root / sub).mkdir(parents=True)
        _jpeg(root / sub / "IMG_1.jpg", color)
    out = tmp_path / "out"

    def vision(self, image_paths, ocr_text="", model=None):
        from lib.exceptions import VisionResponseError

        if Path(image_paths[0]).parent.name == "b":
            raise VisionResponseError("model returned no content")
        return ExtractedMetadata(source_images=[str(p) for p in image_paths], **LABELS["evan"])

    rc, _ = _run(["process", "-f", str(root), "-o", str(out), "-b", "all"], vision=vision)

    assert rc == 1
    (report,) = [json.loads(p.read_text()) for p in out.glob("error_report_*.json")]
    assert [Path(p).parent.name for p in report["images"]] == ["b"]
    assert report["bottle_id"].startswith("IMG_1_") and report["bottle_id"] != "IMG_1"


def test_every_group_processed_once_with_small_batches(photos, tmp_path):
    """No lookahead: each group is analyzed exactly once, in its own batch."""
    out = tmp_path / "out"

    rc, calls = _run(["process", "-f", str(photos), "-o", str(out), "-b", "1"])

    assert rc == 0
    assert len(calls) == 2  # one vision call per group
    assert len(_bottle_dirs(out)) == 2


def test_deprecated_lookahead_is_accepted_and_ignored(photos, tmp_path, caplog):
    out = tmp_path / "out"

    with caplog.at_level("WARNING", logger="cli.commands"):
        args = commands.parse_args(["process", "-f", str(photos), "-o", str(out), "--lookahead", "5", "--dry-run"])
        commands.pre_flight_checks(args)  # may fail later (no server); the warning comes first
    assert commands.LOOKAHEAD_DEPRECATION_WARNING in caplog.text

    rc, calls = _run(["process", "-f", str(photos), "-o", str(out), "-b", "1", "--lookahead", "5"])
    assert rc == 0
    assert len(calls) == 2


STAGG = dict(name="George T. Stagg", distillery="Buffalo Trace", msrp="99.99", proof="135.4 Proof")


def _exif_jpeg(path: Path, seconds: float, stripes: int) -> None:
    """Striped JPEG (distinct look, so no burst-retake match) taken ``seconds`` after a fixed start."""
    import datetime as dt

    from PIL import ImageDraw

    exif = PILImage.Exif()
    taken = dt.datetime(2026, 11, 7, 16, 0, 0) + dt.timedelta(seconds=seconds)
    exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
    image = PILImage.new("RGB", (64, 96), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    width = 64 // (2 * stripes)
    for k in range(stripes):
        draw.rectangle([2 * k * width, 0, (2 * k + 1) * width - 1, 95], fill=(20, 20, 20))
    image.save(path, "JPEG", exif=exif.tobytes())


@pytest.fixture
def stagg_photos(tmp_path):
    """Three photos 3 s and 6 s apart (one time group): two of Stagg 23A, then Stagg 23C."""
    d = tmp_path / "stagg"
    d.mkdir()
    _exif_jpeg(d / "IMG_0001.jpg", 0, 1)
    _exif_jpeg(d / "IMG_0002.jpg", 3, 3)
    _exif_jpeg(d / "IMG_0003.jpg", 9, 5)
    return d


def _stagg_vision(self, image_paths, ocr_text="", model=None):
    """23A on photos 1-2, 23C on photo 3; reports a mixed group when both are sent."""
    names = {Path(p).name for p in image_paths}
    batches = {"23C" if n == "IMG_0003.jpg" else "23A" for n in names}
    return ExtractedMetadata(
        source_images=[str(p) for p in image_paths], vision_model=model, vision_confidence="high",
        batch_number="23A" if "23A" in batches else "23C",
        same_bottle=(len(batches) == 1) if len(names) > 1 else None,
        warnings=[] if len(batches) == 1 else ["Vision model reports photos show different products; check grouping"],
        **STAGG,
    )


def test_mixed_group_is_split_into_separate_bottles(stagg_photos, tmp_path):
    """same_bottle=false on a 3-photo group: split at the 6 s gap, saved as two bottles."""
    out = tmp_path / "out"

    rc, calls = _run(["process", "-f", str(stagg_photos), "-o", str(out), "-b", "5"], vision=_stagg_vision)

    assert rc == 0
    assert [sorted(p.name for p in call) for call in calls] == [
        ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"],  # the group
        ["IMG_0001.jpg", "IMG_0002.jpg"],  # left half
        ["IMG_0003.jpg"],  # right half
    ]
    folders = _bottle_dirs(out)
    assert len(folders) == 2
    saved = {}
    for folder in folders:
        data = json.loads((folder / f"{folder.name}.json").read_text())
        saved[data["metadata"]["batch_number"]] = (
            sorted(p.name for p in (folder / "images").iterdir()),
            data["metadata"]["warnings"],
        )
    assert saved["23A"][0] == ["IMG_0001.jpg", "IMG_0002.jpg"]
    assert saved["23C"][0] == ["IMG_0003.jpg"]
    assert all(any("Split from a group of 3 photos" in w for w in warnings) for _, warnings in saved.values())

    # The batch preview/JSON list both bottles, with distinct IDs.
    from image_processor.preview_generator import PreviewGenerator

    (batch_json,) = out.glob("batch_*.json")
    data = json.loads(batch_json.read_text())
    ids = [
        PreviewGenerator.bottle_id_for(b, ExtractedMetadata.from_dict(m))
        for b, m in zip(data["bottles"], data["metadata"], strict=True)
    ]
    assert ids == ["IMG_0001", "IMG_0003"]
    (preview,) = out.glob("batch_*_preview.md")
    assert "Total bottles in batch: 2" in preview.read_text()
    assert "**Review:** Split from a group of 3 photos" in preview.read_text()


def test_mixed_group_false_alarm_stays_one_bottle(stagg_photos, tmp_path):
    """Halves that read as the same bottle (batch visible on one side only) are kept together."""
    out = tmp_path / "out"

    def vision(self, image_paths, ocr_text="", model=None):
        names = {Path(p).name for p in image_paths}
        return ExtractedMetadata(
            source_images=[str(p) for p in image_paths], vision_model=model,
            batch_number="23A" if names == {"IMG_0003.jpg"} else None,
            same_bottle=False if len(names) == 3 else (True if len(names) > 1 else None),  # noqa: PLR2004
            **STAGG,
        )

    rc, calls = _run(["process", "-f", str(stagg_photos), "-o", str(out), "-b", "all"], vision=vision)

    assert rc == 0
    assert len(calls) == 3
    (folder,) = _bottle_dirs(out)
    assert len(list((folder / "images").iterdir())) == 3
    warnings = json.loads((folder / f"{folder.name}.json").read_text())["metadata"]["warnings"]
    assert any("sibling bottles" in w for w in warnings)


def test_split_check_can_be_turned_off(stagg_photos, tmp_path):
    out = tmp_path / "out"

    rc, calls = _run(
        ["process", "-f", str(stagg_photos), "-o", str(out), "-b", "all", "--split-budget-ratio", "0"],
        vision=_stagg_vision,
    )

    assert rc == 0
    assert len(calls) == 1
    assert len(_bottle_dirs(out)) == 1


def test_output_dir_equal_to_input_folder_is_rejected(photos):
    args = commands.parse_args(["process", "-f", str(photos), "-o", str(photos), "--dry-run"])

    assert commands.pre_flight_checks(args) == 1
    assert not (photos / ".bourbon_output").exists()


def test_output_dir_differing_only_in_case_is_rejected(photos, tmp_path):
    """On case-insensitive filesystems (macOS default) -f Photos -o photos is the same dir."""
    if not (tmp_path / "PHOTOS").exists():
        pytest.skip("filesystem is case-sensitive")
    args = commands.parse_args(["process", "-f", str(photos), "-o", str(tmp_path / "PHOTOS"), "--dry-run"])

    assert commands.pre_flight_checks(args) == 1


def _single_batch_run(photos, out):
    """Process both bottles in one batch; return (batch json path, {name: bottle_id})."""
    from image_processor.preview_generator import PreviewGenerator

    rc, _ = _run(["process", "-f", str(photos), "-o", str(out), "-b", "5"])  # "all" writes no preview
    assert rc == 0
    (batch_json,) = out.glob("batch_*.json")
    data = json.loads(batch_json.read_text())
    ids = {
        b["name"]: PreviewGenerator.bottle_id_for(b, ExtractedMetadata.from_dict(m))
        for b, m in zip(data["bottles"], data["metadata"], strict=True)
    }
    return batch_json, ids


def _feedback(out, batch_json, corrections, tmp_path):
    corrections_file = tmp_path / "corrections.json"
    corrections_file.write_text(json.dumps(corrections))
    batch_id = batch_json.name[len("batch_"):-len(".json")]
    args = commands.parse_args(
        ["feedback", "-o", str(out), "--batch", batch_id, "--corrections", str(corrections_file)]
    )
    return commands.feedback_command(args)


def test_feedback_resaves_only_corrected_bottles(photos, tmp_path):
    out = tmp_path / "out"
    batch_json, _ = _single_batch_run(photos, out)

    rc = _feedback(out, batch_json, [{
        "field_name": "proof", "original_value": "86 Proof", "corrected_value": "90 Proof",
    }], tmp_path)

    assert rc == 0
    assert [p.name for p in _bottle_dirs(out)] == [
        "evan_williams_black_label", "evan_williams_black_label_2", "weller_special_reserve",
    ]
    resaved = json.loads((out / "evan_williams_black_label_2" / "evan_williams_black_label_2.json").read_text())
    assert resaved["proof"] == "90 Proof"
    assert [c["corrected_value"] for c in resaved["corrections"]] == ["90 Proof"]


def test_feedback_skips_bottles_rejected_in_the_batch(photos, tmp_path):
    out = tmp_path / "out"
    batch_json, ids = _single_batch_run(photos, out)
    data = json.loads(batch_json.read_text())
    data["rejected"] = [ids["Weller Special Reserve"]]
    batch_json.write_text(json.dumps(data))

    rc = _feedback(out, batch_json, [
        {"field_name": "proof", "original_value": "86 Proof", "corrected_value": "80 Proof"},
        {"field_name": "proof", "original_value": "90 Proof", "corrected_value": "107 Proof"},
    ], tmp_path)

    assert rc == 0
    assert [p.name for p in _bottle_dirs(out)] == [
        "evan_williams_black_label", "evan_williams_black_label_2", "weller_special_reserve",
    ]
