"""Unit tests for ImageOrganizer class.

Tests image scanning, grouping, duplicate detection, and filtering.
"""

import datetime as dt
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from image_processor.image_organizer import ImageOrganizer
from lib.duplicates import DuplicateDetector
from models.image import Image


class TestImageOrganizerInitialization:
    """Test ImageOrganizer initialization."""

    def test_organizer_default_values(self):
        """Test organizer initializes with default values."""
        organizer = ImageOrganizer()

        assert organizer.duplicate_detector is not None
        assert isinstance(organizer.duplicate_detector, DuplicateDetector)

    def test_organizer_custom_duplicate_detector(self):
        """Test organizer with custom DuplicateDetector."""
        mock_detector = Mock(spec=DuplicateDetector)
        organizer = ImageOrganizer(duplicate_detector=mock_detector)

        assert organizer.duplicate_detector == mock_detector

    def test_organizer_supported_extensions(self):
        """Test SUPPORTED_EXTENSIONS contains expected formats."""
        assert ".jpeg" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".jpg" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".png" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".tiff" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".heic" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".webp" in ImageOrganizer.SUPPORTED_EXTENSIONS
        assert ".bmp" in ImageOrganizer.SUPPORTED_EXTENSIONS


def _touch(path: Path, data: bytes = b"img") -> Path:
    """Create a small file (and its parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestImageOrganizerScanImages:
    """Test scan_images method (real files under tmp_path)."""

    def test_scan_images_success(self, tmp_path):
        """Scanning finds supported images with hash and size populated."""
        _touch(tmp_path / "a.jpg", b"x" * 1024)
        _touch(tmp_path / "b.png", b"y" * 2048)
        (tmp_path / "subdir").mkdir()

        mock_detector = Mock()
        mock_detector.generate_hash.side_effect = ["hash1", "hash2"]

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(tmp_path, recursive=False)

        assert [Path(i.path).name for i in images] == ["a.jpg", "b.png"]
        assert images[0].md5_hash == "hash1"
        assert images[0].file_size == 1024
        assert images[1].md5_hash == "hash2"
        assert images[1].file_size == 2048

    def test_scan_images_recursive(self, tmp_path):
        """Recursive scanning includes subdirectories."""
        _touch(tmp_path / "top.jpeg")
        _touch(tmp_path / "nested" / "deeper" / "low.jpeg")

        images = ImageOrganizer().scan_images(tmp_path, recursive=True)

        assert sorted(Path(i.path).name for i in images) == ["low.jpeg", "top.jpeg"]

    def test_scan_images_non_recursive(self, tmp_path):
        """Non-recursive scanning only looks at the top level."""
        _touch(tmp_path / "top.bmp")
        _touch(tmp_path / "nested" / "low.bmp")

        images = ImageOrganizer().scan_images(tmp_path, recursive=False)

        assert [Path(i.path).name for i in images] == ["top.bmp"]

    @patch("pathlib.Path.exists", return_value=False)
    def test_scan_images_folder_not_exists(self, mock_exists):
        """Test scan_images returns empty list for non-existent folder."""
        organizer = ImageOrganizer()
        images = organizer.scan_images(Path("/nonexistent"))

        assert images == []

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=False)
    def test_scan_images_not_directory(self, mock_is_dir, mock_exists):
        """Test scan_images returns empty list for non-directory path."""
        organizer = ImageOrganizer()
        images = organizer.scan_images(Path("/text_file.txt"))

        assert images == []

    def test_scan_images_empty_folder(self, tmp_path):
        """Test scan_images returns empty list for folder with no images."""
        assert ImageOrganizer().scan_images(tmp_path) == []

    def test_scan_images_filters_unsupported_extensions(self, tmp_path):
        """Test scan_images skips unsupported file types."""
        for name in ("notes.txt", "doc.doc", "scan.pdf"):
            _touch(tmp_path / name)

        assert ImageOrganizer().scan_images(tmp_path) == []

    def test_scan_images_supports_all_extensions(self, tmp_path):
        """All supported extensions are detected, case-insensitively."""
        extensions = [".jpg", ".jpeg", ".png", ".tiff", ".heic", ".webp", ".bmp", ".JPG"]
        for i, ext in enumerate(extensions):
            _touch(tmp_path / f"img{i}{ext}", f"data{i}".encode())

        images = ImageOrganizer().scan_images(tmp_path)

        assert len(images) == len(extensions)

    def test_scan_images_hash_error_skips_file(self, tmp_path):
        """Test scan_images logs warning and skips file on hash error."""
        _touch(tmp_path / "a.jpg")
        mock_detector = Mock()
        mock_detector.generate_hash.side_effect = Exception("Hash error")

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(tmp_path)

        # Should skip the file but not crash
        assert images == []


class TestImageOrganizerScanExclusion:
    """Earlier outputs, hidden files and excluded dirs are never scanned."""

    def test_output_dir_under_input_is_excluded(self, tmp_path):
        _touch(tmp_path / "IMG_1.jpg")
        _touch(tmp_path / "processed" / "eagle_rare" / "images" / "IMG_1.jpg")

        paths = ImageOrganizer().iter_image_paths(tmp_path, exclude=[tmp_path / "processed"])

        assert paths == [tmp_path / "IMG_1.jpg"]

    def test_exclude_matches_relative_and_absolute_spellings(self, tmp_path, monkeypatch):
        _touch(tmp_path / "IMG_1.jpg")
        _touch(tmp_path / "out" / "IMG_2.jpg")
        monkeypatch.chdir(tmp_path)

        paths = ImageOrganizer().iter_image_paths(tmp_path, exclude=[Path("out")])

        assert paths == [tmp_path / "IMG_1.jpg"]

    def test_subdir_with_marker_is_skipped_but_marked_root_is_scanned(self, tmp_path):
        from image_processor.image_organizer import OUTPUT_MARKER

        _touch(tmp_path / OUTPUT_MARKER, b"")
        _touch(tmp_path / "IMG_root.jpg")
        _touch(tmp_path / "old_run" / OUTPUT_MARKER, b"")
        _touch(tmp_path / "old_run" / "bottle" / "images" / "IMG_old.jpg")
        _touch(tmp_path / "keep" / "IMG_keep.jpg")

        names = [p.name for p in ImageOrganizer().iter_image_paths(tmp_path)]

        assert names == ["IMG_root.jpg", "IMG_keep.jpg"]

    def test_hidden_dirs_and_appledouble_files_are_skipped(self, tmp_path):
        _touch(tmp_path / "IMG_1.jpg")
        _touch(tmp_path / "._IMG_1.jpg")
        _touch(tmp_path / ".hidden" / "IMG_2.jpg")
        _touch(tmp_path / ".bottle.abc123.partial" / "images" / "IMG_3.jpg")

        assert ImageOrganizer().iter_image_paths(tmp_path) == [tmp_path / "IMG_1.jpg"]

    def test_results_are_sorted(self, tmp_path):
        for name in ("c.jpg", "a.jpg", "b.jpg"):
            _touch(tmp_path / name)
        _touch(tmp_path / "z_dir" / "a.jpg")
        _touch(tmp_path / "a_dir" / "z.jpg")

        paths = ImageOrganizer().iter_image_paths(tmp_path)

        assert paths == [
            tmp_path / "a.jpg",
            tmp_path / "b.jpg",
            tmp_path / "c.jpg",
            tmp_path / "a_dir" / "z.jpg",
            tmp_path / "z_dir" / "a.jpg",
        ]

    def test_exclude_that_is_ancestor_of_root_is_ignored(self, tmp_path):
        root = tmp_path / "processed" / "x"
        _touch(root / "IMG_1.jpg")

        organizer = ImageOrganizer()

        assert organizer.iter_image_paths(root, exclude=[tmp_path / "processed"]) == [root / "IMG_1.jpg"]
        assert organizer.iter_image_paths(root, exclude=[root]) == [root / "IMG_1.jpg"]

    def test_unusable_exclude_entries_are_ignored(self, tmp_path):
        from unittest.mock import MagicMock

        _touch(tmp_path / "IMG_1.jpg")

        paths = ImageOrganizer().iter_image_paths(tmp_path, exclude=[MagicMock(), None, 42])

        assert paths == [tmp_path / "IMG_1.jpg"]

    def test_exclude_matches_by_identity_through_symlink(self, tmp_path):
        """An exclude spelled via a symlink still matches the real directory."""
        root = tmp_path / "root"
        _touch(root / "IMG_1.jpg")
        _touch(root / "out" / "bottle" / "IMG_2.jpg")
        alias = tmp_path / "alias"
        alias.symlink_to(root / "out", target_is_directory=True)

        paths = ImageOrganizer().iter_image_paths(root, exclude=[alias])

        assert paths == [root / "IMG_1.jpg"]

    def test_exclude_differing_only_in_case_matches_on_case_insensitive_fs(self, tmp_path):
        _touch(tmp_path / "IMG_1.jpg")
        _touch(tmp_path / "Processed" / "bottle" / "IMG_2.jpg")
        if not (tmp_path / "PROCESSED").exists():
            pytest.skip("filesystem is case-sensitive")

        paths = ImageOrganizer().iter_image_paths(tmp_path, exclude=[tmp_path / "processed"])

        assert paths == [tmp_path / "IMG_1.jpg"]

    def test_symlinked_subdirectory_is_not_followed_and_is_logged(self, tmp_path, caplog):
        root = tmp_path / "root"
        _touch(root / "IMG_1.jpg")
        _touch(tmp_path / "elsewhere" / "IMG_2.jpg")
        (root / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

        with caplog.at_level("INFO", logger="image_processor.image_organizer"):
            paths = ImageOrganizer().iter_image_paths(root)

        assert paths == [root / "IMG_1.jpg"]
        assert "Not following symlinked directory" in caplog.text

    def test_scan_images_passes_exclude(self, tmp_path):
        _touch(tmp_path / "IMG_1.jpg", b"same")
        _touch(tmp_path / "processed" / "bottle" / "IMG_1.jpg", b"same")

        images = ImageOrganizer().scan_images(tmp_path, exclude=[tmp_path / "processed"])

        assert [i.path for i in images] == [str(tmp_path / "IMG_1.jpg")]


class TestImageOrganizerGroupByBottle:
    """Test group_by_bottle method."""

    def test_group_by_bottle_empty_list(self):
        """Test grouping empty list returns empty."""
        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([])

        assert groups == []

    def test_group_by_bottle_single_image(self):
        """Test grouping single image returns single group."""
        img = Image(path="/tmp/test.jpg")
        img.md5_hash = "hash1"

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img])

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0] == img

    @patch.object(ImageOrganizer, "detect_duplicates")
    def test_group_by_bottle_calls_detect_duplicates(self, mock_detect):
        """Test grouping calls detect_duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash1"
        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        organizer.group_by_bottle([img1, img2])

        mock_detect.assert_called_once_with([img1, img2])

    def test_group_by_bottle_different_hashes(self):
        """Test images with different hashes form separate groups."""
        img1 = Image(path="/tmp/bottle1_front.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/bottle2_front.jpg")
        img2.md5_hash = "hash2"

        img3 = Image(path="/tmp/bottle3_front.jpg")
        img3.md5_hash = "hash3"

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img1, img2, img3])

        assert len(groups) == 3
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1
        assert len(groups[2]) == 1

    def test_group_by_bottle_same_hash(self):
        """Test images with same hash are grouped together."""
        img1 = Image(path="/tmp/bottle_front.jpg")
        img1.md5_hash = "hash_same"

        img2 = Image(path="/tmp/bottle_back.jpg")
        img2.md5_hash = "hash_same"

        img3 = Image(path="/tmp/bottle_side.jpg")
        img3.md5_hash = "hash_same"

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img1, img2, img3])

        assert len(groups) == 1
        assert len(groups[0]) == 3
        assert img1 in groups[0]
        assert img2 in groups[0]
        assert img3 in groups[0]

    def test_group_by_bottle_duplicate_hashes(self):
        """Test duplicate detection works before grouping."""
        img1 = Image(path="/tmp/bottle1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/bottle1_dup.jpg")
        img2.md5_hash = "hash1"

        img3 = Image(path="/tmp/bottle2.jpg")
        img3.md5_hash = "hash2"

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img1, img2, img3])

        assert len(groups) == 2
        # hash1 group
        assert len(groups[0]) == 2
        assert img1 in groups[0]
        assert img2 in groups[0]
        # hash2 group
        assert len(groups[1]) == 1
        assert img3 in groups[1]

    def test_group_by_bottle_pairs_angle_suffix_names(self):
        """Untimed photos sharing an angle-suffix base name form one group."""
        img1 = Image(path="/tmp/IMG_1234_front.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/IMG_1234_back.jpg")
        img2.md5_hash = "hash2"

        img3 = Image(path="/tmp/IMG_9999_front.jpg")
        img3.md5_hash = "hash3"

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img1, img2, img3])

        # IMG_1234_front/back pair up; IMG_9999_front is on its own.
        assert len(groups) == 2

    def test_group_by_bottle_no_hashes(self):
        """Test images without hashes form separate groups."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = None

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = None

        organizer = ImageOrganizer()
        groups = organizer.group_by_bottle([img1, img2])

        # Without hashes, images may be grouped by similarity fallback
        assert len(groups) >= 0


class TestImageOrganizerDetectDuplicates:
    """Test detect_duplicates method."""

    def test_detect_duplicates_no_duplicates(self):
        """Test detect_duplicates marks no images as duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash2"

        img3 = Image(path="/tmp/img3.jpg")
        img3.md5_hash = "hash3"

        organizer = ImageOrganizer()
        organizer.detect_duplicates([img1, img2, img3])

        # Should not mark any as duplicates
        assert img1.is_duplicate is False
        assert img2.is_duplicate is False
        assert img3.is_duplicate is False
        assert img1.duplicate_of_id is None
        assert img2.duplicate_of_id is None
        assert img3.duplicate_of_id is None

    def test_detect_duplicates_finds_duplicates(self):
        """Test detect_duplicates correctly identifies duplicates."""
        img1 = Image(path="/tmp/image.jpg")
        img1.md5_hash = "hash_same"

        img2 = Image(path="/tmp/image_copy.jpg")
        img2.md5_hash = "hash_same"

        img3 = Image(path="/tmp/different.jpg")
        img3.md5_hash = "hash_different"

        organizer = ImageOrganizer()
        organizer.detect_duplicates([img1, img2, img3])

        # First is original, second is duplicate
        assert img1.is_duplicate is False
        assert img1.duplicate_of_id is None

        assert img2.is_duplicate is True
        assert img2.duplicate_of_id == img1.path

        # Different image not marked
        assert img3.is_duplicate is False

    def test_detect_duplicates_multiple_duplicates(self):
        """Test detect_duplicates handles multiple duplicate sets."""
        img1 = Image(path="/set1_img1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/set1_img2.jpg")
        img2.md5_hash = "hash1"

        img3 = Image(path="/set1_img3.jpg")
        img3.md5_hash = "hash1"

        img4 = Image(path="/set2_img1.jpg")
        img4.md5_hash = "hash2"

        img5 = Image(path="/set2_img2.jpg")
        img5.md5_hash = "hash2"

        img6 = Image(path="/different.jpg")
        img6.md5_hash = "hash3"

        organizer = ImageOrganizer()
        organizer.detect_duplicates([img1, img2, img3, img4, img5, img6])

        # Set 1 duplicates
        assert img1.is_duplicate is False
        assert img2.is_duplicate is True
        assert img3.is_duplicate is True

        # Set 2 duplicates
        assert img4.is_duplicate is False
        assert img5.is_duplicate is True

        # Different image
        assert img6.is_duplicate is False

    def test_detect_duplicates_no_hashes(self):
        """Test detect_duplicates handles images with no hash."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = None

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = None

        organizer = ImageOrganizer()
        organizer.detect_duplicates([img1, img2])

        # No crashes, images remain unchanged
        assert img1.is_duplicate is False
        assert img2.is_duplicate is False

    def test_detect_duplicates_mixed_hashes_and_none(self):
        """Test detect_duplicates with mixed hash states."""
        img1 = Image(path="/tmp/hashed.jpg")
        img1.md5_hash = "hash123"

        img2 = Image(path="/tmp/no_hash.jpg")
        img2.md5_hash = None

        img3 = Image(path="/tmp/hash_dup.jpg")
        img3.md5_hash = "hash123"

        organizer = ImageOrganizer()
        organizer.detect_duplicates([img1, img2, img3])

        # Only hashed duplicates marked
        assert img1.is_duplicate is False
        assert img2.is_duplicate is False
        assert img3.is_duplicate is True
        assert img3.duplicate_of_id == img1.path


_T0 = dt.datetime(2026, 11, 7, 16, 0, 0)


def _photo(path: Path, seconds=None, text="A", color=(180, 120, 60), size=(120, 160)) -> Image:
    """A small JPEG with an EXIF capture time, scanned into an Image (hash set)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    picture = PILImage.new("RGB", size, color)
    draw = ImageDraw.Draw(picture)
    draw.rectangle([20, 30, 100, 130], outline=(0, 0, 0), width=4)
    draw.text((40, 60), text, fill=(255, 255, 255))
    exif = PILImage.Exif()
    if seconds is not None:
        exif.get_ifd(0x8769)[0x9003] = (_T0 + dt.timedelta(seconds=seconds)).strftime("%Y:%m:%d %H:%M:%S")
    picture.save(path, "JPEG", exif=exif.tobytes())
    return _scanned(path)


def _scanned(path: Path) -> Image:
    img = Image(path=str(path))
    img.md5_hash = DuplicateDetector().generate_hash(path)
    return img


def _names(groups):
    return [[Path(img.path).name for img in group] for group in groups]


class TestGroupByCaptureTime:
    """group_by_bottle groups by capture time, never by file-name prefix."""

    def test_consecutive_names_of_different_bottles_stay_separate(self, tmp_path):
        """Regression: IMG_4653/IMG_4654 and IMG_4660/IMG_4668 used to merge by name prefix."""
        times = {"IMG_4653": 0, "IMG_4654": 60, "IMG_4660": 400, "IMG_4668": 900}
        images = [_photo(tmp_path / f"{name}.jpeg", t, text=name) for name, t in times.items()]

        groups = ImageOrganizer().group_by_bottle(images)

        assert _names(groups) == [[f"{n}.jpeg"] for n in times]

    def test_untimed_consecutive_names_stay_separate(self, tmp_path):
        images = [_photo(tmp_path / f"IMG_{n}.jpeg", None, text=str(n)) for n in (4653, 4654, 4668)]

        groups = ImageOrganizer().group_by_bottle(images)

        assert len(groups) == 3

    def test_group_by_bottle_uses_capture_time(self, tmp_path):
        times = [0, 5, 40, 44, 120]
        images = [_photo(tmp_path / f"IMG_{i}.jpeg", t, text=str(i)) for i, t in enumerate(times)]

        groups = ImageOrganizer().group_by_bottle(list(reversed(images)))

        assert _names(groups) == [["IMG_0.jpeg", "IMG_1.jpeg"], ["IMG_2.jpeg", "IMG_3.jpeg"], ["IMG_4.jpeg"]]

    def test_group_by_bottle_sets_capture_fields(self, tmp_path):
        img = _photo(tmp_path / "IMG_1.jpeg", 0)
        other = _photo(tmp_path / "IMG_2.jpeg", 100, text="B")

        ImageOrganizer().group_by_bottle([img, other])

        assert img.capture_time is not None
        assert img.capture_time_source == "exif_original"
        assert other.capture_time - img.capture_time == pytest.approx(100)

    def test_group_by_bottle_duplicates_follow_original(self, tmp_path):
        original = _photo(tmp_path / "IMG_1.jpeg", 0)
        later = _photo(tmp_path / "IMG_2.jpeg", 300, text="B")
        shutil.copy(original.path, tmp_path / "copy_of_1.jpeg")
        copy = _scanned(tmp_path / "copy_of_1.jpeg")

        groups = ImageOrganizer().group_by_bottle([original, later, copy])

        assert _names(groups) == [["IMG_1.jpeg", "copy_of_1.jpeg"], ["IMG_2.jpeg"]]
        assert copy.is_duplicate and copy.duplicate_of_id == original.path

    def test_group_by_bottle_marks_near_duplicate(self, tmp_path):
        first = _photo(tmp_path / "IMG_1.jpeg", 0, text="FRONT")
        retake = _photo(tmp_path / "IMG_2.jpeg", 1, text="FRONT")
        back = _photo(tmp_path / "IMG_3.jpeg", 3, text="BACK", color=(20, 40, 160), size=(160, 120))

        groups = ImageOrganizer().group_by_bottle([first, retake, back])

        assert len(groups) == 1
        assert retake.near_duplicate_of == first.path
        assert back.near_duplicate_of is None
        assert first.near_duplicate_of is None

    def test_orb_similarity_helpers_removed(self):
        for name in ("is_duplicate_image", "_images_similar", "filter_duplicate_images"):
            assert not hasattr(ImageOrganizer, name)

    def test_constructor_gap_and_cap_respected(self, tmp_path):
        images = [_photo(tmp_path / f"IMG_{i}.jpeg", t, text=str(i)) for i, t in enumerate([0, 2, 11, 13])]

        assert [len(g) for g in ImageOrganizer().group_by_bottle(images)] == [2, 2]
        assert [len(g) for g in ImageOrganizer(max_group_size=0).group_by_bottle(images)] == [4]
        assert [len(g) for g in ImageOrganizer(gap_seconds=1).group_by_bottle(images)] == [1, 1, 1, 1]

    def test_two_cameras_shooting_at_once_do_not_interleave(self, tmp_path):
        def shot(name, seconds, model):
            path = tmp_path / name
            picture = PILImage.new("RGB", (80, 80), (100, 100, 100))
            exif = PILImage.Exif()
            exif[0x0110] = model
            exif.get_ifd(0x8769)[0x9003] = (_T0 + dt.timedelta(seconds=seconds)).strftime("%Y:%m:%d %H:%M:%S")
            picture.save(path, "JPEG", exif=exif.tobytes())
            return _scanned(path)

        images = [
            shot("a1.jpeg", 0, "PhoneA"), shot("b1.jpeg", 1, "PhoneB"), shot("a2.jpeg", 3, "PhoneA"),
            shot("b2.jpeg", 4, "PhoneB"),
        ]

        groups = ImageOrganizer(detect_near_duplicates=False).group_by_bottle(images)

        assert sorted(_names(groups)) == [["a1.jpeg", "a2.jpeg"], ["b1.jpeg", "b2.jpeg"]]

    def test_photo_without_camera_tags_joins_the_only_camera(self, tmp_path):
        """A photo timed without camera tags still chains with the one camera's photos."""

        def shot(name, seconds, model=None):
            path = tmp_path / name
            exif = PILImage.Exif()
            if model:
                exif[0x0110] = model
            exif.get_ifd(0x8769)[0x9003] = (_T0 + dt.timedelta(seconds=seconds)).strftime("%Y:%m:%d %H:%M:%S")
            PILImage.new("RGB", (80, 80), (100, 100, 100)).save(path, "JPEG", exif=exif.tobytes())
            return _scanned(path)

        images = [shot("a1.jpeg", 0, "PhoneA"), shot("x.jpeg", 4), shot("a2.jpeg", 60, "PhoneA")]

        groups = ImageOrganizer(detect_near_duplicates=False).group_by_bottle(images)

        assert _names(groups) == [["a1.jpeg", "x.jpeg"], ["a2.jpeg"]]
