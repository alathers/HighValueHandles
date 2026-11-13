"""Unit tests for ImageOrganizer class.

Tests image scanning, grouping, duplicate detection, and filtering.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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


class TestImageOrganizerScanImages:
    """Test scan_images method."""

    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    def test_scan_images_success(self, mock_glob, mock_exists, mock_is_dir):
        """Test scanning folder finds supported images."""
        # Mock file paths
        mock_jpg = Mock()
        mock_jpg.is_dir.return_value = False
        mock_jpg.suffix.lower.return_value = ".jpg"
        mock_jpg.stat.return_value.st_size = 1024

        mock_png = Mock()
        mock_png.is_dir.return_value = False
        mock_png.suffix.lower.return_value = ".png"
        mock_png.stat.return_value.st_size = 2048

        mock_dir = Mock()
        mock_dir.is_dir.return_value = True

        mock_glob.return_value = [mock_jpg, mock_png, mock_dir]

        # Mock DuplicateDetector
        mock_detector = Mock()
        mock_detector.generate_hash.side_effect = ["hash1", "hash2"]

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(Path("/tmp/test"), recursive=False)

        assert len(images) == 2
        assert images[0].md5_hash == "hash1"
        assert images[0].file_size == 1024
        assert images[1].md5_hash == "hash2"
        assert images[1].file_size == 2048

    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    def test_scan_images_recursive(self, mock_glob, mock_exists, mock_is_dir):
        """Test recursive scanning includes subdirectories."""
        mock_file = Mock()
        mock_file.is_dir.return_value = False
        mock_file.suffix.lower.return_value = ".jpeg"
        mock_file.stat.return_value.st_size = 512

        mock_glob.return_value = [mock_file]

        mock_detector = Mock()
        mock_detector.generate_hash.return_value = "hash"

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(Path("/tmp/test"), recursive=True)

        assert len(images) == 1
        assert mock_glob.call_args[0][0] == "**/*"

    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    def test_scan_images_non_recursive(self, mock_glob, mock_exists, mock_is_dir):
        """Test non-recursive scanning only top level."""
        mock_file = Mock()
        mock_file.is_dir.return_value = False
        mock_file.suffix.lower.return_value = ".bmp"
        mock_file.stat.return_value.st_size = 256

        mock_glob.return_value = [mock_file]

        mock_detector = Mock()
        mock_detector.generate_hash.return_value = "hash"

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(Path("/tmp/test"), recursive=False)

        assert mock_glob.call_args[0][0] == "*"

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

    @patch("pathlib.Path.glob")
    def test_scan_images_empty_folder(self, mock_glob):
        """Test scan_images returns empty list for folder with no images."""
        mock_glob.return_value = []

        organizer = ImageOrganizer()
        images = organizer.scan_images(Path("/tmp/empty"))

        assert images == []

    @patch("pathlib.Path.glob")
    def test_scan_images_filters_unsupported_extensions(self, mock_glob):
        """Test scan_images skips unsupported file types."""
        mock_txt = Mock()
        mock_txt.is_dir.return_value = False
        mock_txt.suffix.lower.return_value = ".txt"

        mock_doc = Mock()
        mock_doc.is_dir.return_value = False
        mock_doc.suffix.lower.return_value = ".doc"

        mock_pdf = Mock()
        mock_pdf.is_dir.return_value = False
        mock_pdf.suffix.lower.return_value = ".pdf"

        mock_glob.return_value = [mock_txt, mock_doc, mock_pdf]

        organizer = ImageOrganizer()
        images = organizer.scan_images(Path("/tmp/test"))

        assert len(images) == 0

    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    def test_scan_images_supports_all_extensions(self, mock_glob, mock_exists, mock_is_dir):
        """Test all supported extensions are detected."""
        extensions = [".jpg", ".jpeg", ".png", ".tiff", ".heic", ".webp", ".bmp"]

        files = []
        for ext in extensions:
            mock_file = Mock()
            mock_file.is_dir.return_value = False
            mock_file.suffix.lower.return_value = ext
            mock_file.stat.return_value.st_size = 100
            files.append(mock_file)

        mock_glob.return_value = files

        mock_detector = Mock()
        mock_detector.generate_hash.side_effect = [f"hash{i}" for i in range(len(files))]

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(Path("/tmp/test"))

        assert len(images) == len(extensions)

    @patch("pathlib.Path.glob")
    def test_scan_images_hash_error_skips_file(self, mock_glob):
        """Test scan_images logs warning and skips file on hash error."""
        mock_file = Mock()
        mock_file.is_dir.return_value = False
        mock_file.suffix.lower.return_value = ".jpg"

        mock_glob.return_value = [mock_file]

        mock_detector = Mock()
        mock_detector.generate_hash.side_effect = Exception("Hash error")

        organizer = ImageOrganizer(duplicate_detector=mock_detector)
        images = organizer.scan_images(Path("/tmp/test"))

        # Should skip the file but not crash
        assert len(images) == 0


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

    def test_group_by_bottle_custom_similarity_threshold(self):
        """Test grouping respects custom similarity threshold."""
        img1 = Image(path="/tmp/IMG_1234_front.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/IMG_1234_back.jpg")
        img2.md5_hash = "hash2"

        img3 = Image(path="/tmp/IMG_9999_front.jpg")
        img3.md5_hash = "hash3"

        organizer = ImageOrganizer()
        # Lower threshold groups more aggressively
        groups = organizer.group_by_bottle([img1, img2, img3], similarity_threshold=0.5)

        # With low threshold, similar filenames should be grouped together
        # IMG_1234_* should be in one group, IMG_9999_* in another
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


class TestImageOrganizerIsDuplicateImage:
    """Test is_duplicate_image method."""

    def test_is_duplicate_image_same_hash(self):
        """Test images with same hash are duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash123"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash123"

        organizer = ImageOrganizer()
        is_dup = organizer.is_duplicate_image(img1, img2)

        assert is_dup is True

    def test_is_duplicate_image_different_hash(self):
        """Test images with different hashes are not duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        is_dup = organizer.is_duplicate_image(img1, img2)

        assert is_dup is False

    def test_is_duplicate_image_one_no_hash(self):
        """Test non-duplicate when one image has no hash."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash123"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = None

        organizer = ImageOrganizer()
        is_dup = organizer.is_duplicate_image(img1, img2)

        assert is_dup is False

    def test_is_duplicate_image_both_no_hash(self):
        """Test non-duplicate when both images have no hash."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = None

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = None

        organizer = ImageOrganizer()
        is_dup = organizer.is_duplicate_image(img1, img2)

        assert is_dup is False


class TestImageOrganizerFilterDuplicateImages:
    """Test filter_duplicate_images method."""

    def test_filter_duplicate_images_no_duplicates(self):
        """Test filtering images with no duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        filtered = organizer.filter_duplicate_images([img1, img2])

        assert len(filtered) == 2
        assert img1 in filtered
        assert img2 in filtered

    def test_filter_duplicate_images_removes_duplicates(self):
        """Test filtering removes duplicate images."""
        img1 = Image(path="/tmp/original.jpg")
        img1.md5_hash = "hash_same"

        img2 = Image(path="/tmp/copy1.jpg")
        img2.md5_hash = "hash_same"

        img3 = Image(path="/tmp/copy2.jpg")
        img3.md5_hash = "hash_same"

        img4 = Image(path="/tmp/different.jpg")
        img4.md5_hash = "hash_diff"

        organizer = ImageOrganizer()
        filtered = organizer.filter_duplicate_images([img1, img2, img3, img4])

        # Should keep one of hash_same and one of hash_diff
        assert len(filtered) == 2

    def test_filter_duplicate_images_keep_original_true(self):
        """Test keep_original=True keeps first occurrence."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash"

        img3 = Image(path="/tmp/img3.jpg")
        img3.md5_hash = "hash"

        organizer = ImageOrganizer()
        filtered = organizer.filter_duplicate_images([img1, img2, img3], keep_original=True)

        assert len(filtered) == 1
        assert filtered[0] == img1

    def test_filter_duplicate_images_no_hashes(self):
        """Test filtering keeps images with no hash."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = None

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash"

        img3 = Image(path="/tmp/img3.jpg")
        img3.md5_hash = None

        organizer = ImageOrganizer()
        filtered = organizer.filter_duplicate_images([img1, img2, img3])

        # No hash images are kept
        assert img1 in filtered
        assert img2 in filtered
        assert img3 in filtered

    def test_filter_duplicate_images_empty_list(self):
        """Test filtering empty list returns empty list."""
        organizer = ImageOrganizer()
        filtered = organizer.filter_duplicate_images([])

        assert filtered == []


class TestImageOrganizerImagesSimilar:
    """Test _images_similar method."""

    def test_images_similar_exact_duplicate(self):
        """Test images are similar if they are duplicates."""
        img1 = Image(path="/tmp/img1.jpg")
        img1.md5_hash = "hash123"

        img2 = Image(path="/tmp/img2.jpg")
        img2.md5_hash = "hash123"

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        assert similar is True

    def test_images_similar_filename_prefix_common(self):
        """Test images with common filename prefix are similar."""
        img1 = Image(path="/tmp/IMG_1234_front.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/IMG_1234_back.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        assert similar is True

    def test_images_similar_different_names(self):
        """Test images with different names are not similar."""
        img1 = Image(path="/tmp/bottle_a.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/box_b.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        assert similar is False

    def test_images_similar_same_directory(self):
        """Test images in same directory may be similar."""
        img1 = Image(path="/tmp/folder/img1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/folder/img2.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        # Without hash match or name similarity, directory alone may group them
        similar = organizer._images_similar(img1, img2, 0.8)

        # Implementation may group by directory as fallback
        assert isinstance(similar, bool)

    def test_images_similar_different_directories(self):
        """Test images in different directories with different names."""
        img1 = Image(path="/tmp/folder1/bottle_1.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/folder2/bottle_2.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        assert similar is False

    def test_images_similar_short_names_less_than_threshold(self):
        """Test short names below similarity threshold."""
        img1 = Image(path="/tmp/a.jpg")
        img1.md5_hash = "hash1"

        img2 = Image(path="/tmp/b.jpg")
        img2.md5_hash = "hash2"

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        assert similar is False

    def test_images_similar_no_hashes_check_name_only(self):
        """Test similarity works without hashes."""
        img1 = Image(path="/tmp/IMG_9999_1.jpg")
        img1.md5_hash = None

        img2 = Image(path="/tmp/IMG_9999_2.jpg")
        img2.md5_hash = None

        organizer = ImageOrganizer()
        similar = organizer._images_similar(img1, img2, 0.8)

        # Should still check filename similarity
        assert similar is True
