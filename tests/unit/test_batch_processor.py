"""Unit tests for BatchProcessor class.

Tests batch slicing, lookahead algorithm, vision similarity scoring,
batch processing, and user feedback integration.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from image_processor.batch_processor import BatchProcessor, BatchResult
from models.bottle import Correction, ExtractedMetadata
from models.image import Image


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization."""

    def test_processor_default_values(self):
        """Test processor initializes with default values."""
        processor = BatchProcessor()

        assert processor.batch_size == 5
        assert processor.lookahead == 3
        assert processor.similarity_threshold == 0.7
        assert processor.bottle_analyzer is not None

    def test_processor_custom_values(self):
        """Test processor initializes with custom values."""
        with patch("image_processor.batch_processor.BottleAnalyzer") as mock_analyzer:
            mock_analyzer.return_value = Mock()
            processor = BatchProcessor(
                batch_size=10,
                lookahead=5,
                similarity_threshold=0.85,
            )

            assert processor.batch_size == 10
            assert processor.lookahead == 5
            assert processor.similarity_threshold == 0.85

    def test_processor_batch_size_all(self):
        """Test processor accepts 'all' string for batch_size."""
        processor = BatchProcessor(batch_size="all")

        assert processor.batch_size == "all"

    def test_processor_custom_bottle_analyzer(self):
        """Test processor accepts custom BottleAnalyzer."""
        custom_analyzer = Mock()
        processor = BatchProcessor(bottle_analyzer=custom_analyzer)

        assert processor.bottle_analyzer == custom_analyzer

    def test_processor_batch_size_converted_to_int(self):
        """Test batch_size is converted to int when numeric."""
        processor = BatchProcessor(batch_size="7")

        assert processor.batch_size == 7
        assert isinstance(processor.batch_size, int)

    def test_processor_batch_size_zero_allowed(self):
        """Test batch_size=0 is allowed and treated as 'all' mode."""
        processor = BatchProcessor(batch_size=0)

        assert processor.batch_size == 0


class TestBatchProcessorSliceIntoBatches:
    """Test batch slicing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()

    def test_slice_single_group(self):
        """Test slicing single image group into one batch."""
        image_groups = [
            [Image(path=Path("/test/image1.jpg"))],
        ]
        batches = self.processor.slice_into_batches(image_groups)

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_slice_multiple_groups(self):
        """Test slicing multiple image groups into batches."""
        image_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 11)
        ]
        batches = self.processor.slice_into_batches(image_groups)

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_slice_with_remainder(self):
        """Test slicing handles remainder groups."""
        image_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 8)
        ]
        batches = self.processor.slice_into_batches(image_groups)

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 2

    def test_slice_empty_list(self):
        """Test slicing empty list returns empty list."""
        batches = self.processor.slice_into_batches([])

        assert batches == []

    def test_slice_batch_size_all(self):
        """Test slicing with batch_size='all' returns single batch."""
        processor = BatchProcessor(batch_size="all")
        image_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 11)
        ]
        batches = processor.slice_into_batches(image_groups)

        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_slice_batch_size_zero(self):
        """Test slicing with batch_size=0 returns single batch."""
        processor = BatchProcessor(batch_size=0)
        image_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 11)
        ]
        batches = processor.slice_into_batches(image_groups)

        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_slice_batch_size_one(self):
        """Test slicing with batch_size=1 creates many batches."""
        processor = BatchProcessor(batch_size=1)
        image_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 4)
        ]
        batches = processor.slice_into_batches(image_groups)

        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)

    def test_slice_preserves_group_structure(self):
        """Test slicing preserves image group structure."""
        # Multiple images per group
        image_groups = [
            [Image(path=Path(f"/test/group1_img{i}.jpg")) for i in range(3)],
            [Image(path=Path(f"/test/group2_img{i}.jpg")) for i in range(2)],
        ]
        batches = self.processor.slice_into_batches(image_groups)

        assert len(batches) == 1
        assert len(batches[0]) == 2
        assert len(batches[0][0]) == 3
        assert len(batches[0][1]) == 2


class TestBatchProcessorApplyLookahead:
    """Test lookahead algorithm for multi-angle grouping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor(batch_size=2, lookahead=3)

    def test_lookahead_empty_current_batch(self):
        """Test lookahead with empty current batch."""
        current_batch = []
        remaining_groups = [[Image(path=Path("/test/image1.jpg"))]]

        new_batch, new_remaining = self.processor.apply_lookahead(
            current_batch, remaining_groups
        )

        assert new_batch == []
        assert new_remaining == remaining_groups

    def test_lookahead_empty_remaining(self):
        """Test lookahead with no remaining groups."""
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = []

        new_batch, new_remaining = self.processor.apply_lookahead(
            current_batch, remaining_groups
        )

        assert new_batch == current_batch
        assert new_remaining == []

    def test_lookahead_no_similar_images(self):
        """Test lookahead with no similar images in peek window."""
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(2, 6)
        ]

        # Mock vision_similarity_score to return 0 (no similarity)
        with patch.object(
            self.processor, "vision_similarity_score", return_value=0.0
        ):
            new_batch, new_remaining = self.processor.apply_lookahead(
                current_batch, remaining_groups
            )

        assert new_batch == current_batch
        assert new_remaining == remaining_groups

    def test_lookahead_moves_similar_group(self):
        """Test lookahead moves similar group to current batch."""
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [
            [Image(path=Path("/test/image2.jpg"))],  # Similar
            [Image(path=Path("/test/image3.jpg"))],
        ]

        # Mock vision_similarity_score: first remaining group similar
        with patch.object(
            self.processor, "vision_similarity_score", side_effect=[0.85, 0.0]
        ):
            new_batch, new_remaining = self.processor.apply_lookahead(
                current_batch, remaining_groups
            )

        assert len(new_batch) == 2
        assert len(new_remaining) == 1
        assert new_batch[1] == remaining_groups[0]

    def test_lookahead_multiple_similar_groups(self):
        """Test lookahead moves multiple similar groups."""
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [
            [Image(path=Path("/test/image2.jpg"))],  # Similar
            [Image(path=Path("/test/image3.jpg"))],  # Similar
            [Image(path=Path("/test/image4.jpg"))],
        ]

        # Mock: first two groups similar
        with patch.object(
            self.processor,
            "vision_similarity_score",
            side_effect=[0.85, 0.9, 0.0],
        ):
            new_batch, new_remaining = self.processor.apply_lookahead(
                current_batch, remaining_groups
            )

        assert len(new_batch) == 3
        assert len(new_remaining) == 1

    def test_lookahead_respects_lookahead_window(self):
        """Test lookahead only checks within lookahead window."""
        processor = BatchProcessor(batch_size=2, lookahead=2)
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(2, 7)
        ]  # 5 groups

        # Mock: all check similar, but only first 2 in window
        with patch.object(
            processor, "vision_similarity_score", return_value=0.85
        ):
            new_batch, new_remaining = processor.apply_lookahead(
                current_batch, remaining_groups
            )

        # Should only move 2 groups (lookahead window)
        assert len(new_batch) == 3
        assert len(new_remaining) == 3

    def test_lookahead_lookahead_zero(self):
        """Test lookahead=0 returns unchanged batches."""
        processor = BatchProcessor(batch_size=2, lookahead=0)
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(2, 4)
        ]

        new_batch, new_remaining = processor.apply_lookahead(
            current_batch, remaining_groups
        )

        assert new_batch == current_batch
        assert new_remaining == remaining_groups

    def test_lookahead_edge_case_empty_candidate_group(self):
        """Test lookahead handles empty candidate groups gracefully."""
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [[], [Image(path=Path("/test/image2.jpg"))]]

        new_batch, new_remaining = self.processor.apply_lookahead(
            current_batch, remaining_groups
        )

        assert len(new_batch) == 1
        assert len(new_remaining) == 2


class TestBatchProcessorVisionSimilarityScore:
    """Test ORB feature matching similarity scoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.ORB_create")
    @patch("cv2.BFMatcher")
    def test_similarity_score_identical_images(
        self, mock_bf, mock_orb, mock_cvtcolor, mock_imread
    ):
        """Test identical images return high similarity."""
        # Mock image loading
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        # Mock grayscale conversion
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_gray

        # Mock ORB detector
        mock_orb_instance = Mock()
        mock_orb.return_value = mock_orb_instance

        # Mock descriptor detection: 10 descriptors, all zeros (identical)
        num_descriptors = 10
        descriptor_size = 32
        mock_descriptors = np.zeros((num_descriptors, descriptor_size), dtype=np.uint8)
        mock_orb_instance.detectAndCompute.return_value = ([], mock_descriptors)

        # Mock matcher: matches with zero distance (identical)
        mock_matcher = Mock()
        mock_matches = [Mock(distance=0.0) for _ in range(num_descriptors)]
        mock_matcher.match.return_value = mock_matches
        mock_bf.return_value = mock_matcher

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 1.0

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.ORB_create")
    @patch("cv2.BFMatcher")
    def test_similarity_score_different_images(
        self, mock_bf, mock_orb, mock_cvtcolor, mock_imread
    ):
        """Test different images return low similarity."""
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_gray

        mock_orb_instance = Mock()
        mock_orb.return_value = mock_orb_instance

        # Mock descriptors with maximum distance (all bits different)
        num_descriptors = 10
        descriptor_size = 32
        mock_descriptors1 = np.zeros((num_descriptors, descriptor_size), dtype=np.uint8)
        mock_descriptors2 = np.ones((num_descriptors, descriptor_size), dtype=np.uint8)

        # First call returns des1, second returns des2
        mock_orb_instance.detectAndCompute.side_effect = [
            ([], mock_descriptors1),
            ([], mock_descriptors2),
        ]

        mock_matcher = Mock()
        # Maximum Hamming distance for 32-byte descriptors
        mock_matches = [Mock(distance=256.0) for _ in range(num_descriptors)]
        mock_matcher.match.return_value = mock_matches
        mock_bf.return_value = mock_matcher

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == pytest.approx(0.0, abs=0.01)

    @patch("cv2.imread")
    def test_similarity_score_load_failure(self, mock_imread):
        """Test image load failure returns zero similarity."""
        mock_imread.return_value = None

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 0.0

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.ORB_create")
    def test_similarity_score_insufficient_descriptors(
        self, mock_orb, mock_cvtcolor, mock_imread
    ):
        """Test insufficient descriptors return zero similarity."""
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_gray

        mock_orb_instance = Mock()
        mock_orb.return_value = mock_orb_instance

        # Only 3 descriptors (below threshold of 5)
        mock_descriptors = np.zeros((3, 32), dtype=np.uint8)
        mock_orb_instance.detectAndCompute.return_value = ([], mock_descriptors)

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 0.0

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.ORB_create")
    def test_similarity_score_no_descriptors(
        self, mock_orb, mock_cvtcolor, mock_imread
    ):
        """Test no descriptors detected return zero similarity."""
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_gray

        mock_orb_instance = Mock()
        mock_orb.return_value = mock_orb_instance

        # No descriptors (None)
        mock_orb_instance.detectAndCompute.return_value = ([], None)

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 0.0

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.ORB_create")
    @patch("cv2.BFMatcher")
    def test_similarity_score_no_matches(
        self, mock_bf, mock_orb, mock_cvtcolor, mock_imread
    ):
        """Test no feature matches return zero similarity."""
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cvtcolor.return_value = mock_gray

        mock_orb_instance = Mock()
        mock_orb.return_value = mock_orb_instance

        mock_descriptors = np.zeros((10, 32), dtype=np.uint8)
        mock_orb_instance.detectAndCompute.return_value = ([], mock_descriptors)

        mock_matcher = Mock()
        mock_matcher.match.return_value = []  # No matches
        mock_bf.return_value = mock_matcher

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 0.0

    @patch("cv2.imread")
    def test_similarity_score_exception_handling(self, mock_imread):
        """Test exception handling on similarity computation."""
        mock_imread.side_effect = Exception("OpenCV error")

        similarity = self.processor.vision_similarity_score(
            "/test/image1.jpg", "/test/image2.jpg"
        )

        assert similarity == 0.0


class TestBatchProcessorProcessBatch:
    """Test batch processing orchestration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()

    @patch("image_processor.batch_processor.BottleAnalyzer")
    def test_process_batch_single_group(self, mock_analyzer_class):
        """Test processing batch with single image group."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        processor = BatchProcessor(bottle_analyzer=mock_analyzer)

        # Mock process_single_bottle
        bottle_dict = {"name": "Test Bottle", "distillery": "Test Distillery"}
        metadata = ExtractedMetadata(
            full_text="OCR text", selected_text="Selected"
        )
        mock_analyzer.process_single_bottle.return_value = (bottle_dict, metadata)

        batch = [[Image(path=Path("/test/image1.jpg"))]]
        result = processor.process_batch(batch)

        assert len(result.bottle_dicts) == 1
        assert len(result.metadata_list) == 1
        assert result.bottle_dicts[0] == bottle_dict
        assert result.failed_bottles == []

    @patch("image_processor.batch_processor.BottleAnalyzer")
    def test_process_batch_multiple_groups(self, mock_analyzer_class):
        """Test processing batch with multiple image groups."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        processor = BatchProcessor(bottle_analyzer=mock_analyzer)

        # Mock process_single_bottle for multiple bottles
        call_count = 0
        def mock_process(bottle_images, vision_model):
            nonlocal call_count
            call_count += 1
            bottle_dict = {
                "name": f"Bottle {call_count}",
                "distillery": "Test Distillery",
            }
            metadata = ExtractedMetadata(
                full_text="OCR text", selected_text="Selected"
            )
            return bottle_dict, metadata

        mock_analyzer.process_single_bottle.side_effect = mock_process

        batch = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 4)
        ]
        result = processor.process_batch(batch)

        assert len(result.bottle_dicts) == 3
        assert len(result.metadata_list) == 3
        assert result.failed_bottles == []

    @patch("image_processor.batch_processor.BottleAnalyzer")
    def test_process_batch_with_failures(self, mock_analyzer_class):
        """Test processing batch handles individual failures."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        processor = BatchProcessor(bottle_analyzer=mock_analyzer)

        # Mock: first succeeds, second fails
        bottle_dict = {"name": "Test Bottle", "distillery": "Test Distillery"}
        metadata = ExtractedMetadata(
            full_text="OCR text", selected_text="Selected"
        )
        mock_analyzer.process_single_bottle.side_effect = [
            (bottle_dict, metadata),
            Exception("Processing failed"),
            (bottle_dict, metadata),
        ]

        batch = [
            [Image(path=Path(f"/test/image{i}.jpg"))] for i in range(1, 4)
        ]
        result = processor.process_batch(batch)

        assert len(result.bottle_dicts) == 2
        assert len(result.failed_bottles) == 1
        assert "image2" in result.failed_bottles[0]

    @patch("image_processor.batch_processor.BottleAnalyzer")
    def test_process_batch_empty_batch(self, mock_analyzer_class):
        """Test processing empty batch returns empty result."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        processor = BatchProcessor(bottle_analyzer=mock_analyzer)

        result = processor.process_batch([])

        assert result.bottle_dicts == []
        assert result.metadata_list == []
        assert result.failed_bottles == []

    @patch("image_processor.batch_processor.BottleAnalyzer")
    def test_process_batch_passes_vision_model(self, mock_analyzer_class):
        """Test processing batch passes vision model parameter."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        processor = BatchProcessor(bottle_analyzer=mock_analyzer)

        bottle_dict = {"name": "Test Bottle", "distillery": "Test Distillery"}
        metadata = ExtractedMetadata(
            full_text="OCR text", selected_text="Selected"
        )
        mock_analyzer.process_single_bottle.return_value = (bottle_dict, metadata)

        batch = [[Image(path=Path("/test/image1.jpg"))]]
        result = processor.process_batch(batch, vision_model="custom-model")

        # Verify vision_model was passed
        mock_analyzer.process_single_bottle.assert_called_once()
        call_kwargs = mock_analyzer.process_single_bottle.call_args.kwargs
        assert call_kwargs["vision_model"] == "custom-model"


class TestBatchProcessorLearnFromFeedback:
    """Test adaptive threshold based on corrections."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor(similarity_threshold=0.7)

    def test_learn_from_empty_corrections(self):
        """Test learning with no corrections does nothing."""
        initial_threshold = self.processor.similarity_threshold
        self.processor.learn_from_feedback([])

        assert self.processor.similarity_threshold == initial_threshold

    def test_learn_from_merge_corrections_tightens_threshold(self):
        """Test merge corrections tighten threshold."""
        corrections = [
            Correction(
                field_name="grouping",
                original_value="separate",
                corrected_value="merge",
                corrected_at=datetime.now(),
            )
            for _ in range(2)
        ]

        self.processor.learn_from_feedback(corrections)

        assert self.processor.similarity_threshold > 0.7

    def test_learn_from_split_corrections_loosens_threshold(self):
        """Test split corrections loosen threshold."""
        corrections = [
            Correction(
                field_name="grouping",
                original_value="grouped",
                corrected_value="split",
                corrected_at=datetime.now(),
            )
            for _ in range(2)
        ]

        self.processor.learn_from_feedback(corrections)

        assert self.processor.similarity_threshold < 0.7

    def test_learn_balanced_feedback(self):
        """Test balanced merge/split feedback keeps threshold unchanged."""
        corrections = [
            Correction(
                field_name="grouping",
                original_value="separate",
                corrected_value="merge",
                corrected_at=datetime.now(),
            ),
            Correction(
                field_name="grouping",
                original_value="grouped",
                corrected_value="split",
                corrected_at=datetime.now(),
            ),
        ]

        initial_threshold = self.processor.similarity_threshold
        self.processor.learn_from_feedback(corrections)

        assert self.processor.similarity_threshold == initial_threshold

    def test_learn_threshold_caps_at_1_0(self):
        """Test threshold caps at maximum value of 1.0."""
        processor = BatchProcessor(similarity_threshold=0.95)
        corrections = [
            Correction(
                field_name="grouping",
                original_value="separate",
                corrected_value="merge",
                corrected_at=datetime.now(),
            )
            for _ in range(10)
        ]

        processor.learn_from_feedback(corrections)

        assert processor.similarity_threshold == 1.0

    def test_learn_threshold_caps_at_0_3(self):
        """Test threshold caps at minimum value of 0.3."""
        processor = BatchProcessor(similarity_threshold=0.35)
        corrections = [
            Correction(
                field_name="grouping",
                original_value="grouped",
                corrected_value="split",
                corrected_at=datetime.now(),
            )
            for _ in range(10)
        ]

        processor.learn_from_feedback(corrections)

        assert processor.similarity_threshold == 0.3

    def test_learn_from_non_grouping_corrections_ignored(self):
        """Test non-grouping corrections are ignored."""
        corrections = [
            Correction(
                field_name="name",
                original_value="Wrong",
                corrected_value="Correct",
                corrected_at=datetime.now(),
            ),
            Correction(
                field_name="distillery",
                original_value="Wrong",
                corrected_value="Correct",
                corrected_at=datetime.now(),
            ),
        ]

        initial_threshold = self.processor.similarity_threshold
        self.processor.learn_from_feedback(corrections)

        assert self.processor.similarity_threshold == initial_threshold

    def test_learn_adjustment_capped(self):
        """Test threshold adjustment is capped at 0.15."""
        processor = BatchProcessor(similarity_threshold=0.5)
        # More than 3 merges should still cap at 0.15 adjustment
        corrections = [
            Correction(
                field_name="grouping",
                original_value="separate",
                corrected_value="merge",
                corrected_at=datetime.now(),
            )
            for _ in range(5)
        ]

        processor.learn_from_feedback(corrections)

        assert processor.similarity_threshold == pytest.approx(0.65, abs=0.01)


class TestBatchProcessorApplyCorrections:
    """Test applying user corrections to bottle dictionaries."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()

    def test_apply_corrections_empty(self):
        """Test applying empty corrections list."""
        bottle_dicts = [{"name": "Test", "distillery": "Test"}]

        result = self.processor.apply_corrections(bottle_dicts, [])

        assert result == bottle_dicts

    def test_apply_correction_updates_field(self):
        """Test applying correction updates bottle field."""
        bottle_dicts = [
            {"name": "Wrong Name", "distillery": "Test Distillery"}
        ]
        corrections = [
            Correction(
                field_name="name",
                original_value="Wrong Name",
                corrected_value="Correct Name",
                corrected_at=datetime.now(),
            )
        ]

        result = self.processor.apply_corrections(bottle_dicts, corrections)

        assert result[0]["name"] == "Correct Name"
        assert result[0]["distillery"] == "Test Distillery"

    def test_apply_correction_stores_history(self):
        """Test applying correction stores history in bottle dict."""
        bottle_dicts = [
            {"name": "Wrong Name", "distillery": "Test Distillery"}
        ]
        correction_time = datetime.now()
        corrections = [
            Correction(
                field_name="name",
                original_value="Wrong Name",
                corrected_value="Correct Name",
                corrected_at=correction_time,
            )
        ]

        result = self.processor.apply_corrections(bottle_dicts, corrections)

        assert "corrections" in result[0]
        assert len(result[0]["corrections"]) == 1
        assert result[0]["corrections"][0]["field_name"] == "name"
        assert result[0]["corrections"][0]["original_value"] == "Wrong Name"
        assert result[0]["corrections"][0]["corrected_value"] == "Correct Name"

    def test_apply_multiple_corrections(self):
        """Test applying multiple corrections to multiple bottles."""
        bottle_dicts = [
            {"name": "Bottle 1", "distillery": "Dist 1"},
            {"name": "Bottle 2", "distillery": "Dist 2"},
        ]
        corrections = [
            Correction(
                field_name="name",
                original_value="Bottle 1",
                corrected_value="Corrected 1",
                corrected_at=datetime.now(),
            ),
            Correction(
                field_name="distillery",
                original_value="Dist 2",
                corrected_value="Corrected 2",
                corrected_at=datetime.now(),
            ),
        ]

        result = self.processor.apply_corrections(bottle_dicts, corrections)

        assert result[0]["name"] == "Corrected 1"
        assert result[1]["distillery"] == "Corrected 2"

    def test_apply_corrections_creates_history_list(self):
        """Test applying multiple corrections creates history list."""
        bottle_dicts = [
            {"name": "Wrong", "distillery": "Test"}
        ]
        corrections = [
            Correction(
                field_name="name",
                original_value="Wrong",
                corrected_value="Correct 1",
                corrected_at=datetime.now(),
            ),
            Correction(
                field_name="name",
                original_value="Correct 1",
                corrected_value="Correct 2",
                corrected_at=datetime.now(),
            ),
        ]

        result = self.processor.apply_corrections(bottle_dicts, corrections)

        assert len(result[0]["corrections"]) == 2
        assert result[0]["corrections"][0]["original_value"] == "Wrong"
        assert result[0]["corrections"][1]["original_value"] == "Correct 1"


class TestBatchProcessorFindMatchingBottle:
    """Test matching bottles for corrections."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()

    def test_find_matching_bottle_by_path(self):
        """Test finding bottle by path."""
        bottle_dicts = [
            {
                "name": "Bottle 1",
                "distillery": "Dist 1",
                "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
            }
        ]
        correction = Correction(
            field_name="name",
            original_value="Bottle 1",
            corrected_value="Corrected",
            corrected_at=datetime.now(),
        )
        correction.bottle_path = "/path/to/image1.jpg"

        result = self.processor._find_matching_bottle(bottle_dicts, correction)

        assert result is not None
        assert result["name"] == "Bottle 1"

    def test_find_matching_bottle_by_name(self):
        """Test finding bottle by name."""
        bottle_dicts = [
            {"name": "Test Bottle", "distillery": "Test"}
        ]
        correction = Correction(
            field_name="distillery",
            original_value="Test",
            corrected_value="Corrected",
            corrected_at=datetime.now(),
        )
        correction.bottle_name = "Test Bottle"

        result = self.processor._find_matching_bottle(bottle_dicts, correction)

        assert result is not None
        assert result["name"] == "Test Bottle"

    def test_find_matching_bottle_not_found(self):
        """Test finding non-existent bottle returns None."""
        bottle_dicts = [
            {"name": "Bottle 1", "distillery": "Dist 1"},
            {"name": "Bottle 2", "distillery": "Dist 2"},
        ]
        correction = Correction(
            field_name="name",
            original_value="Bottle 3",
            corrected_value="Corrected",
            corrected_at=datetime.now(),
        )
        correction.bottle_name = "Nonexistent"

        result = self.processor._find_matching_bottle(bottle_dicts, correction)

        assert result is None

    def test_find_matching_bottle_no_matching_attrs(self):
        """Test finding bottle by field value when path/name not specified."""
        bottle_dicts = [
            {"name": "Bottle 1", "distillery": "Dist 1"}
        ]
        correction = Correction(
            field_name="name",
            original_value="Bottle 1",
            corrected_value="Corrected",
            corrected_at=datetime.now(),
        )

        result = self.processor._find_matching_bottle(bottle_dicts, correction)

        # Should match by field original_value when no explicit path/name
        assert result is not None
        assert result["name"] == "Bottle 1"


class TestBatchResultDataclass:
    """Test BatchResult dataclass."""

    def test_batch_result_fields(self):
        """Test BatchResult has correct fields."""
        result = BatchResult(
            batch_id=123,
            bottle_dicts=[{"name": "Test"}],
            metadata_list=[ExtractedMetadata(full_text="OCR", selected_text="Sel")],
            corrections=[
                Correction(
                    field_name="name",
                    original_value="Wrong",
                    corrected_value="Right",
                    corrected_at=datetime.now(),
                )
            ],
            failed_bottles=["bottle1", "bottle2"],
        )

        assert result.batch_id == 123
        assert len(result.bottle_dicts) == 1
        assert len(result.metadata_list) == 1
        assert len(result.corrections) == 1
        assert len(result.failed_bottles) == 2

    def test_batch_result_defaults(self):
        """Test BatchResult default values."""
        result = BatchResult(batch_id=1)

        assert result.bottle_dicts == []
        assert result.metadata_list == []
        assert result.corrections == []
        assert result.failed_bottles == []

    def test_batch_result_empty_lists_are_mutable(self):
        """Test BatchResult empty lists can be mutated."""
        result = BatchResult(batch_id=1)

        result.bottle_dicts.append({"name": "Test"})
        result.metadata_list.append(ExtractedMetadata(full_text="OCR", selected_text="Sel"))
        result.failed_bottles.append("failed")

        assert len(result.bottle_dicts) == 1
        assert len(result.metadata_list) == 1
        assert len(result.failed_bottles) == 1
