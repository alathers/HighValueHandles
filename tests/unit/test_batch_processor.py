"""Unit tests for BatchProcessor class.

Tests batch slicing, the deprecated lookahead no-op, the same-bottle split
check (backstop), batch processing, and user feedback integration.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from image_processor.batch_processor import (
    ADJACENT_SAME_HINT,
    FALSE_ALARM_HINT,
    MIN_SPLIT_BUDGET_CALLS,
    SPLIT_BUDGET_WARNING,
    BatchProcessor,
    BatchResult,
    photo_units,
    split_budget_for,
    split_group_at_largest_gap,
)
from image_processor.bottle_analyzer import MIXED_GROUP_WARNING
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
    """Lookahead is deprecated: grouping by capture time keeps bottles inside one batch."""

    def test_apply_lookahead_is_a_noop(self):
        processor = BatchProcessor(batch_size=2, lookahead=3)
        current_batch = [[Image(path=Path("/test/image1.jpg"))]]
        remaining_groups = [[Image(path=Path(f"/test/image{i}.jpg"))] for i in range(2, 6)]

        new_batch, new_remaining = processor.apply_lookahead(current_batch, remaining_groups)

        assert new_batch is current_batch
        assert new_remaining is remaining_groups

    def test_orb_similarity_removed(self):
        assert not hasattr(BatchProcessor, "vision_similarity_score")


# --- same-bottle split check ("backstop") ----------------------------------

STAGG = {"name": "George T. Stagg", "distillery": "Buffalo Trace", "proof": "135.4 Proof"}


def _photos(times, prefix="IMG_", bottle_ref=None):
    """Images named IMG_0..IMG_n with the given capture times (None = untimed)."""
    return [
        Image(path=Path(f"/shoot/{prefix}{k}.jpg"), capture_time=t, bottle_ref=bottle_ref)
        for k, t in enumerate(times)
    ]


class FakeAnalyzer:
    """Stands in for BottleAnalyzer.process_single_bottle.

    ``label_of`` maps a photo name to the label fields it shows. A call with
    photos of more than one label reports ``same_bottle=False`` and the
    majority label (as the real model is asked to); ``mixed_always`` makes
    every multi-photo call report False (an over-eager model).
    """

    def __init__(self, label_of, mixed_always=False, fail_on=None):
        self.label_of = label_of
        self.mixed_always = mixed_always
        self.fail_on = fail_on or {}
        self.calls: list[list[str]] = []

    def process_single_bottle(self, images, vision_model=None):
        names = [Path(img.path).name for img in images]
        self.calls.append(names)
        error = self.fail_on.get(tuple(names))
        if error is not None:
            raise error
        sent = [n for img, n in zip(images, names) if not img.is_duplicate and not img.near_duplicate_of]
        labels = [self.label_of[n] for n in sent]
        keys = [tuple(sorted(lbl.items())) for lbl in labels]
        majority = dict(max(set(keys), key=keys.count))
        mixed = len(sent) > 1 and (self.mixed_always or len(set(keys)) > 1)
        metadata = ExtractedMetadata(
            **majority,
            source_images=[str(img.path) for img in images],
            same_bottle=(not mixed) if len(sent) > 1 else None,
            warnings=[MIXED_GROUP_WARNING] if mixed else [],
        )
        bottle_dict = {"name": majority.get("name", "Unknown"), "images": [str(img.path) for img in images]}
        return bottle_dict, metadata


def _names(result):
    return [[Path(p).name for p in d["images"]] for d in result.bottle_dicts]


class TestSplitBudget:
    def test_budget_formula(self):
        assert split_budget_for(0, 0.35) == 0
        assert split_budget_for(10, 0) == 0
        assert split_budget_for(1, 0.35) == MIN_SPLIT_BUDGET_CALLS
        assert split_budget_for(20, 0.35) == 7  # not 8 from float noise
        assert split_budget_for(52, 0.35) == 19
        assert split_budget_for(52, 0.5) == 26

    @pytest.mark.parametrize("ratio", [-0.1, 2.5])
    def test_ratio_out_of_range_rejected(self, ratio):
        with pytest.raises(ValueError, match="split_budget_ratio"):
            BatchProcessor(bottle_analyzer=Mock(), split_budget_ratio=ratio)

    def test_budget_shared_across_batches_and_exhausted_warns(self):
        """One budget for the run: 10 groups -> 4 calls -> two splits, the third mixed group is kept."""
        groups = []
        labels = {}
        for g in range(10):
            imgs = _photos([0.0, 2.0], prefix=f"G{g}_", bottle_ref=f"G{g}_0")
            groups.append(imgs)
            mixed = g in (0, 4, 8)  # one mixed group in each of three batches
            labels[f"G{g}_0.jpg"] = {**STAGG, "batch_number": "23A"}
            labels[f"G{g}_1.jpg"] = {**STAGG, "batch_number": "23C" if mixed else "23A"}
        analyzer = FakeAnalyzer(labels)
        processor = BatchProcessor(batch_size=4, bottle_analyzer=analyzer, split_budget_ratio=0.35)

        assert processor.start_run(groups) == 4
        results = [processor.process_batch(batch) for batch in processor.slice_into_batches(groups)]

        assert [len(r.bottle_dicts) for r in results] == [5, 5, 2]  # groups 0 and 4 split, 8 not
        assert len(analyzer.calls) == 10 + 4
        assert processor.split_calls_used == 4
        assert processor.split_calls_remaining == 0
        kept = results[2].metadata_list[0]
        assert SPLIT_BUDGET_WARNING in kept.warnings
        assert MIXED_GROUP_WARNING in kept.warnings
        assert processor.split_stats["over_budget"] == 1
        assert "2 split" in processor.split_summary()

    def test_process_batch_without_start_run_still_bounded(self):
        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        analyzer = FakeAnalyzer(labels)
        processor = BatchProcessor(bottle_analyzer=analyzer)

        processor.process_batch([_photos([0.0, 4.0])])
        first_budget = processor.split_budget
        processor.process_batch([_photos([0.0, 4.0])])

        assert first_budget == MIN_SPLIT_BUDGET_CALLS
        assert processor.split_calls_used == 4  # second batch drew from the same budget
        assert len(analyzer.calls) == 6

    def test_ratio_zero_disables_split_check(self):
        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        analyzer = FakeAnalyzer(labels)
        processor = BatchProcessor(bottle_analyzer=analyzer, split_budget_ratio=0)

        result = processor.process_batch([_photos([0.0, 4.0])])

        assert len(analyzer.calls) == 1
        assert len(result.bottle_dicts) == 1
        assert SPLIT_BUDGET_WARNING not in result.metadata_list[0].warnings


class TestSameBottleSplitCheck:
    def test_same_bottle_true_costs_one_call(self):
        labels = {f"IMG_{k}.jpg": STAGG for k in range(3)}
        analyzer = FakeAnalyzer(labels)

        result = BatchProcessor(bottle_analyzer=analyzer).process_batch([_photos([0.0, 3.0, 6.0])])

        assert len(analyzer.calls) == 1
        assert _names(result) == [["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]]

    def test_split_at_largest_gap_into_distinct_bottles(self):
        """Gaps [2, 9, 2]: split between photos 1 and 2; Stagg 23A vs 23C stays split."""
        labels = {
            "IMG_0.jpg": {**STAGG, "batch_number": "23A"},
            "IMG_1.jpg": {**STAGG, "batch_number": "23A"},
            "IMG_2.jpg": {**STAGG, "batch_number": "23C"},
            "IMG_3.jpg": {**STAGG, "batch_number": "23C"},
        }
        analyzer = FakeAnalyzer(labels)
        group = _photos([0.0, 2.0, 11.0, 13.0], bottle_ref="IMG_0")
        processor = BatchProcessor(bottle_analyzer=analyzer)
        processor.start_run([group])

        result = processor.process_batch([group])

        assert len(analyzer.calls) == 3
        assert _names(result) == [["IMG_0.jpg", "IMG_1.jpg"], ["IMG_2.jpg", "IMG_3.jpg"]]
        assert [m.batch_number for m in result.metadata_list] == ["23A", "23C"]
        assert [d["bottle_id"] for d in result.bottle_dicts] == ["IMG_0", "IMG_2"]
        assert [img.bottle_ref for img in group] == ["IMG_0", "IMG_0", "IMG_2", "IMG_2"]
        for metadata in result.metadata_list:
            assert any(w.startswith("Split from a group of 4 photos") for w in metadata.warnings)
            assert "batch_number differ" in metadata.warnings[-1]
        assert result.failed_bottles == []

    def test_split_part_id_never_collides_with_another_group(self):
        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        group = _photos([0.0, 5.0], bottle_ref="IMG_0")
        other = [Image(path=Path("/elsewhere/IMG_1.jpg"), bottle_ref="IMG_1")]
        processor = BatchProcessor(bottle_analyzer=FakeAnalyzer({**labels}))
        processor.start_run([group, other])

        result = processor.process_batch([group])

        new_id = result.bottle_dicts[1]["bottle_id"]
        assert new_id.startswith("IMG_1_") and new_id != "IMG_1"

    def test_false_alarm_keeps_one_bottle_with_review_hint(self):
        """Front shows no batch, back shows 23A: wildcard, same bottle -> merged result kept."""
        labels = {
            "IMG_0.jpg": {**STAGG},
            "IMG_1.jpg": {**STAGG, "batch_number": "23A"},
        }
        analyzer = FakeAnalyzer(labels)  # different label dicts -> the group call says mixed

        result = BatchProcessor(bottle_analyzer=analyzer).process_batch([_photos([0.0, 7.0])])

        assert len(analyzer.calls) == 3
        assert _names(result) == [["IMG_0.jpg", "IMG_1.jpg"]]
        warnings = result.metadata_list[0].warnings
        assert FALSE_ALARM_HINT in warnings
        assert MIXED_GROUP_WARNING not in warnings
        # Persisted data must not still claim the photos are mixed.
        assert result.metadata_list[0].same_bottle is None
        assert result.metadata_list[0].to_dict()["same_bottle"] is None

    def test_over_eager_model_does_not_tear_one_bottle_apart(self):
        """Every multi-photo call says mixed, but all photos read the same: one bottle."""
        labels = {f"IMG_{k}.jpg": STAGG for k in range(3)}
        analyzer = FakeAnalyzer(labels, mixed_always=True)
        processor = BatchProcessor(bottle_analyzer=analyzer)

        result = processor.process_batch([_photos([0.0, 2.0, 9.0])])

        # [0,1,2] -> [0,1] | [2]; [0,1] still "mixed" -> [0] | [1] agree, so [0,1]
        # is one bottle; it agrees with [2] too, so the whole group is one bottle.
        assert _names(result) == [["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]]
        assert len(analyzer.calls) == 5
        metadata = result.metadata_list[0]
        assert FALSE_ALARM_HINT in metadata.warnings
        assert metadata.same_bottle is None
        assert not any(w.startswith("Split from") for w in metadata.warnings)
        assert (processor.split_stats["flagged"], processor.split_stats["false_alarm"]) == (1, 1)
        assert processor.split_stats["split"] == 0

    def test_half_still_mixed_and_really_mixed_keeps_split(self):
        """Left half [0,1] mixes 23A/23C (no conflict with the right half's majority): split kept."""
        labels = {
            "IMG_0.jpg": {**STAGG, "batch_number": "23A"},
            "IMG_1.jpg": {**STAGG, "batch_number": "23C"},
            "IMG_2.jpg": {**STAGG, "batch_number": "23C"},
        }
        analyzer = FakeAnalyzer(labels, mixed_always=True)
        processor = BatchProcessor(bottle_analyzer=analyzer)

        result = processor.process_batch([_photos([0.0, 2.0, 9.0])])

        # [0,1] reads 23A or 23C as majority; either way [0] vs [1] conflict on batch.
        assert _names(result) == [["IMG_0.jpg"], ["IMG_1.jpg"], ["IMG_2.jpg"]]
        assert processor.split_stats["flagged"] == 1  # nested checks do not count as groups
        assert processor.split_stats["split"] == 1

    def test_recursion_stops_at_single_photos(self):
        """Worst case for n photos is 1 + 2(n-1) calls; no single photo is analyzed twice."""
        n = 6
        labels = {f"IMG_{k}.jpg": {**STAGG, "batch_number": f"B{k}"} for k in range(n)}
        analyzer = FakeAnalyzer(labels)
        group = _photos([3.0 * k for k in range(n)])
        processor = BatchProcessor(bottle_analyzer=analyzer, split_budget_ratio=2.0)
        processor.start_run([group] * 10)  # budget 20, enough for everything

        result = processor.process_batch([group])

        assert sorted(_names(result)) == [[f"IMG_{k}.jpg"] for k in range(n)]
        assert len(analyzer.calls) == 1 + 2 * (n - 1)
        assert (processor.split_stats["flagged"], processor.split_stats["split"]) == (1, 1)
        assert "1 group(s) flagged as mixed, 1 split" in processor.split_summary()
        singles = [tuple(c) for c in analyzer.calls if len(c) == 1]
        assert len(singles) == len(set(singles)) == n

    def test_single_photo_never_re_asked(self):
        analyzer = Mock()
        analyzer.process_single_bottle.return_value = (
            {"name": "X"},
            ExtractedMetadata(name="X", same_bottle=False),  # meaningless for one photo
        )

        result = BatchProcessor(bottle_analyzer=analyzer).process_batch([_photos([0.0])])

        assert analyzer.process_single_bottle.call_count == 1
        assert len(result.bottle_dicts) == 1

    def test_duplicates_and_burst_retakes_stay_with_their_original(self):
        group = _photos([0.0, None, 1.0, 9.0])
        group[1].is_duplicate = True  # exact copy of IMG_0, placed right after it
        group[2].near_duplicate_of = str(group[0].path)

        left, right = split_group_at_largest_gap(group)

        assert [Path(i.path).name for i in left] == ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]
        assert [Path(i.path).name for i in right] == ["IMG_3.jpg"]
        assert len(photo_units(group)) == 2

    def test_untimed_group_splits_in_the_middle(self):
        left, right = split_group_at_largest_gap(_photos([None, None, None, None]))
        assert len(left) == 2 and len(right) == 2

    def test_failed_split_check_keeps_group_with_warning(self):
        from lib.exceptions import VisionResponseError

        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        analyzer = FakeAnalyzer(labels, fail_on={("IMG_1.jpg",): VisionResponseError("no content")})

        result = BatchProcessor(bottle_analyzer=analyzer).process_batch([_photos([0.0, 5.0])])

        assert len(result.bottle_dicts) == 1
        assert result.failed_bottles == []
        assert any("split check failed (VisionResponseError: no content)" in w for w in result.metadata_list[0].warnings)

    def test_fatal_error_in_split_check_propagates(self):
        from lib.exceptions import OllamaConnectionError

        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        analyzer = FakeAnalyzer(labels, fail_on={("IMG_0.jpg",): OllamaConnectionError("gone")})

        with pytest.raises(OllamaConnectionError):
            BatchProcessor(bottle_analyzer=analyzer).process_batch([_photos([0.0, 5.0])])


class TestAdjacentSameBottleHint:
    """Report-only 'possibly the same bottle' hints and the --group-max-size nudge."""

    def _run(self, groups, labels, batch_size=5, max_group_size=3, **kwargs):
        processor = BatchProcessor(
            batch_size=batch_size,
            bottle_analyzer=FakeAnalyzer(labels, **kwargs),
            max_group_size=max_group_size,
        )
        processor.start_run(groups)
        results = [processor.process_batch(batch) for batch in processor.slice_into_batches(groups)]
        return processor, results

    @staticmethod
    def _group(prefix, times):
        return _photos(times, prefix=prefix, bottle_ref=f"{prefix}0")

    def test_adjacent_identical_label_gets_hint_across_batches_never_merged(self):
        """Blanton's box+bottle split 3+2 by the cap, and by the batch boundary too."""
        first = self._group("A_", [0.0, 4.0, 8.0])
        second = self._group("B_", [40.0, 44.0])
        labels = {f"{p}{k}.jpg": {"name": "Blanton's", "proof": "93"} for p in ("A_", "B_") for k in range(3)}

        processor, results = self._run([first, second], labels, batch_size=1)

        assert [len(r.bottle_dicts) for r in results] == [1, 1]  # not merged
        expected = ADJACENT_SAME_HINT.format(bottle_id="A_0")
        assert expected in results[1].metadata_list[0].warnings
        assert expected.startswith("Possibly the same bottle as A_0 (adjacent photos, identical label)")
        assert results[0].metadata_list[0].warnings == []
        assert processor.split_stats["adjacent_hint"] == 1
        assert processor.split_stats["adjacent_hint_at_cap"] == 1
        nudge = processor.group_size_nudge()
        assert nudge is not None and "--group-max-size 4" in nudge

    @pytest.mark.parametrize(
        ("second_times", "second_label"),
        [
            ([75.0, 79.0], {"name": "Blanton's", "proof": "93"}),  # 67 s after: too far
            ([40.0, 44.0], {"name": "Blanton's", "proof": "103"}),  # proof read on both differs
            ([40.0, 44.0], {"name": "Eagle Rare"}),  # different name
            ([None, None], {"name": "Blanton's", "proof": "93"}),  # untimed
            ([40.0, 44.0], {"proof": "93"}),  # no name read: would match anything
        ],
    )
    def test_no_hint(self, second_times, second_label):
        first = self._group("A_", [0.0, 4.0, 8.0])
        second = self._group("B_", second_times)
        labels = {f"A_{k}.jpg": {"name": "Blanton's", "proof": "93"} for k in range(3)}
        labels.update({f"B_{k}.jpg": second_label for k in range(2)})

        processor, results = self._run([first, second], labels)

        assert not any(w.startswith("Possibly the same bottle") for m in results[0].metadata_list for w in m.warnings)
        assert processor.split_stats["adjacent_hint"] == 0
        assert processor.group_size_nudge() is None

    def test_failed_bottle_breaks_adjacency(self):
        from lib.exceptions import VisionResponseError

        groups = [self._group("A_", [0.0]), self._group("F_", [10.0]), self._group("B_", [20.0])]
        labels = {"A_0.jpg": {"name": "Weller"}, "F_0.jpg": {"name": "Weller"}, "B_0.jpg": {"name": "Weller"}}
        processor, results = self._run(groups, labels, fail_on={("F_0.jpg",): VisionResponseError("x")})

        assert results[0].failed_bottles == ["F_0"]
        assert processor.split_stats["adjacent_hint"] == 0

    def test_nudge_needs_a_quarter_of_hints_at_cap(self):
        # Four singles 10 s apart with the same label: 3 hints, none at cap 3.
        groups = [self._group(f"S{g}_", [10.0 * g]) for g in range(4)]
        labels = {f"S{g}_0.jpg": {"name": "Elmer T. Lee"} for g in range(4)}
        processor, _ = self._run(groups, labels)
        assert processor.split_stats["adjacent_hint"] == 3
        assert processor.group_size_nudge() is None

        # Same, but one group is a full triple: 2 of 3 hints touch it (>= 25%).
        groups[1] = self._group("S1_", [10.0, 12.0, 14.0])
        labels.update({f"S1_{k}.jpg": {"name": "Elmer T. Lee"} for k in range(3)})
        processor, _ = self._run(groups, labels)
        assert (processor.split_stats["adjacent_hint"], processor.split_stats["adjacent_hint_at_cap"]) == (3, 2)
        assert "consider --group-max-size 4" in processor.group_size_nudge()

        # No cap: never nudges.
        processor, _ = self._run(groups, labels, max_group_size=0)
        assert processor.group_size_nudge() is None


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


class TestBatchProcessorVisionFailures:
    """Per-bottle failures are recorded with reasons; fatal model errors abort the batch."""

    def test_process_batch_default_vision_model_is_none(self):
        analyzer = Mock()
        analyzer.process_single_bottle.return_value = ({"name": "X"}, ExtractedMetadata())
        BatchProcessor(bottle_analyzer=analyzer).process_batch([[Image(path=Path("/a.jpg"))]])
        assert analyzer.process_single_bottle.call_args.kwargs["vision_model"] is None

    def test_failure_reason_recorded(self):
        from lib.exceptions import VisionResponseError

        analyzer = Mock()
        analyzer.process_single_bottle.side_effect = [
            VisionResponseError("returned no content"),
            ({"name": "X"}, ExtractedMetadata()),
            RuntimeError("disk full"),
        ]
        batch = [[Image(path=Path(f"/test/IMG_{i}.jpg"))] for i in range(1, 4)]

        result = BatchProcessor(bottle_analyzer=analyzer).process_batch(batch)

        assert result.failed_bottles == ["IMG_1", "IMG_3"]
        assert result.errors == {
            "IMG_1": "VisionResponseError: returned no content",
            "IMG_3": "RuntimeError: disk full",
        }
        assert len(result.bottle_dicts) == 1

    @pytest.mark.parametrize(
        "error_name", ["VisionModelUnavailableError", "OllamaConnectionError", "ModelServerError"]
    )
    def test_fatal_errors_reraised(self, error_name):
        import lib.exceptions as exceptions

        error = getattr(exceptions, error_name)("gone")
        analyzer = Mock()
        analyzer.process_single_bottle.side_effect = error
        batch = [[Image(path=Path("/a.jpg"))], [Image(path=Path("/b.jpg"))]]

        with pytest.raises(type(error)) as exc_info:
            BatchProcessor(bottle_analyzer=analyzer).process_batch(batch)

        assert exc_info.value is error
        analyzer.process_single_bottle.assert_called_once()  # stops at the first bottle


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
