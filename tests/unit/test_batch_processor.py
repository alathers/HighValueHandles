"""Unit tests for BatchProcessor class.

Tests batch slicing, the same-bottle split check (backstop) and batch
processing. Corrections are tested in ``test_corrections.py``.
"""

import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from image_processor.batch_processor import (
    ADJACENT_SAME_HINT,
    FALSE_ALARM_HINT,
    MIN_SPLIT_BUDGET_CALLS,
    SPLIT_BUDGET_WARNING,
    BOTTLE_CONTEXT,
    BatchProcessor,
    BatchResult,
    BottleLogFilter,
    describe_error,
    photo_units,
    split_budget_for,
    split_group_at_largest_gap,
)
from image_processor.bottle_analyzer import MIXED_GROUP_WARNING
from models.bottle import ExtractedMetadata
from models.image import Image


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization."""

    def test_processor_default_values(self):
        """Test processor initializes with default values."""
        processor = BatchProcessor()

        assert processor.batch_size == 5
        assert processor.bottle_analyzer is not None

    def test_processor_custom_values(self):
        """Test processor initializes with custom values."""
        with patch("image_processor.batch_processor.BottleAnalyzer") as mock_analyzer:
            mock_analyzer.return_value = Mock()
            processor = BatchProcessor(batch_size=10)

            assert processor.batch_size == 10

    def test_only_batch_size_is_positional(self):
        params = inspect.signature(BatchProcessor.__init__).parameters
        assert [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD] == ["self", "batch_size"]
        assert set(params) == {
            "self", "batch_size", "bottle_analyzer", "split_budget_ratio", "max_group_size", "executor", "hints"
        }

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


class TestRemovedGroupingLeftovers:
    def test_orb_similarity_removed(self):
        assert not hasattr(BatchProcessor, "vision_similarity_score")
        assert not hasattr(BatchProcessor, "apply_corrections")


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
        self.notes: dict[tuple[str, ...], str] = {}

    def process_single_bottle(self, images, vision_model=None, *, reviewer_note="", cancel=None):
        names = [Path(img.path).name for img in images]
        self.calls.append(names)
        self.notes[tuple(names)] = reviewer_note
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
        def mock_process(bottle_images, vision_model, **_kwargs):
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
        processor.process_batch(batch, vision_model="custom-model")

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
        assert {k: describe_error(v) for k, v in result.errors.items()} == {
            "IMG_1": "VisionResponseError: returned no content",
            "IMG_3": "RuntimeError: disk full",
        }
        assert all(isinstance(e, Exception) for e in result.errors.values())  # the real exceptions
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


class TestBatchResultDataclass:
    """Test BatchResult dataclass."""

    def test_batch_result_fields(self):
        """Test BatchResult has correct fields."""
        result = BatchResult(
            batch_id=123,
            bottle_dicts=[{"name": "Test"}],
            metadata_list=[ExtractedMetadata(full_text="OCR", selected_text="Sel")],
            failed_bottles=["bottle1", "bottle2"],
        )

        assert result.batch_id == 123
        assert len(result.bottle_dicts) == 1
        assert len(result.metadata_list) == 1
        assert len(result.failed_bottles) == 2

    def test_batch_result_defaults(self):
        """Test BatchResult default values."""
        result = BatchResult(batch_id=1)

        assert result.bottle_dicts == []
        assert result.metadata_list == []
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


# --- --concurrency: two phases, deterministic output ------------------------

WAIT = 5  # seconds; a barrier or event that times out means the code did not run in parallel


@pytest.fixture
def pool():
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bottle")
    yield executor
    executor.shutdown(wait=True, cancel_futures=True)


def _group(prefix, times=(0.0,), start=0.0):
    return _photos([start + t for t in times], prefix=prefix, bottle_ref=f"{prefix}0")


class OrderedAnalyzer:
    """Group k (k < n-1) finishes only after group k+1: completion order is reversed."""

    def __init__(self, names):
        self.names = names
        self.done = {n: threading.Event() for n in names}
        self.completed: list[str] = []
        self.lock = threading.Lock()

    def process_single_bottle(self, images, vision_model=None, *, reviewer_note="", cancel=None):
        name = Path(images[0].path).stem
        k = self.names.index(name)
        if k + 1 < len(self.names):
            assert self.done[self.names[k + 1]].wait(WAIT), "groups did not run in parallel"
        with self.lock:
            self.completed.append(name)
        self.done[name].set()
        return {"name": name, "images": [str(i.path) for i in images]}, ExtractedMetadata(name=name)


class TestConcurrency:
    def test_results_in_input_order_when_completion_is_reversed(self, pool):
        names = ["A_0", "B_0", "C_0"]
        analyzer = OrderedAnalyzer(names)
        batch = [_group("A_", start=0), _group("B_", start=1000), _group("C_", start=2000)]

        result = BatchProcessor(bottle_analyzer=analyzer, executor=pool).process_batch(batch)

        assert analyzer.completed == ["C_0", "B_0", "A_0"]
        assert [d["name"] for d in result.bottle_dicts] == names
        assert [m.name for m in result.metadata_list] == names

    @staticmethod
    def _scripted_run(executor):
        """Mixed groups (split and false alarm), adjacent identical labels, a failure; 2 batches."""
        labels = {}
        groups = []
        for g in range(8):
            imgs = _group(f"G{g}_", times=(0.0, 3.0), start=g * 30.0)
            groups.append(imgs)
            mixed = g in (1, 5)
            labels[f"G{g}_0.jpg"] = {**STAGG, "batch_number": "23A"}
            labels[f"G{g}_1.jpg"] = {**STAGG, "batch_number": "23C" if mixed else "23A"}
        from lib.exceptions import VisionResponseError

        analyzer = FakeAnalyzer(labels, fail_on={("G3_0.jpg", "G3_1.jpg"): VisionResponseError("no content")})
        processor = BatchProcessor(batch_size=4, bottle_analyzer=analyzer, executor=executor, max_group_size=2)
        processor.start_run(groups)
        results = [processor.process_batch(b) for b in processor.slice_into_batches(groups)]
        return processor, results

    def test_concurrency_1_and_4_give_identical_output(self, pool):
        serial, serial_results = self._scripted_run(None)
        parallel, parallel_results = self._scripted_run(pool)

        def summary(results):
            return [
                (
                    r.bottle_dicts,
                    [m.to_dict() for m in r.metadata_list],
                    r.failed_bottles,
                    {k: describe_error(v) for k, v in r.errors.items()},
                )
                for r in results
            ]

        assert summary(parallel_results) == summary(serial_results)
        assert [d["bottle_id"] for r in parallel_results for d in r.bottle_dicts] == [
            d["bottle_id"] for r in serial_results for d in r.bottle_dicts
        ]
        assert parallel.split_stats == serial.split_stats
        assert parallel.split_stats["split"] == 2
        assert parallel.split_stats["adjacent_hint"] == serial.split_stats["adjacent_hint"] > 0
        assert parallel.split_calls_used == serial.split_calls_used
        assert parallel._progress_done == serial._progress_done == 8

    def test_split_halves_run_concurrently(self, pool):
        barrier = threading.Barrier(2, timeout=WAIT)
        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        fake = FakeAnalyzer(labels)

        class HalvesMeet:
            def process_single_bottle(self, images, vision_model=None, **kwargs):
                if len(images) == 1:
                    barrier.wait()  # both halves must be in flight at once
                return fake.process_single_bottle(images, vision_model, **kwargs)

        result = BatchProcessor(bottle_analyzer=HalvesMeet(), executor=pool).process_batch(
            [_photos([0.0, 5.0], bottle_ref="IMG_0")]
        )

        assert _names(result) == [["IMG_0.jpg"], ["IMG_1.jpg"]]
        assert [d["bottle_id"] for d in result.bottle_dicts] == ["IMG_0", "IMG_1"]

    def test_fatal_error_cancels_pending_work_and_reraises(self):
        from image_processor.bottle_analyzer import AnalysisCancelledError
        from lib.exceptions import OllamaConnectionError

        fatal = OllamaConnectionError("server gone")
        started: list[str] = []
        finished: list[str] = []
        lock = threading.Lock()

        class Analyzer:
            def process_single_bottle(self, images, vision_model=None, *, reviewer_note="", cancel=None):
                name = Path(images[0].path).stem
                if cancel is not None and cancel.is_set():
                    raise AnalysisCancelledError("run is aborting")
                with lock:
                    started.append(name)
                if name == "G1_0":
                    raise fatal
                if name == "G0_0":
                    assert cancel.wait(WAIT)  # still in flight when the fatal error arrives
                with lock:
                    finished.append(name)
                return {"name": name, "images": [str(i.path) for i in images]}, ExtractedMetadata(name=name)

        batch = [_group(f"G{g}_", start=g * 100.0) for g in range(6)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            processor = BatchProcessor(bottle_analyzer=Analyzer(), executor=executor)
            with pytest.raises(OllamaConnectionError) as excinfo:
                processor.process_batch(batch)

        assert excinfo.value is fatal
        assert sorted(started) == ["G0_0", "G1_0"]  # nothing new started after the fatal error
        assert finished == ["G0_0"]  # the in-flight analysis was waited for
        assert processor.cancel.is_set()

    def test_first_fatal_error_in_input_order_wins(self, pool):
        from lib.exceptions import ModelServerError, VisionModelUnavailableError

        errors = {"A_0": VisionModelUnavailableError("model gone"), "B_0": ModelServerError("server gone")}
        both_started = threading.Barrier(2, timeout=WAIT)

        class Analyzer:
            def process_single_bottle(self, images, vision_model=None, **_kwargs):
                both_started.wait()
                raise errors[Path(images[0].path).stem]

        with pytest.raises(VisionModelUnavailableError):
            BatchProcessor(bottle_analyzer=Analyzer(), executor=pool).process_batch(
                [_group("A_"), _group("B_", start=500)]
            )

    def test_non_fatal_error_does_not_cancel_the_others(self, pool):
        from lib.exceptions import VisionResponseError

        class Analyzer:
            def __init__(self):
                self.cancel_seen = []

            def process_single_bottle(self, images, vision_model=None, *, reviewer_note="", cancel=None):
                name = Path(images[0].path).stem
                self.cancel_seen.append(cancel.is_set())
                if name == "B_0":
                    raise VisionResponseError("no content")
                return {"name": name, "images": [str(i.path) for i in images]}, ExtractedMetadata(name=name)

        analyzer = Analyzer()
        batch = [_group("A_"), _group("B_", start=500), _group("C_", start=1000), _group("D_", start=1500)]
        result = BatchProcessor(bottle_analyzer=analyzer, executor=pool).process_batch(batch)

        assert [d["name"] for d in result.bottle_dicts] == ["A_0", "C_0", "D_0"]
        assert result.failed_bottles == ["B_0"]
        assert isinstance(result.errors["B_0"], VisionResponseError)
        assert analyzer.cancel_seen == [False] * 4

    def test_hint_note_passed_to_first_pass_and_both_halves(self, pool):
        labels = {"IMG_0.jpg": {**STAGG, "batch_number": "23A"}, "IMG_1.jpg": {**STAGG, "batch_number": "23C"}}
        analyzer = FakeAnalyzer(labels)
        hints = Mock()
        hints.note_for.side_effect = lambda bottle_id: "neck label" if bottle_id == "IMG_0" else ""

        BatchProcessor(bottle_analyzer=analyzer, executor=pool, hints=hints).process_batch(
            [_photos([0.0, 5.0], bottle_ref="IMG_0")]
        )

        assert analyzer.notes == {
            ("IMG_0.jpg", "IMG_1.jpg"): "neck label",
            ("IMG_0.jpg",): "neck label",
            ("IMG_1.jpg",): "neck label",
        }


class TestBottleLogFilter:
    def test_bottle_defaults_to_dash_and_is_set_inside_a_task(self, pool, caplog):
        handler_filter = BottleLogFilter()
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        assert handler_filter.filter(record) and record.bottle == "-"

        seen = []

        class Analyzer:
            def process_single_bottle(self, images, vision_model=None, **_kwargs):
                rec = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
                handler_filter.filter(rec)
                seen.append((Path(images[0].path).stem, rec.bottle, BOTTLE_CONTEXT.get()))
                return {"name": "X", "images": [str(i.path) for i in images]}, ExtractedMetadata(name="X")

        BatchProcessor(bottle_analyzer=Analyzer(), executor=pool).process_batch(
            [_group("A_"), _group("B_", start=500)]
        )
        assert sorted(seen) == [("A_0", "A_0", "A_0"), ("B_0", "B_0", "B_0")]
        assert BOTTLE_CONTEXT.get() == "-"
