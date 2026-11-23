"""Process bourbon bottles in configurable batches.

Handles batch slicing, per-bottle analysis with the same-bottle split check
(the "backstop"), and user feedback integration for iterative refinement.

Same-bottle split check
-----------------------
Photos are grouped by capture time before batching (see
:mod:`image_processor.capture_grouping`), so a bottle never straddles a
batch. Time alone sometimes chains two bottles into one group; the vision
model then answers ``same_bottle=false`` for it. Such a group is split at its
largest capture-time gap and each half is analyzed again:

- If the halves disagree on a field both of them read (name, release, batch,
  proof, age; see :mod:`image_processor.bottle_identity`), the split is kept
  and every half becomes its own bottle (halves the model still calls mixed
  are split again, down to single photos).
- Otherwise the flag is treated as a false alarm: the group stays one bottle
  with a review hint (same-product sibling bottles may be merged), and its
  ``same_bottle`` is cleared to None (checked, not confirmed).
- A half the model still calls mixed is checked again; if every half then
  comes back as one bottle and the two halves still agree, the whole group
  is a false alarm too (an over-eager model does not tear a bottle apart).

Extra analyses are bounded by a run-wide budget computed once over all
groups (:meth:`BatchProcessor.start_run`) and shared by every batch. The
budget counts analyses, not model requests: each analysis may retry an
unusable vision answer once and re-runs OCR on its photos. When the budget is
used up, a flagged group is kept whole with a warning.

Review hints (never merged automatically): a bottle whose photos start at most
:data:`ADJACENT_HINT_MAX_GAP_SECONDS` after the previous bottle's and whose
label agrees with it (:func:`bottle_identity.identity_conflicts` finds
nothing) is marked "possibly the same bottle as <id>". This catches bottles
the ``--group-max-size`` cap or a long pause split apart, but it also fires
for sibling bottles with identical labels, so a human decides. When many of
those hints join groups exactly at the cap size, :meth:`group_size_nudge`
suggests raising it.
"""

import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from image_processor.bottle_analyzer import MIXED_GROUP_WARNING, BottleAnalyzer
from image_processor.bottle_identity import identity_conflicts
from image_processor.capture_grouping import DEFAULT_MAX_GROUP_SIZE, split_at_largest_gap
from lib.exceptions import FATAL_MODEL_ERRORS
from models.bottle import Correction, ExtractedMetadata
from models.image import Image, group_bottle_id

if TYPE_CHECKING:
    from lib.model_selector import ModelSelector

logger = logging.getLogger(__name__)

#: Default share of groups that may get a split check: the run may spend
#: ``ceil(ratio * groups)`` extra analyses (at least
#: :data:`MIN_SPLIT_BUDGET_CALLS`). Each split costs :data:`CALLS_PER_SPLIT`.
#: An analysis is one vision request plus OCR; a retry after an unusable
#: answer is not counted, so model requests can reach twice the budget.
DEFAULT_SPLIT_BUDGET_RATIO = 0.35
#: Largest accepted ``split_budget_ratio``.
MAX_SPLIT_BUDGET_RATIO = 2.0
#: Smallest non-zero budget, so a small run can still check a flagged group.
MIN_SPLIT_BUDGET_CALLS = 4
#: Analyses one split costs (one per half).
CALLS_PER_SPLIT = 2

#: Added when a flagged group cannot be checked because the budget is used up.
SPLIT_BUDGET_WARNING = (
    "Not split: this run's same-bottle split-check budget is used up "
    "(raise --split-budget-ratio to check more groups); review this group"
)
#: Added when re-analyzing the halves of a flagged group failed.
SPLIT_FAILED_WARNING = "Not split: the same-bottle split check failed ({error}); review this group"
#: Added when both halves read as the same bottle (the flag was a false alarm).
FALSE_ALARM_HINT = (
    "The vision model reported different products, but both halves of the group read as the "
    "same bottle, so it was kept as one. Review: sibling bottles of one product (e.g. two "
    "batches without a visible batch number) may be merged here"
)
#: Most seconds from one bottle's last photo to the next bottle's first photo
#: for the "possibly the same bottle" review hint.
ADJACENT_HINT_MAX_GAP_SECONDS = 60.0
#: Share of adjacent-bottle hints touching a group at the size cap from which
#: :meth:`BatchProcessor.group_size_nudge` suggests a larger cap.
GROUP_SIZE_NUDGE_SHARE = 0.25
#: Review hint for adjacent bottles with an identical label (never merged).
ADJACENT_SAME_HINT = (
    "Possibly the same bottle as {bottle_id} (adjacent photos, identical label); "
    "not merged, review"
)
#: Added to each bottle produced by a kept split.
SPLIT_HINT = (
    "Split from a group of {count} photos ({names}) because the vision model reported "
    "different products ({reason}); review the split"
)


def split_budget_for(n_groups: int, ratio: float) -> int:
    """Extra vision calls the split check may spend in a run of ``n_groups`` groups.

    Args:
        n_groups: Number of bottle groups in the whole run.
        ratio: Extra calls per group; 0 or less disables the split check.

    Returns:
        ``max(MIN_SPLIT_BUDGET_CALLS, ceil(ratio * n_groups))``, or 0 when
        disabled or there are no groups.
    """
    if ratio <= 0 or n_groups <= 0:
        return 0
    # round() keeps float noise (0.35 * 20 = 7.000000000000001) from adding a call.
    return max(MIN_SPLIT_BUDGET_CALLS, math.ceil(round(ratio * n_groups, 9)))


def photo_units(images: list[Image]) -> list[list[Image]]:
    """Group photos into units that must stay together when a group is split.

    A unit is a photo plus the exact duplicates and burst retakes
    (``near_duplicate_of``) that follow it.
    """
    units: list[list[Image]] = []
    for img in images:
        if units and (getattr(img, "is_duplicate", False) or getattr(img, "near_duplicate_of", None)):
            units[-1].append(img)
        else:
            units.append([img])
    return units


def split_group_at_largest_gap(images: list[Image]) -> tuple[list[Image], list[Image]]:
    """Split one bottle group in two at its largest capture-time gap.

    Uses :func:`capture_grouping.split_at_largest_gap` (ties go to the gap
    nearest the middle). Duplicates and burst retakes stay with their
    original. Without capture times on every unit the group is split in the
    middle.

    Args:
        images: The group, in capture order, with at least two units.

    Returns:
        (left, right) photo lists, both non-empty.
    """
    units = photo_units(images)
    times: list[float | None] = [getattr(unit[0], "capture_time", None) for unit in units]
    if any(t is None for t in times):
        times = [float(k) for k in range(len(units))]
    left_idx, right_idx = split_at_largest_gap(list(range(len(units))), times)
    return [img for k in left_idx for img in units[k]], [img for k in right_idx for img in units[k]]


def _add_warning(metadata: object, warning: str) -> None:
    """Append a warning to ``metadata.warnings`` once (ignores objects without a list)."""
    warnings = getattr(metadata, "warnings", None)
    if isinstance(warnings, list) and warning not in warnings:
        warnings.append(warning)


@dataclass
class _Analysis:
    """One analyzed bottle: analyzer output plus the photos it covers."""

    bottle_dict: dict
    metadata: ExtractedMetadata
    images: list[Image]
    #: Photos (duplicates excluded) of the capture-time group it came from.
    group_size: int = 0


@dataclass
class BatchResult:
    """Result of processing a single batch.

    A group split by the same-bottle check yields several bottles, so a
    batch can have more ``bottle_dicts`` than groups.

    Attributes:
        batch_id: Sequential batch identifier (1, 2, 3...).
        bottle_dicts: List of processed bottle dictionaries.
        metadata_list: List of extracted metadata for each bottle.
        corrections: List of user corrections applied to this batch.
        failed_bottles: List of bottle IDs that failed processing.
        errors: Why each failed bottle failed, keyed by bottle ID, as
            ``'<ExceptionType>: <message>'``.
    """

    batch_id: int
    bottle_dicts: list[dict] = field(default_factory=list)
    metadata_list: list[ExtractedMetadata] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    failed_bottles: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class BatchProcessor:
    """Process bourbon bottles in configurable batches.

    Manages batch slicing, batch processing orchestration (with the
    same-bottle split check), and user feedback integration.

    Attributes:
        batch_size: Number of bottles per batch, or "all" for unbatched.
        lookahead: Deprecated and ignored (photos are grouped by capture
            time, so a bottle cannot straddle batches).
        bottle_analyzer: BottleAnalyzer instance for processing.
        split_budget_ratio: Extra analyses per group the split check may
            spend over a run (0 disables it).
        max_group_size: The grouping's photos-per-bottle cap (0 = none), used
            only to suggest a larger cap.
        split_stats: Counters over the run. Per group the model flagged:
            'flagged', and its outcome 'split' or 'false_alarm'. Per check
            (nested halves included): 'over_budget', 'failed'. Review hints:
            'adjacent_hint', 'adjacent_hint_at_cap'.
    """

    def __init__(
        self,
        batch_size: int | str = 5,
        lookahead: int = 3,
        bottle_analyzer: Optional[BottleAnalyzer] = None,
        similarity_threshold: float = 0.7,
        model_selector: Optional["ModelSelector"] = None,
        *,
        split_budget_ratio: float = DEFAULT_SPLIT_BUDGET_RATIO,
        max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Number of bottles per batch, or "all" for unbatched.
            lookahead: Deprecated, ignored; kept so old callers still work.
            bottle_analyzer: Optional BottleAnalyzer instance.
            similarity_threshold: Grouping threshold adjusted by
                ``learn_from_feedback`` (0.0-1.0).
            model_selector: Optional ModelSelector for per-stage model resolution.
            split_budget_ratio: Extra analyses per group the same-bottle
                split check may spend over the run, 0 to disable
                (0-``MAX_SPLIT_BUDGET_RATIO``).
            max_group_size: The grouping's photos-per-bottle cap (0 = none);
                see :meth:`group_size_nudge`.

        Raises:
            ValueError: If ``split_budget_ratio`` is out of range.
        """
        if batch_size == "all":
            self.batch_size = batch_size
        else:
            self.batch_size = int(batch_size)
        if not 0 <= split_budget_ratio <= MAX_SPLIT_BUDGET_RATIO:
            message = f"split_budget_ratio must be between 0 and {MAX_SPLIT_BUDGET_RATIO:g}, got {split_budget_ratio}"
            raise ValueError(message)
        self.lookahead = lookahead
        self.bottle_analyzer = bottle_analyzer or BottleAnalyzer()
        self.similarity_threshold = similarity_threshold
        self.model_selector = model_selector
        self.split_budget_ratio = split_budget_ratio
        self.max_group_size = max_group_size
        self.split_stats: Counter[str] = Counter()
        self._previous: _Analysis | None = None
        self._split_budget: int | None = None
        self._split_calls_used = 0
        self._reserved_ids: set[str] = set()

        logger.debug(
            "Initialized BatchProcessor: batch_size=%s, split_budget_ratio=%.2f",
            self.batch_size,
            self.split_budget_ratio,
        )

    def start_run(self, image_groups: list[list[Image]]) -> int:
        """Set the run-wide split-check budget from all of the run's groups.

        Call once, before batching. Every later ``process_batch`` draws from
        this one budget; it is not reset per batch. (``process_batch`` calls
        this with its own batch only if it was never called.)

        Args:
            image_groups: All bottle groups of the run.

        Returns:
            The budget, in extra vision calls.
        """
        self._split_budget = split_budget_for(len(image_groups), self.split_budget_ratio)
        self._split_calls_used = 0
        self._reserved_ids = {group_bottle_id(group) for group in image_groups if group}
        self.split_stats = Counter()
        self._previous = None
        logger.info(
            "Same-bottle split check: up to %d extra analyses for %d groups",
            self._split_budget,
            len(image_groups),
        )
        return self._split_budget

    @property
    def split_budget(self) -> int:
        """The run's split-check budget in extra analyses (0 before ``start_run``)."""
        return self._split_budget or 0

    @property
    def split_calls_used(self) -> int:
        """Extra analyses spent by the split check so far in this run."""
        return self._split_calls_used

    @property
    def split_calls_remaining(self) -> int:
        """Extra analyses the split check may still spend in this run."""
        return max(0, self.split_budget - self._split_calls_used)

    def split_summary(self) -> str:
        """One-line summary of the split check over the run, for the final log."""
        stats = self.split_stats
        return (
            f"Same-bottle split check: {stats['flagged']} group(s) flagged as mixed, "
            f"{stats['split']} split, {stats['false_alarm']} kept as false alarms; "
            f"{stats['over_budget']} check(s) skipped (budget), {stats['failed']} check(s) failed; "
            f"{self._split_calls_used}/{self.split_budget} extra analyses; "
            f"{stats['adjacent_hint']} 'possibly the same bottle' review hint(s)"
        )

    def group_size_nudge(self) -> str | None:
        """Suggest a larger ``--group-max-size`` when the cap seems to split bottles.

        Returns:
            A message when at least :data:`GROUP_SIZE_NUDGE_SHARE` of this
            run's "possibly the same bottle" hints join a group of exactly
            ``max_group_size`` photos, else None.
        """
        hints, at_cap = self.split_stats["adjacent_hint"], self.split_stats["adjacent_hint_at_cap"]
        cap = self.max_group_size
        if cap <= 0 or hints == 0 or at_cap < GROUP_SIZE_NUDGE_SHARE * hints:
            return None
        return (
            f"{at_cap} of {hints} 'possibly the same bottle' hint(s) join a group of exactly {cap} "
            f"photos (the --group-max-size cap); if you take more than {cap} photos of a bottle, "
            f"consider --group-max-size {cap + 1}"
        )

    def slice_into_batches(
        self,
        image_groups: list[list[Image]],
    ) -> list[list[list[Image]]]:
        """Split bottle image groups into batches.

        Args:
            image_groups: List of bottle image groups from ImageOrganizer.

        Returns:
            List of batches, where each batch is a list of image groups.
        """
        if self.batch_size == 0 or str(self.batch_size).lower() == "all":
            logger.info("Batch mode: processing all images in single batch")
            return [image_groups]

        batches: list[list[list[Image]]] = []

        for i in range(0, len(image_groups), self.batch_size):
            batch = image_groups[i : i + self.batch_size]
            batches.append(batch)
            logger.debug("Created batch %d: %d image groups", len(batches), len(batch))

        logger.info("Split %d image groups into %d batches", len(image_groups), len(batches))
        return batches

    def apply_lookahead(
        self,
        current_batch: list[list[Image]],
        remaining_groups: list[list[Image]],
    ) -> tuple[list[list[Image]], list[list[Image]]]:
        """Deprecated no-op: returns both arguments unchanged.

        Lookahead used to pull visually similar groups from later batches
        into the current one. Photos are now grouped by capture time before
        batching, so a bottle never straddles a batch, and the ORB similarity
        it relied on matched different bottles as often as the same one.
        Kept for one release so old callers do not break.
        """
        logger.debug("apply_lookahead is deprecated and does nothing")
        return current_batch, remaining_groups

    def process_batch(
        self,
        batch: list[list[Image]],
        vision_model: str | None = None,
    ) -> BatchResult:
        """Process one batch of bottle image groups.

        Each group is analyzed once; a group the vision model reports as
        mixed may be split (see the module docstring), adding one bottle per
        part. Split parts get their own bottle ID (the first part keeps the
        group's) when the run assigned IDs (``Image.bottle_ref``).

        A bottle whose analysis fails is recorded in ``failed_bottles`` (with
        its reason in ``errors``) and the batch continues. Errors after which
        no bottle can succeed (``FATAL_MODEL_ERRORS``: model unusable, server
        unreachable) are re-raised so the caller can abort the run.

        Args:
            batch: List of image groups to process.
            vision_model: Vision model name; if None, the analyzer resolves it
                (model selector, then the default vision model).

        Returns:
            BatchResult with processed bottles and metadata.

        Raises:
            VisionModelUnavailableError, OllamaConnectionError, ModelServerError:
                Re-raised unchanged from the analyzer.
        """
        batch_id = id(batch)
        logger.info("Processing batch with %d image groups", len(batch))

        # Pass model_selector to bottle_analyzer if available (T164)
        if self.model_selector is not None:
            self.bottle_analyzer.model_selector = self.model_selector
        if self._split_budget is None:
            logger.debug("start_run() was not called; split-check budget computed from this batch")
            self.start_run(batch)

        result = BatchResult(batch_id=batch_id)

        for i, image_group in enumerate(batch, 1):
            # Run-unique ID assigned by the caller, else the first photo's stem.
            bottle_id = group_bottle_id(image_group)
            logger.debug("Processing bottle %d/%d: %s", i, len(batch), bottle_id)

            try:
                analyses = self._analyze_group(image_group, vision_model)
            except FATAL_MODEL_ERRORS:
                raise
            except Exception as e:
                logger.error("Failed to process bottle %s: %s: %s", bottle_id, type(e).__name__, e)
                logger.debug("Traceback for bottle %s", bottle_id, exc_info=True)
                result.failed_bottles.append(bottle_id)
                result.errors[bottle_id] = f"{type(e).__name__}: {e}"
                self._previous = None  # a failed bottle sits between its neighbours
                continue

            self._assign_bottle_ids(image_group, analyses)
            for analysis in analyses:
                self._hint_adjacent(analysis)
                result.bottle_dicts.append(analysis.bottle_dict)
                result.metadata_list.append(analysis.metadata)
            logger.info("Successfully processed bottle: %s", bottle_id)

        logger.info(
            "Batch processing complete: %d successful, %d failed",
            len(result.bottle_dicts),
            len(result.failed_bottles),
        )

        return result

    def _analyze(self, images: list[Image], vision_model: str | None, group_size: int = 0) -> _Analysis:
        """One analysis (one vision request, plus a retry if unusable, and OCR) for a list of photos."""
        bottle_dict, metadata = self.bottle_analyzer.process_single_bottle(images, vision_model=vision_model)
        return _Analysis(bottle_dict, metadata, list(images), group_size)

    def _analyze_group(self, group: list[Image], vision_model: str | None) -> list[_Analysis]:
        """Analyze one group and run the same-bottle split check on it.

        Returns:
            One analysis per resulting bottle, in capture order (the first
            contains the group's first photo).
        """
        group_size = sum(1 for img in group if not getattr(img, "is_duplicate", False))
        return self._resolve_mixed(self._analyze(group, vision_model, group_size), vision_model)

    def _hint_adjacent(self, analysis: _Analysis) -> None:
        """Add the "possibly the same bottle" review hint against the previous bottle.

        Fires when this bottle's first photo was taken at most
        :data:`ADJACENT_HINT_MAX_GAP_SECONDS` after the previous bottle's last
        one and both read a name and no field conflicts. Never merges.
        """
        previous, self._previous = self._previous, analysis
        if previous is None:
            return
        prev_times = [t for t in (getattr(i, "capture_time", None) for i in previous.images) if t is not None]
        cur_times = [t for t in (getattr(i, "capture_time", None) for i in analysis.images) if t is not None]
        if not prev_times or not cur_times:
            return
        gap = min(cur_times) - max(prev_times)
        if not 0 <= gap <= ADJACENT_HINT_MAX_GAP_SECONDS:
            return
        names = [getattr(a.metadata, "name", None) for a in (previous, analysis)]
        if not all(isinstance(n, str) and n.strip() for n in names):
            return  # an unread name would match anything
        if identity_conflicts(previous.metadata, analysis.metadata):
            return
        previous_id = previous.bottle_dict.get("bottle_id") or group_bottle_id(previous.images)
        logger.info(
            "Review hint: %s may be the same bottle as %s (%.0f s apart, identical label)",
            analysis.bottle_dict.get("bottle_id") or group_bottle_id(analysis.images),
            previous_id,
            gap,
        )
        _add_warning(analysis.metadata, ADJACENT_SAME_HINT.format(bottle_id=previous_id))
        self.split_stats["adjacent_hint"] += 1
        cap = self.max_group_size
        if cap > 0 and cap in (previous.group_size, analysis.group_size):
            self.split_stats["adjacent_hint_at_cap"] += 1

    def _resolve_mixed(self, analysis: _Analysis, vision_model: str | None, depth: int = 0) -> list[_Analysis]:
        """Split an analysis the model called mixed, if the halves really differ.

        Recurses into halves the model still calls mixed; stops at single
        photos (a photo with its duplicates/burst retakes) and when the
        run's budget cannot pay for another split. If the halves agree and
        each comes back as one bottle (after its own check, if it was still
        called mixed), the whole group is kept as one bottle.

        Args:
            analysis: The analysis to check.
            vision_model: Vision model name (None: the analyzer resolves it).
            depth: 0 for a whole group, more for its halves (only depth 0
                counts toward the per-group stats).
        """
        if getattr(analysis.metadata, "same_bottle", None) is not False or self.split_budget_ratio <= 0:
            return [analysis]
        if len(photo_units(analysis.images)) < 2:  # noqa: PLR2004 - nothing left to split
            return [analysis]
        names = ", ".join(Path(img.path).name for img in analysis.images)
        if depth == 0:
            self.split_stats["flagged"] += 1
        if self.split_calls_remaining < CALLS_PER_SPLIT:
            self.split_stats["over_budget"] += 1
            logger.warning("Not splitting mixed group (%s): split-check budget used up", names)
            _add_warning(analysis.metadata, SPLIT_BUDGET_WARNING)
            return [analysis]

        left, right = split_group_at_largest_gap(analysis.images)
        self._split_calls_used += CALLS_PER_SPLIT
        logger.info(
            "Mixed group (%s): checking halves %s | %s",
            names,
            [Path(img.path).name for img in left],
            [Path(img.path).name for img in right],
        )
        try:
            halves = [
                self._analyze(left, vision_model, analysis.group_size),
                self._analyze(right, vision_model, analysis.group_size),
            ]
        except FATAL_MODEL_ERRORS:
            raise
        except Exception as e:
            self.split_stats["failed"] += 1
            logger.warning("Split check failed for %s: %s: %s; keeping one bottle", names, type(e).__name__, e)
            _add_warning(analysis.metadata, SPLIT_FAILED_WARNING.format(error=f"{type(e).__name__}: {e}"))
            return [analysis]

        conflicts = identity_conflicts(halves[0].metadata, halves[1].metadata)
        reason = f"{', '.join(conflicts)} differ" if conflicts else "a half still shows different products"
        hint = SPLIT_HINT.format(count=len(analysis.images), names=names, reason=reason)
        for half in halves:
            _add_warning(half.metadata, hint)  # discarded again if this turns out a false alarm
        parts = [self._resolve_mixed(half, vision_model, depth + 1) for half in halves]

        if not conflicts and all(
            len(part) == 1 and getattr(part[0].metadata, "same_bottle", None) is not False for part in parts
        ):
            # Neither half is (still) mixed and they agree: the flag was a false alarm.
            if depth == 0:
                self.split_stats["false_alarm"] += 1
            logger.info("Halves of %s read as the same bottle; keeping one bottle", names)
            warnings = getattr(analysis.metadata, "warnings", None)
            if isinstance(warnings, list) and MIXED_GROUP_WARNING in warnings:
                warnings.remove(MIXED_GROUP_WARNING)
            if hasattr(analysis.metadata, "same_bottle"):
                analysis.metadata.same_bottle = None  # checked: not confirmed as mixed
            _add_warning(analysis.metadata, FALSE_ALARM_HINT)
            return [analysis]

        if depth == 0:
            self.split_stats["split"] += 1
        logger.info("Splitting %s: %s", names, reason)
        return [*parts[0], *parts[1]]

    def _assign_bottle_ids(self, group: list[Image], analyses: list[_Analysis]) -> None:
        """Set ``bottle_id`` on each result when the run assigned group IDs.

        The first result keeps the group's ID; further results (split parts)
        get their first photo's stem, made unique within the run, and their
        photos' ``bottle_ref`` is updated to match.
        """
        if not group or not group[0].bottle_ref:
            return
        group_id = group_bottle_id(group)
        for k, analysis in enumerate(analyses):
            bottle_id = group_id if k == 0 else self._new_bottle_id(analysis.images)
            if k:
                for img in analysis.images:
                    img.bottle_ref = bottle_id
            analysis.bottle_dict["bottle_id"] = bottle_id

    def _new_bottle_id(self, images: list[Image]) -> str:
        """Run-unique ID for a split part: its first photo's stem (+ path hash if taken)."""
        first = Path(images[0].path)
        candidate = first.stem
        if candidate in self._reserved_ids:
            digest = hashlib.sha1(str(first.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
            candidate = f"{first.stem}_{digest}"
        base, n = candidate, 2
        while candidate in self._reserved_ids:
            candidate = f"{base}_{n}"
            n += 1
        self._reserved_ids.add(candidate)
        return candidate

    def learn_from_feedback(
        self,
        corrections: list[Correction],
    ) -> None:
        """Learn from user corrections to improve future grouping.

        Analyzes user corrections to adjust heuristics for subsequent batches.
        Adjusts similarity_threshold based on merge/split correction patterns.

        Args:
            corrections: List of user corrections from feedback.
        """
        if not corrections:
            return

        logger.info("Learning from %d corrections", len(corrections))

        # Analyze correction patterns
        merge_corrections = [
            c for c in corrections if c.field_name == "grouping" and c.corrected_value == "merge"
        ]
        split_corrections = [
            c for c in corrections if c.field_name == "grouping" and c.corrected_value == "split"
        ]

        # Adjust similarity threshold based on correction patterns
        merge_count = len(merge_corrections)
        split_count = len(split_corrections)

        if merge_count > split_count:
            # Too many splits - we're being too aggressive about grouping
            # Tighten threshold to reduce false positives
            adjustment = 0.05 * min(merge_count, 3)  # Cap at 0.15 adjustment
            new_threshold = min(1.0, self.similarity_threshold + adjustment)
            logger.info(
                "Tightening similarity threshold: %.2f -> %.2f (based on %d merge corrections)",
                self.similarity_threshold,
                new_threshold,
                merge_count,
            )
            self.similarity_threshold = new_threshold
        elif split_count > merge_count:
            # Too many merges - we're being too conservative about grouping
            # Loosen threshold to catch more related items
            adjustment = 0.05 * min(split_count, 3)  # Cap at 0.15 adjustment
            new_threshold = max(0.3, self.similarity_threshold - adjustment)
            logger.info(
                "Loosening similarity threshold: %.2f -> %.2f (based on %d split corrections)",
                self.similarity_threshold,
                new_threshold,
                split_count,
            )
            self.similarity_threshold = new_threshold
        else:
            logger.debug(
                "Balanced feedback: %d merges, %d splits - no threshold change",
                merge_count,
                split_count,
            )

    def apply_corrections(
        self,
        bottle_dicts: list[dict],
        corrections: list[Correction],
    ) -> list[dict]:
        """Apply user corrections to bottle dictionaries.

        Args:
            bottle_dicts: List of bottle dictionaries from batch result.
            corrections: List of corrections to apply.

        Returns:
            Updated list of bottle dictionaries with corrections applied.
        """
        if not corrections:
            return bottle_dicts

        logger.info("Applying %d corrections to %d bottles", len(corrections), len(bottle_dicts))

        # Apply corrections to matching bottles
        for correction in corrections:
            # Find bottle by path or name (simplified matching)
            matching_bottle = self._find_matching_bottle(bottle_dicts, correction)

            if matching_bottle:
                # Apply field correction
                original_value = matching_bottle.get(correction.field_name)
                matching_bottle[correction.field_name] = correction.corrected_value

                # Store correction history
                if "corrections" not in matching_bottle:
                    matching_bottle["corrections"] = []

                matching_bottle["corrections"].append(
                    {
                        "field_name": correction.field_name,
                        "original_value": original_value,
                        "corrected_value": correction.corrected_value,
                        "corrected_at": correction.corrected_at.isoformat() if correction.corrected_at else None,
                    }
                )

                logger.info(
                    "Applied correction to bottle: %s = %s (was %s)",
                    correction.field_name,
                    correction.corrected_value,
                    original_value,
                )

        return bottle_dicts

    def _find_matching_bottle(
        self,
        bottle_dicts: list[dict],
        correction: Correction,
    ) -> Optional[dict]:
        """Find matching bottle dictionary for a correction.

        Simple matching based on bottle path or name. In production,
        could use more sophisticated matching.

        Args:
            bottle_dicts: List of bottle dictionaries.
            correction: Correction to match.

        Returns:
            Matching bottle dictionary or None.
        """
        # Try to match by path if available in correction metadata
        if hasattr(correction, "bottle_path"):
            for bottle in bottle_dicts:
                if "images" in bottle and correction.bottle_path in bottle["images"]:
                    return bottle

        # Try to match by name
        if hasattr(correction, "bottle_name"):
            for bottle in bottle_dicts:
                if bottle.get("name") == correction.bottle_name:
                    return bottle

        # If no explicit identifiers, match by field value
        # First find any bottle where the field matches the original_value
        for bottle in bottle_dicts:
            if bottle.get(correction.field_name) == correction.original_value:
                return bottle

        # Also try to match where the field matches the corrected_value
        # This handles sequential corrections on the same field
        for bottle in bottle_dicts:
            if bottle.get(correction.field_name) == correction.corrected_value:
                return bottle

        return None
