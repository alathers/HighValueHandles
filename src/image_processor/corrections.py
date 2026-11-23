"""Apply human corrections to analyzed bottles, by bottle ID only.

A correction (:class:`models.bottle.Correction`) names the bottle it is for
(``bottle_id``), the field, the value the person saw (``original_value``) and
the new value. It is applied only when:

- a bottle with that exact ID is in the list (IDs come from
  :meth:`PreviewGenerator.bottle_id_for`, the same IDs interactive review
  shows and failure reports use), and
- the field is one of :data:`CORRECTABLE_FIELDS` (a typo such as ``"nmae"``
  would otherwise "apply" to a key nothing reads, and the person would
  believe the bottle was fixed), and
- that bottle's field still holds ``original_value`` (compared as stripped
  ``str()`` forms; a missing value counts as ``""``).

Anything else is returned unapplied with a reason, and no other bottle is
ever touched: a correction is never matched by field value, so an edit to
one bottle cannot change a different bottle that happens to share the value.
"""

from __future__ import annotations

import logging
from typing import Any

from image_processor.preview_generator import PreviewGenerator
from models.bottle import Correction

logger = logging.getLogger(__name__)

#: Reason returned for a correction whose ``bottle_id`` matches no bottle.
UNKNOWN_BOTTLE_ID = "unknown bottle_id"
#: Reason returned when the bottle no longer holds ``original_value``.
STALE_ORIGINAL_VALUE = "stale original value"
#: Reason returned for a correction of a field not in :data:`CORRECTABLE_FIELDS`.
UNKNOWN_FIELD = "field is not correctable"

#: Bottle fields a person may correct (interactive edit, corrections file,
#: retry ``suggested_fix``, hint overrides). Anything else (``corrections``,
#: ``bottle_id``, ``images``, ``confidence``, misspelled names, ...) is refused.
CORRECTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "distillery",
        "bottler",
        "specific_release",
        "batch_number",
        "proof",
        "abv",
        "age",
        "release_year",
        "msrp",
        "tasting_notes",
    }
)


def _comparable(value: Any) -> str:  # noqa: ANN401
    """``str(value).strip()``, with None as ``""``."""
    return "" if value is None else str(value).strip()


def apply_corrections(
    bottle_dicts: list[dict],
    metadata_list: list,
    corrections: list[Correction],
) -> tuple[list[dict], list[tuple[Correction, str]]]:
    """Apply each correction to the one bottle it names.

    Corrections are applied in order, so a later correction of the same field
    must give the earlier one's ``corrected_value`` as its ``original_value``.
    Each applied correction is appended (``Correction.to_dict()``) to that
    bottle's ``corrections`` list.

    Args:
        bottle_dicts: Analyzer output dicts (modified in place).
        metadata_list: The matching ExtractedMetadata, same order (may be
            shorter, e.g. a batch JSON without metadata).
        corrections: Corrections to apply.

    Returns:
        ``(bottle_dicts, unapplied)``: the same list, and every correction
        that was not applied paired with the reason
        (:data:`UNKNOWN_BOTTLE_ID`, :data:`UNKNOWN_FIELD` or
        :data:`STALE_ORIGINAL_VALUE`).
    """
    by_id: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for index, bottle_dict in enumerate(bottle_dicts):
        metadata = metadata_list[index] if index < len(metadata_list) else None
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        if bottle_id in by_id:
            duplicate_ids.add(bottle_id)
        by_id[bottle_id] = bottle_dict
    for bottle_id in duplicate_ids:
        # Two bottles share an ID: a correction could not say which it means.
        logger.warning("Bottle ID %s is not unique in this batch; its corrections are not applied", bottle_id)
        by_id.pop(bottle_id, None)

    unapplied: list[tuple[Correction, str]] = []
    for correction in corrections:
        bottle_dict = by_id.get(correction.bottle_id)
        if bottle_dict is None:
            logger.warning(
                "Correction not applied: no bottle with ID %s (%s -> %r)",
                correction.bottle_id, correction.field_name, correction.corrected_value,
            )
            unapplied.append((correction, UNKNOWN_BOTTLE_ID))
            continue
        if correction.field_name not in CORRECTABLE_FIELDS:
            logger.warning(
                "Correction not applied to %s: %r is not a correctable field (allowed: %s)",
                correction.bottle_id, correction.field_name, ", ".join(sorted(CORRECTABLE_FIELDS)),
            )
            unapplied.append((correction, UNKNOWN_FIELD))
            continue
        current = bottle_dict.get(correction.field_name)
        if _comparable(current) != _comparable(correction.original_value):
            logger.warning(
                "Correction not applied to %s: %s is now %r, not the %r the correction expects",
                correction.bottle_id, correction.field_name, current, correction.original_value,
            )
            unapplied.append((correction, STALE_ORIGINAL_VALUE))
            continue
        bottle_dict[correction.field_name] = correction.corrected_value
        recorded = bottle_dict.get("corrections")
        if not isinstance(recorded, list):
            recorded = []
            bottle_dict["corrections"] = recorded
        recorded.append(correction.to_dict())
        logger.info(
            "Applied correction to %s: %s = %r (was %r; source %s)",
            correction.bottle_id, correction.field_name, correction.corrected_value, current,
            correction.source,
        )
    return bottle_dicts, unapplied
