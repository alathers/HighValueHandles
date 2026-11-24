"""Structured per-bottle hints written by the collection owner (``--hints``).

A hints file is TOML::

    [bottle.IMG_4643]
    note = "Batch number is on the neck label in the third photo."
    source = "Owner inspected the bottle, 2026-11-20"
    [bottle.IMG_4643.fields]
    batch_number = "C923"

- ``note`` is reading guidance only. It goes into the vision prompt in a
  clearly delimited section the model must never copy values from.
- ``fields`` are values a person established. They are applied after
  analysis as corrections whose ``source`` names the hints file, the bottle
  and the person's ``source`` text (required whenever ``fields`` is given).
  A hint wins over the label reading, but any disagreement is kept as a
  review warning naming both values; it is never silently dropped.

The book is read-only after loading, so worker threads may share it.
"""

from __future__ import annotations

import logging
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from lib.settings import SettingsError

logger = logging.getLogger(__name__)

#: Longest note passed to the vision prompt (characters).
NOTE_MAX_CHARS = 500

#: Fields a hint may set.
HINT_FIELDS: tuple[str, ...] = (
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
)
_NUMERIC_FIELDS = frozenset({"proof", "abv", "msrp"})
_BOTTLE_KEYS = frozenset({"note", "source", "fields"})
_MAX_PROOF = 200
_MAX_ABV = 100
_MAX_MSRP = 100_000
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class BottleHint:
    """Hints for one bottle."""

    bottle_id: str
    note: str = ""
    source: str = ""
    fields: Mapping[str, Any] = field(default_factory=dict)


def _field_error(name: str, value: Any) -> str | None:  # noqa: ANN401, PLR0911
    if name not in HINT_FIELDS:
        return f"unknown field '{name}' (allowed: {', '.join(HINT_FIELDS)})"
    if name in _NUMERIC_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{name} must be a number"
        limit = {"proof": _MAX_PROOF, "abv": _MAX_ABV, "msrp": _MAX_MSRP}[name]
        if not 0 < value <= limit:
            return f"{name} must be > 0 and <= {limit}"
        return None
    if name == "release_year":
        if isinstance(value, bool) or not isinstance(value, int) or not 1000 <= value <= 9999:  # noqa: PLR2004
            return "release_year must be a 4-digit integer"
        return None
    if not isinstance(value, str) or not value.strip():
        return f"{name} must be a non-empty string"
    return None


def format_hint_value(name: str, value: Any) -> str:  # noqa: ANN401
    """A hint value in the bottle dict's format ('101 Proof', '50.5%', '59.99', '2023')."""
    if name == "proof":
        return f"{float(value):g} Proof"
    if name == "abv":
        return f"{float(value):g}%"
    if name == "msrp":
        return str(Decimal(str(value)).quantize(Decimal("0.01")))
    if name == "release_year":
        return str(int(value))
    return " ".join(str(value).split())


def _number(value: Any) -> float | None:  # noqa: ANN401
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def _same_value(name: str, label: Any, hint: str) -> bool:  # noqa: ANN401
    if name in _NUMERIC_FIELDS or name == "release_year":
        a, b = _number(label), _number(hint)
        return a is not None and b is not None and abs(a - b) < 1e-9  # noqa: PLR2004
    return " ".join(str(label).split()).casefold() == " ".join(hint.split()).casefold()


@dataclass(frozen=True)
class HintBook:
    """All hints of one file, keyed by bottle ID (read-only)."""

    path: Path | None = None
    hints: Mapping[str, BottleHint] = field(default_factory=dict)

    EMPTY: ClassVar[HintBook]

    @classmethod
    def load(cls, path: Path) -> HintBook:
        """Read and validate a hints file.

        Raises:
            SettingsError: Listing every problem (bad TOML, unknown keys or
                fields, values of the wrong type, ``fields`` without ``source``).
        """
        path = Path(path)
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError:
            raise SettingsError([f"hints file not found: {path}"]) from None
        except tomllib.TOMLDecodeError as e:
            raise SettingsError([f"hints file {path}: invalid TOML: {e}"]) from None
        except OSError as e:
            raise SettingsError([f"hints file {path}: cannot read: {e}"]) from None

        errors: list[str] = []
        for key in data:
            if key != "bottle":
                errors.append(f"hints file {path}: unknown table [{key}] (only [bottle.<id>] tables are allowed)")
        bottles = data.get("bottle", {})
        if not isinstance(bottles, dict):
            errors.append(f"hints file {path}: 'bottle' must be a table of [bottle.<id>] tables")
            bottles = {}
        hints: dict[str, BottleHint] = {}
        for bottle_id, entry in bottles.items():
            where = f"hints file {path} [bottle.{bottle_id}]"
            if not isinstance(entry, dict):
                errors.append(f"{where}: must be a table")
                continue
            for key in entry:
                if key not in _BOTTLE_KEYS:
                    errors.append(f"{where}: unknown key '{key}' (allowed: note, source, fields)")
            note = entry.get("note", "")
            source = entry.get("source", "")
            raw_fields = entry.get("fields", {})
            if not isinstance(note, str):
                errors.append(f"{where}: note must be a string")
                note = ""
            if not isinstance(source, str):
                errors.append(f"{where}: source must be a string")
                source = ""
            if not isinstance(raw_fields, dict):
                errors.append(f"{where}: fields must be a table")
                raw_fields = {}
            if raw_fields and not source.strip():
                errors.append(
                    f"{where}: 'source' is required with [fields] (who established the values, and how)"
                )
            for name, value in raw_fields.items():
                problem = _field_error(name, value)
                if problem:
                    errors.append(f"{where}.fields: {problem}")
            hints[str(bottle_id)] = BottleHint(
                bottle_id=str(bottle_id),
                note=note.strip(),
                source=source.strip(),
                fields=MappingProxyType(dict(raw_fields)),
            )
        if errors:
            raise SettingsError(errors)
        logger.info("Loaded hints for %d bottle(s) from %s", len(hints), path)
        return cls(path=path, hints=MappingProxyType(hints))

    def __len__(self) -> int:
        return len(self.hints)

    def note_for(self, bottle_id: str) -> str:
        """The bottle's reading note ('' if none), at most :data:`NOTE_MAX_CHARS` characters."""
        hint = self.hints.get(bottle_id)
        if hint is None or not hint.note:
            return ""
        note = " ".join(hint.note.split())
        return note[:NOTE_MAX_CHARS]

    def fields_for(self, bottle_id: str) -> Mapping[str, Any]:
        """The bottle's field values (empty if none)."""
        hint = self.hints.get(bottle_id)
        return hint.fields if hint is not None else MappingProxyType({})

    def source_for(self, bottle_id: str) -> str:
        """Who established the bottle's field values ('' if none)."""
        hint = self.hints.get(bottle_id)
        return hint.source if hint is not None else ""

    def unused(self, seen_ids: Iterable[str]) -> list[str]:
        """Hint bottle IDs that match none of ``seen_ids`` (typos, or bottles not in this run)."""
        seen = set(seen_ids)
        return sorted(k for k in self.hints if k not in seen)


HintBook.EMPTY = HintBook()


def apply_hint_overrides(
    bottle_dicts: list[dict],
    metadata_list: list,
    hints: HintBook,
    hints_path: Path | None = None,
) -> tuple[list, list]:
    """Apply the hints' field values to the bottles they name, as sourced corrections.

    Runs on the main thread after analysis. For each hinted field a
    ``Correction(source='hint:<file>#<bottle_id>: <source>')`` is applied
    through :func:`image_processor.corrections.apply_corrections` (by bottle
    ID only). When the label reading (the metadata) holds a different
    non-null value, the warning ``hint overrides label read for <field>:
    label '<x>', hint '<y>'`` is added to that bottle's metadata.

    Returns:
        ``(applied, unapplied)``: the corrections applied, and
        ``(correction, reason)`` pairs that were not.
    """
    from image_processor.corrections import apply_corrections
    from image_processor.preview_generator import PreviewGenerator
    from models.bottle import Correction

    if not hints.hints:
        return [], []
    path = Path(hints_path) if hints_path is not None else hints.path
    file_name = path.name if path is not None else "hints"
    corrections: list[Correction] = []
    for index, bottle_dict in enumerate(bottle_dicts):
        metadata = metadata_list[index] if index < len(metadata_list) else None
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        values = hints.fields_for(bottle_id)
        if not values:
            continue
        source = hints.source_for(bottle_id)
        for name, raw in values.items():
            hint_value = format_hint_value(name, raw)
            current = bottle_dict.get(name)
            label = getattr(metadata, name, None) if metadata is not None else None
            if label not in (None, "") and not _same_value(name, label, hint_value):
                warning = f"hint overrides label read for {name}: label '{label}', hint '{hint_value}'"
                warnings = getattr(metadata, "warnings", None)
                if isinstance(warnings, list) and warning not in warnings:
                    warnings.append(warning)
                logger.warning("Bottle %s: %s", bottle_id, warning)
            corrections.append(
                Correction(
                    bottle_id=bottle_id,
                    field_name=name,
                    original_value="" if current is None else str(current),
                    corrected_value=hint_value,
                    source=f"hint:{file_name}#{bottle_id}: {source}",
                )
            )
    if not corrections:
        return [], []
    _, unapplied = apply_corrections(bottle_dicts, metadata_list, corrections)
    failed = {id(c) for c, _ in unapplied}
    applied = [c for c in corrections if id(c) not in failed]
    for correction, reason in unapplied:
        logger.error(
            "Hint not applied (%s): bottle %s, %s -> %r",
            reason, correction.bottle_id, correction.field_name, correction.corrected_value,
        )
    if applied:
        logger.info("Applied %d hint value(s) from %s", len(applied), file_name)
    return applied, unapplied
