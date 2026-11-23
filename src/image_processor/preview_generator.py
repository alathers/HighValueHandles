"""Generate preview reports and output files for bourbon bottles.

Creates markdown output, saves images, writes per-bottle failure reports,
and provides preview for user review.
"""

import json
import logging
import os
import re
import shutil
import traceback
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from image_processor.image_organizer import OUTPUT_MARKER
from lib.error_handling import create_error_report
from lib.exceptions import FileSystemError
from models.bottle import Bottle, ExtractedMetadata, format_money, parse_money

logger = logging.getLogger(__name__)

#: Version of the per-bottle JSON and failure-report layouts.
OUTPUT_SCHEMA_VERSION = 1

#: Highest numeric suffix tried before giving up on a unique output name.
_MAX_NAME_SUFFIX = 10_000

_PLACEHOLDER_NAMES = {"", "unknown", "unknown bottle", "n/a", "none"}


def slugify(text: Optional[str], fallback: str = "bottle", max_len: int = 60) -> str:
    """Turn text into a safe lowercase file/folder name.

    Only ``[a-z0-9_]`` survive, so the result can never contain a path
    separator, ``..`` or characters that some filesystems reject, and two
    names differing only in case map to the same slug (safe on
    case-insensitive filesystems).

    Examples: ``"Blanton's: Gold/Edition? <1>"`` -> ``'blantons_gold_edition_1'``,
    ``'../../etc'`` -> ``'etc'``.

    Args:
        text: Text to convert (None is treated as empty).
        fallback: Returned when nothing usable is left.
        max_len: Maximum length of the slug.

    Returns:
        The slug, or ``fallback``.
    """
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"['`\u2019]", "", ascii_text).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    slug = slug[:max_len].rstrip("_")
    return slug or fallback


def _is_placeholder_name(name: Optional[str]) -> bool:
    """True for empty names and 'Unknown ...' placeholders from the analyzer."""
    normalized = (name or "").strip().lower()
    return normalized in _PLACEHOLDER_NAMES or normalized.startswith("unknown")


def _unique_path(directory: Path, stem: str, suffix: str = "") -> Path:
    """First of ``stem``, ``stem_2``, ``stem_3``... (plus suffix) not in ``directory``.

    Raises:
        FileExistsError: If every candidate up to ``_MAX_NAME_SUFFIX`` exists.
    """
    for n in range(1, _MAX_NAME_SUFFIX + 1):
        candidate = directory / (f"{stem}{suffix}" if n == 1 else f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No free name for '{stem}{suffix}' in {directory}")


class PreviewGenerator:
    """Generate preview reports and output files for bourbon bottles.

    Handles markdown generation, file saving, and error report creation.

    Attributes:
        output_dir: Base output directory for processed bottles.
    """

    def __init__(self, output_dir: Path, *, mark_output_dir: bool = True):
        """Initialize preview generator.

        Args:
            output_dir: Base output directory (required; created if missing).
            mark_output_dir: Write the output marker into ``output_dir`` so
                later scans skip it. Off when ``output_dir`` contains the
                photos being processed (each bottle folder is still marked).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mark_output_dir = mark_output_dir

    def generate_bottle_markdown(
        self,
        bottle: Bottle,
        image_links: Optional[List[str]] = None,
        model_used: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> str:
        """Generate markdown content for bottle.

        Args:
            bottle: Bottle model instance (metadata may be None).
            image_links: Relative paths of the copied images (e.g.
                ``'images/IMG_1.jpg'``); rendered as links, the first marked
                primary. Default: the names of ``bottle.metadata.source_images``.
            model_used: Vision model that extracted the data.
            confidence: Extraction confidence ('HIGH'/'MEDIUM'/'LOW').

        Returns:
            Markdown formatted string.
        """
        metadata = bottle.metadata
        lines = [f"# {bottle.name}", ""]

        # Distillery
        lines.append("## Distillery")
        lines.append(f"{bottle.distillery}")

        # MSRP
        lines.append("\n## MSRP")
        lines.append(format_money(bottle.msrp))

        # Release Information section
        lines.append("\n## Release Information")

        if bottle.proof:
            lines.append(f"- **Proof:** {bottle.proof}")

        if bottle.abv:
            lines.append(f"- **ABV:** {bottle.abv}")

        if bottle.age:
            lines.append(f"- **Age:** {bottle.age}")

        if bottle.specific_release:
            lines.append(f"- **Release:** {bottle.specific_release}")

        if metadata is not None and metadata.batch_number:
            lines.append(f"- **Batch:** {metadata.batch_number}")

        # Tasting Notes
        if bottle.tasting_notes:
            lines.append("\n## Tasting Notes")
            lines.append(bottle.tasting_notes)

        # Reference Links
        if bottle.reference_links:
            lines.append("\n## References")
            for ref in bottle.reference_links:
                lines.append(f"[{ref.label}]({ref.url})")

        # Quantity section
        lines.append("\n## Quantity")
        lines.append(f"Total quantity: {bottle.quantity.count}")
        total_value = bottle.total_value
        if bottle.quantity.count > 1 and total_value is not None:
            unit_value = total_value / bottle.quantity.count
            lines.append(f"Value per bottle: {format_money(unit_value)}")
        lines.append(f"Total value: {format_money(total_value)}")

        # Images
        lines.append("\n## Images")
        if image_links is not None:
            for i, link in enumerate(image_links, 1):
                primary = " (primary)" if i == 1 else ""
                lines.append(f"{i}. [{Path(link).name}]({link}){primary}")
            image_count = len(image_links)
        else:
            source_images = list(metadata.source_images) if metadata is not None else []
            for i, img in enumerate(source_images, 1):
                lines.append(f"{i}. {Path(img).name}")
            image_count = len(source_images)

        # Processing Notes
        if bottle.corrections:
            lines.append("\n## Corrections")
            for correction in bottle.corrections:
                lines.append(f"- **{correction.field_name}**: `{correction.original_value}` → `{correction.corrected_value}`")

        # Processing Details
        lines.append("\n## Processing Details")
        lines.append(f"- **Images Processed**: {image_count}")
        lines.append(f"- **Model**: {model_used or 'unknown'}")
        if confidence:
            lines.append(f"- **Extraction Confidence**: {confidence}")
        extracted_at = bottle.extracted_at
        if extracted_at.tzinfo is None:  # naive values are UTC (older files)
            extracted_at = extracted_at.replace(tzinfo=UTC)
        # Local time with its UTC offset, e.g. 2026-11-22T21:38:09-06:00.
        lines.append(f"- **Extracted**: {extracted_at.astimezone().isoformat(timespec='seconds')}")
        warnings = list(metadata.warnings) if metadata is not None else []
        for warning in warnings:
            lines.append(f"- **Warning**: {warning}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def bottle_id_for(
        bottle_dict: Optional[Dict[str, Any]],
        metadata: Optional[ExtractedMetadata],
    ) -> str:
        """Identify a bottle: its run-assigned ``bottle_id``, else its primary image's stem.

        Matches the IDs ``BatchProcessor`` uses for failed bottles and that
        ``interactive_prompt`` shows, so rejects and reports line up. A
        ``process`` run assigns unique IDs (``bottle_dict['bottle_id']``), so
        photos with the same name in different folders never collide.

        Args:
            bottle_dict: Analyzer output (its ``bottle_id`` / ``images``), or None.
            metadata: Extracted metadata (its ``source_images``), or None.

        Returns:
            The bottle ID, or 'unknown'.
        """
        assigned = (bottle_dict or {}).get("bottle_id")
        if isinstance(assigned, str) and assigned:
            return assigned
        if metadata is not None and metadata.source_images:
            return Path(metadata.source_images[0]).stem
        images = (bottle_dict or {}).get("images") or []
        if images:
            return Path(images[0]).stem
        return "unknown"

    def save_bottle_files(
        self,
        bottle: Bottle,
        metadata: Optional[ExtractedMetadata] = None,
        prefix: Optional[str] = None,
        *,
        model_used: Optional[str] = None,
        confidence: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a bottle's output folder atomically, never overwriting.

        Writes ``<output_dir>/<stem>/`` containing ``<stem>.md``,
        ``<stem>.json`` (``Bottle.to_dict()`` plus extra keys), ``<stem>.txt``
        (OCR text) and ``images/`` with a copy of every source image. The stem
        is the slugified bottle name (``unknown_<primary image>`` for
        placeholder names); if that folder exists, ``_2``, ``_3``... is used.

        The folder is built in a hidden ``.<stem>.*.partial`` directory inside
        ``output_dir`` and renamed into place, so readers never see a
        half-written folder and a failure leaves nothing behind. Originals
        are only read. A missing source image is logged and listed in the
        JSON's ``missing_images``; it is not an error.

        Args:
            bottle: Bottle to save (build it with ``Bottle.from_analysis``).
            metadata: Extracted metadata; its images are added to the
                bottle's and its ``full_text`` is written to ``<stem>.txt``.
            prefix: Folder name to use instead of the bottle name (slugified).
            model_used: Vision model that extracted the data.
            confidence: Extraction confidence ('HIGH'/'MEDIUM'/'LOW').
            extra: Additional top-level keys for the JSON file.

        Returns:
            Path to the created output folder.

        Raises:
            FileSystemError: If the folder could not be written; nothing is
                left in ``output_dir`` in that case.
        """
        meta = bottle.metadata or metadata or ExtractedMetadata()
        sources: List[Path] = []
        seen: set = set()
        for image_path in [*meta.source_images, *(metadata.source_images if metadata else [])]:
            if str(image_path) not in seen:
                seen.add(str(image_path))
                sources.append(Path(image_path))
        ocr_text = (metadata or meta).full_text or ""
        bottle_id = Path(sources[0]).stem if sources else "unknown"

        if prefix:
            stem = slugify(prefix)
        elif _is_placeholder_name(bottle.name):
            stem = f"unknown_{slugify(bottle_id, fallback='bottle')}"
        else:
            stem = slugify(bottle.name)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mark_output_dir:
            try:
                (self.output_dir / OUTPUT_MARKER).touch()
            except OSError:
                logger.debug("Could not write output marker in %s", self.output_dir, exc_info=True)

        staging: Optional[Path] = None
        try:
            staging = self._make_staging_dir(stem)
            # Marks the bottle folder itself as output, so it stays out of
            # later scans even when its parent is used as the scan root.
            (staging / OUTPUT_MARKER).touch()
            image_dir = staging / "images"
            image_dir.mkdir()

            copied: List[str] = []
            missing: List[str] = []
            used_names: set = set()
            for src in sources:
                if not src.is_file():
                    logger.warning("Source image not found, not copied: %s", src)
                    missing.append(str(src))
                    continue
                name, n = src.name, 2
                while name.lower() in used_names:
                    name = f"{src.stem}_{n}{src.suffix}"
                    n += 1
                used_names.add(name.lower())
                shutil.copy2(src, image_dir / name)
                copied.append(f"images/{name}")
                logger.debug("Copied image %s to %s", src, image_dir / name)

            final = _unique_path(self.output_dir, stem)
            data = {
                **bottle.to_dict(),
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "bottle_id": bottle_id,
                "batch_number": meta.batch_number,
                "model_used": model_used,
                "confidence": confidence,
                "source_images": [str(p) for p in sources],
                "images": copied,
                "missing_images": missing,
                "total_value": str(bottle.total_value) if bottle.total_value is not None else None,
                **(extra or {}),
            }
            self._write_bottle_files(
                staging,
                final.name,
                markdown=self.generate_bottle_markdown(
                    bottle, image_links=copied, model_used=model_used, confidence=confidence
                ),
                json_text=json.dumps(data, indent=2, ensure_ascii=False, default=str),
                ocr_text=ocr_text,
            )
            final = self._move_into_place(staging, final, stem)
            staging = None
        except Exception as e:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            raise FileSystemError(
                f"Failed to save output for bottle '{bottle.name}' in {self.output_dir}: "
                f"{type(e).__name__}: {e}"
            ) from e

        logger.info("Saved bottle '%s' to %s (%d image(s))", bottle.name, final, len(copied))
        return final

    def _make_staging_dir(self, stem: str) -> Path:
        """Create a hidden ``.<stem>.<random>.partial`` directory in ``output_dir``.

        Uses a plain ``mkdir`` (not ``tempfile.mkdtemp``, which forces mode
        0700) so the folder renamed into place gets the same umask-derived
        permissions as any other directory the user creates.
        """
        for _ in range(100):
            staging = self.output_dir / f".{stem}.{uuid.uuid4().hex}.partial"
            try:
                staging.mkdir()
            except FileExistsError:
                continue
            return staging
        raise FileExistsError(f"Could not create a staging directory in {self.output_dir}")

    @staticmethod
    def _write_bottle_files(
        folder: Path,
        stem: str,
        *,
        markdown: str,
        json_text: str,
        ocr_text: str,
    ) -> None:
        """Write ``<stem>.md``, ``<stem>.json`` and ``<stem>.txt`` into ``folder``."""
        (folder / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (folder / f"{stem}.json").write_text(json_text + "\n", encoding="utf-8")
        (folder / f"{stem}.txt").write_text(ocr_text, encoding="utf-8")

    @staticmethod
    def _move_into_place(staging: Path, final: Path, stem: str) -> Path:
        """Rename the staging folder to ``final`` or the next free ``stem_N``.

        If another process claimed ``final`` after it was chosen, the rename
        fails (the target is not empty) and the next free name is tried; the
        files inside are renamed to match. An existing folder is never
        replaced.

        Returns:
            The folder's final path.
        """
        for _ in range(_MAX_NAME_SUFFIX):
            try:
                if final.exists():
                    raise FileExistsError(final)
                staging.rename(final)
                return final
            except (FileExistsError, IsADirectoryError, NotADirectoryError):
                pass
            except OSError as e:
                if not final.exists():
                    raise
                logger.debug("Output name %s was taken: %s", final.name, e)
            new_final = _unique_path(final.parent, stem)
            for ext in (".md", ".json", ".txt"):
                (staging / f"{final.name}{ext}").rename(staging / f"{new_final.name}{ext}")
            final = new_final
        raise FileExistsError(f"No free output name for '{stem}' in {final.parent}")

    def generate_batch_preview_report(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        rejected_bottles: Optional[List[str]] = None,
        batch_id: int = 1,
    ) -> str:
        """Generate preview report for multiple bottles with grouping summary table.

        Creates markdown with grouping summary table (Bottle Name, Quantity, Images,
        Confidence), extracted metadata details per bottle, rejected/low-confidence items,
        and instructions for feedback.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            rejected_bottles: List of rejected bottle identifiers.
            batch_id: Batch identifier for this report.

        Returns:
            Markdown formatted preview report.
        """
        lines = [f"# Batch Preview - Batch {batch_id}\n", "\n"]
        lines.append(f"Total bottles in batch: {len(bottle_dicts)}\n")

        if rejected_bottles:
            lines.append(f"Rejected bottles: {len(rejected_bottles)}\n")

        lines.append("\n## Grouping Summary\n")
        lines.append("| Bottle Name | Quantity | Images | Confidence |\n")
        lines.append("|-------------|----------|--------|------------|\n")

        for bottle_dict, metadata in zip(bottle_dicts, metadata_list):
            name = bottle_dict.get("name", "Unknown")
            quantity = bottle_dict.get("quantity", 1)
            image_count = len(metadata.source_images)

            # Calculate confidence level
            filled_fields = sum(
                1
                for field in ["name", "distillery", "msrp", "proof", "age"]
                if bottle_dict.get(field)
            )
            if filled_fields >= 4:
                confidence = "HIGH"
            elif filled_fields >= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            lines.append(f"| {name} | {quantity} | {image_count} | {confidence} |\n")

        # Rejected/low-confidence items section
        low_confidence_bottles = []
        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list)):
            filled_fields = sum(
                1
                for field in ["name", "distillery", "msrp", "proof", "age"]
                if bottle_dict.get(field)
            )
            if filled_fields < 2:
                low_confidence_bottles.append(i + 1)

        if low_confidence_bottles or rejected_bottles:
            lines.append("\n## Low-Confidence or Rejected Items\n")

            if low_confidence_bottles:
                lines.append("**Low Confidence:**\n")
                for idx in low_confidence_bottles:
                    lines.append(f"- Bottle {idx} (check extracted metadata)\n")

            if rejected_bottles:
                lines.append("\n**Rejected:**\n")
                for bottle_id in rejected_bottles:
                    lines.append(f"- {bottle_id}\n")

        # Detailed metadata per bottle
        lines.append("\n## Extracted Metadata Details\n\n")

        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list), 1):
            lines.append(f"### Bottle {i}\n")
            lines.append(f"**Name:** {bottle_dict.get('name', 'Unknown')}\n")
            lines.append(f"**Distillery:** {bottle_dict.get('distillery', 'Unknown')}\n")
            lines.append(f"**MSRP:** {format_money(parse_money(bottle_dict.get('msrp')))}\n")

            if bottle_dict.get("proof"):
                lines.append(f"**Proof:** {bottle_dict['proof']}\n")

            if bottle_dict.get("age"):
                lines.append(f"**Age:** {bottle_dict['age']}\n")

            if bottle_dict.get("specific_release"):
                lines.append(f"**Release:** {bottle_dict['specific_release']}\n")

            lines.append(f"**Quantity:** {bottle_dict.get('quantity', 1)}\n")

            # Review hints (e.g. a group split by the same-bottle check)
            warnings = getattr(metadata, "warnings", None)
            lines.extend(f"**Review:** {w}\n" for w in (warnings if isinstance(warnings, list) else []))

            # Images section
            lines.append(f"**Images:** {len(metadata.source_images)}\n")
            for img_path in metadata.source_images:
                img_name = Path(img_path).name
                lines.append(f"- {img_name}\n")

            # OCR preview
            if metadata.full_text:
                lines.append("\n**OCR Preview:**\n")
                lines.append("```\n")
                preview = metadata.full_text[:300]
                if len(metadata.full_text) > 300:
                    preview += "..."
                lines.append(f"{preview}\n")
                lines.append("```\n")

            lines.append("\n")

        # Feedback instructions
        lines.append("## Feedback Instructions\n")
        lines.append("Review the batch grouping and metadata above. To provide feedback:\n\n")
        lines.append("1. **Interactive Mode:** Use the [A]ccept/[R]eject/[E]dit/[Q]uit options\n")
        lines.append("   - **A** - Accept the grouping and metadata as-is\n")
        lines.append("   - **R** - Reject this bottle for re-processing\n")
        lines.append("   - **E** - Edit specific fields (name, distillery, MSRP, etc.)\n")
        lines.append("   - **Q** - Quit the review (remaining bottles are kept as-is)\n\n")
        lines.append(
            "2. **Separate Feedback Command:** Use the `feedback` command with a JSON corrections "
            "file: a list of `{bottle_id, field_name, original_value, corrected_value}`\n"
        )
        lines.append("   ```\n")
        lines.append(
            f"   bourbon-process feedback --batch {batch_id} --corrections corrections.json\n"
        )
        lines.append("   ```\n\n")
        lines.append(
            "3. **Retry Failed Items:** Use the `retry` command with an error report file\n"
        )
        lines.append("   ```\n")
        lines.append("   bourbon-process retry --report error_report_<bottle_id>.json\n")
        lines.append("   ```\n")

        return "\n".join(lines)

    def save_batch_preview(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        batch_id: Union[int, str] = 1,
        output_path: Optional[Path] = None,
        rejected_bottles: Optional[List[str]] = None,
    ) -> Path:
        """Save batch preview report to file with batch-specific naming.

        Writes the markdown preview to ``batch_{batch_id}_preview.md`` and the
        batch data (bottles, metadata, rejected IDs) to a sibling
        ``batch_{batch_id}.json``, which the ``feedback`` command loads.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            batch_id: Batch identifier for filename generation.
            output_path: Markdown file path, or a directory to write
                ``batch_{batch_id}_preview.md`` into (default: output_dir).
            rejected_bottles: IDs of bottles rejected or failed in this batch.

        Returns:
            Path to saved preview report.
        """
        if output_path is None:
            output_path = self.output_dir / f"batch_{batch_id}_preview.md"
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / f"batch_{batch_id}_preview.md"

        rejected = list(rejected_bottles or [])
        markdown = self.generate_batch_preview_report(
            bottle_dicts,
            metadata_list,
            rejected_bottles=rejected,
            batch_id=batch_id,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Saved batch preview to %s", output_path)

        if output_path.name.endswith("_preview.md"):
            json_path = output_path.with_name(output_path.name[: -len("_preview.md")] + ".json")
        else:
            json_path = output_path.with_suffix(".json")
        batch_data = {
            "batch_id": str(batch_id),
            "generated_at": datetime.now(UTC).isoformat(),
            "bottles": bottle_dicts,
            "metadata": [m.to_dict() for m in metadata_list],
            "rejected": rejected,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(batch_data, indent=2, ensure_ascii=False, default=str))
        logger.info("Saved batch data to %s", json_path)

        return output_path

    def save_failure_report(
        self,
        bottle_id: str,
        images: List[str],
        error: Union[BaseException, str],
        stage: str,
        retry_count: int = 0,
    ) -> Path:
        """Write a per-bottle failure report that ``retry --report`` can read.

        The report is ``<output_dir>/error_report_<bottle_id>[_N].json`` (an
        existing report is never overwritten) and holds ``bottle_id``,
        ``images`` (absolute paths) and ``error_message``, plus error type,
        stage, hint, details and timestamp.

        Args:
            bottle_id: Bottle identifier (primary image stem).
            images: All image paths of the bottle.
            error: The exception, or a description of the failure.
            stage: Where it failed: 'analysis', 'save' or 'rejected'.
            retry_count: Retries already attempted.

        Returns:
            Path to the written report.
        """
        if isinstance(error, BaseException):
            error_message = f"{type(error).__name__}: {error}"
            info = create_error_report(error)
            error_type, hint = info.error_type.name, info.hint
            details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        else:
            error_message = str(error)
            error_type, hint, details = "UNKNOWN", None, None

        absolute_images = []
        for image in images:
            try:
                absolute_images.append(str(Path(image).absolute()))
            except (TypeError, ValueError):
                absolute_images.append(str(image))

        report = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "bottle_id": bottle_id,
            "images": absolute_images,
            "error_message": error_message,
            "error_type": error_type,
            "stage": stage,
            "hint": hint,
            "details": details,
            "timestamp": datetime.now(UTC).isoformat(),
            "retry_count": retry_count,
            "suggested_fix": {},
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"error_report_{slugify(bottle_id, fallback='unknown')}"
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
        report_path = self._publish_new_file(stem, ".json", text)
        logger.info("Saved %s failure report for %s to %s", stage, bottle_id, report_path)
        return report_path

    def _publish_new_file(self, stem: str, suffix: str, text: str) -> Path:
        """Write ``text`` to the first free ``<stem>[_N]<suffix>`` in ``output_dir``.

        The content is written to a hidden temporary file first (created with
        ``open(..., 'x')`` so it gets umask-derived permissions) and then
        hard-linked to the final name. ``os.link`` fails instead of replacing
        an existing file, so a report written concurrently by another run is
        never overwritten: the next suffix is tried. On a filesystem without
        hard links the final name is created directly with mode ``'x'``.

        Returns:
            Path of the published file.
        """
        tmp: Optional[Path] = None
        for _ in range(100):
            candidate = self.output_dir / f".{stem}.{uuid.uuid4().hex}.tmp"
            try:
                with candidate.open("x", encoding="utf-8") as f:
                    f.write(text)
            except FileExistsError:
                continue
            tmp = candidate
            break
        if tmp is None:
            raise FileExistsError(f"Could not create a temporary file in {self.output_dir}")

        try:
            target = _unique_path(self.output_dir, stem, suffix)
            for _ in range(_MAX_NAME_SUFFIX):
                try:
                    os.link(tmp, target)
                    return target
                except FileExistsError:
                    target = _unique_path(self.output_dir, stem, suffix)
                except OSError:
                    # No hard links here (e.g. some network/FAT filesystems):
                    # create the final file exclusively instead.
                    logger.debug("Hard link not supported in %s", self.output_dir, exc_info=True)
                    break
            while True:
                try:
                    with target.open("x", encoding="utf-8") as f:
                        f.write(text)
                    return target
                except FileExistsError:
                    target = _unique_path(self.output_dir, stem, suffix)
        finally:
            tmp.unlink(missing_ok=True)

    def interactive_prompt(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
    ) -> tuple[List[Dict[str, str]], List[str]]:
        """Prompt user interactively for corrections after batch processing.

        Displays each bottle's key information and waits for user input.
        Supports [A]ccept, [R]eject, [E]dit, and [Q]uit actions. Edits are
        returned as correction dicts carrying the bottle's ``bottle_id``; the
        bottle dicts themselves are not changed here (apply the corrections
        with :func:`image_processor.corrections.apply_corrections`).

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.

        Returns:
            Tuple of (corrections, rejected_bottles) where:
                - corrections: List of correction dicts (bottle_id, field_name,
                  original_value, corrected_value, corrected_at)
                - rejected_bottles: List of bottle identifiers marked for rejection
        """
        corrections = []
        rejected_bottles = []

        print("\n" + "=" * 70)
        print("BATCH PROCESSING - INTERACTIVE REVIEW")
        print("=" * 70)
        print(f"\nProcessing {len(bottle_dicts)} bottles in this batch")
        print("Actions: [A]ccept | [R]eject | [E]dit | [Q]uit\n")

        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list), 1):
            # Display bottle identifier
            bottle_id = self.bottle_id_for(bottle_dict, metadata)
            print("-" * 70)
            print(f"Bottle {i}/{len(bottle_dicts)} - ID: {bottle_id}")
            print("-" * 70)

            # Display key bottle information
            print(f"\nName:        {bottle_dict.get('name', 'Unknown')}")
            print(f"Distillery:  {bottle_dict.get('distillery', 'Unknown')}")
            print(f"MSRP:        {format_money(parse_money(bottle_dict.get('msrp')))}")

            # Display optional fields if present
            proof = bottle_dict.get("proof")
            if proof:
                print(f"Proof:       {proof}")

            age = bottle_dict.get("age")
            if age:
                print(f"Age:         {age}")

            release = bottle_dict.get("specific_release")
            if release:
                print(f"Release:     {release}")

            quantity = bottle_dict.get("quantity", 1)
            print(f"Quantity:    {quantity}")

            # Display image count
            image_count = len(metadata.source_images)
            print(f"Images:      {image_count}")

            # The extraction confidence recorded for this bottle (not re-derived).
            print(f"Confidence:  {bottle_dict.get('confidence') or 'Unknown'}")

            # Wait for a recognised action: a stray key (or a bare Enter) must
            # never count as accepting a bottle someone has not looked at.
            print("\nAction [A/R/E/Q]? ", end="", flush=True)
            action = input().strip().upper()
            while action not in ("A", "R", "E", "Q"):
                print(f"⚠ Unknown action '{action}'; enter A, R, E or Q: ", end="", flush=True)
                action = input().strip().upper()

            # Handle user action
            if action == "Q":
                print("\nQuitting interactive review...")
                break
            elif action == "A":
                print(f"✓ Accepted bottle {i}: {bottle_dict.get('name', 'Unknown')}")
            elif action == "R":
                rejected_bottles.append(bottle_id)
                print(f"✗ Rejected bottle {i}: {bottle_id}")
            elif action == "E":
                # Edit specific fields
                print(f"\nEditing bottle {i}: {bottle_dict.get('name', 'Unknown')}")
                print("Press Enter to keep current value for each field\n")

                for field in ["name", "distillery", "msrp", "proof", "age", "specific_release"]:
                    raw = bottle_dict.get(field)
                    current = "" if raw is None else str(raw)
                    label = field.replace("_", " ").title()
                    new_value = input(f"{label:14} [{current}]: ").strip()

                    if new_value and new_value != current.strip():
                        corrections.append(
                            {
                                "field_name": field,
                                "original_value": current,
                                "corrected_value": new_value,
                                "bottle_id": bottle_id,
                                "corrected_at": datetime.now(UTC).isoformat(),
                            }
                        )
                        print(f"  Will update: {field} = {new_value}")

        # Summary
        print("\n" + "=" * 70)
        print("INTERACTIVE REVIEW SUMMARY")
        print("=" * 70)
        accepted = len(bottle_dicts) - len(rejected_bottles)
        print(f"Accepted:     {accepted}/{len(bottle_dicts)}")
        print(f"Rejected:     {len(rejected_bottles)}")
        print(f"Corrections:  {len(corrections)}")
        print("=" * 70)

        return corrections, rejected_bottles
