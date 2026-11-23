"""Organize and group images by bottle identity.

Scans image folders, detects duplicates, groups related images,
and prepares bottle image sets for processing.
"""

import logging
import os
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

from image_processor.capture_grouping import (
    DEFAULT_GAP_SECONDS,
    DEFAULT_MAX_GROUP_SIZE,
    NEAR_DUP_MAX_GAP_SECONDS,
    angle_base,
    dhash64,
    is_near_duplicate,
    natural_key,
    resolve_capture_times,
    segment_by_time,
)
from lib.duplicates import DuplicateDetector
from models.image import Image

logger = logging.getLogger(__name__)

#: Marker file written into every output directory. A directory below the
#: scan root that contains it holds earlier results and is never scanned.
OUTPUT_MARKER = ".bourbon_output"


def _dir_identity(path: Path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` of an existing directory, else None.

    Comparing identities instead of resolved paths also matches paths that
    differ only in letter case on case-insensitive filesystems (the macOS
    default) and paths reached through symlinks or hard-to-normalize mounts.
    """
    try:
        st = os.stat(path)
    except (TypeError, ValueError, OSError):
        return None
    return (st.st_dev, st.st_ino)


def _resolve_excludes(root: Path, exclude: Iterable[Path]) -> set[tuple[int, int]]:
    """Identities of exclude directories, dropping unusable ones and ancestors of ``root``.

    Entries that do not exist or cannot be turned into a path (e.g. mocks in
    tests) are skipped. An entry that is ``root`` or contains it is ignored,
    so ``-f processed/x -o processed`` still scans ``processed/x``.

    Args:
        root: Resolved scan root.
        exclude: Directories to leave out of the scan.

    Returns:
        ``(st_dev, st_ino)`` of each exclude directory below ``root`` or elsewhere.
    """
    root_and_parents = {ident for ident in map(_dir_identity, [root, *root.parents]) if ident}
    identities: set[tuple[int, int]] = set()
    for entry in exclude or ():
        try:
            path = Path(entry)
        except (TypeError, ValueError):
            continue
        ident = _dir_identity(path)
        if ident is None or ident in root_and_parents:
            continue
        identities.add(ident)
    return identities


class ImageOrganizer:
    """Organize bourbon bottle images for processing.

    Scans input folders, detects duplicates, groups related images,
    and prepares image sets for bottle analysis.

    Attributes:
        supported_extensions: Set of supported image file extensions.
        duplicate_detector: DuplicateDetector for hash-based duplicate detection.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".jpeg", ".jpg", ".png", ".tiff", ".heic", ".webp", ".bmp"}

    def __init__(
        self,
        duplicate_detector: DuplicateDetector | None = None,
        gap_seconds: float = DEFAULT_GAP_SECONDS,
        max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
        detect_near_duplicates: bool = True,
    ):
        """Initialize image organizer.

        Args:
            duplicate_detector: Optional DuplicateDetector instance.
            gap_seconds: Most seconds between consecutive photos of one
                bottle (see :mod:`image_processor.capture_grouping`).
            max_group_size: Most photos per bottle group (0 = no cap).
            detect_near_duplicates: Flag burst retakes (near-identical photos
                taken a few seconds apart) with ``Image.near_duplicate_of``.
        """
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.gap_seconds = gap_seconds
        self.max_group_size = max_group_size
        self.detect_near_duplicates = detect_near_duplicates

    def iter_image_paths(
        self,
        folder: Path,
        recursive: bool = True,
        exclude: Iterable[Path] = (),
    ) -> list[Path]:
        """List supported image files below ``folder`` in sorted order.

        Leaves out hidden files and directories (including AppleDouble
        ``._*`` files and in-progress ``.partial`` output), subdirectories
        that contain :data:`OUTPUT_MARKER` and anything under ``exclude``.
        Paths are returned as found below ``folder`` (not resolved).
        Symlinked subdirectories are not entered (logged at INFO), matching
        ``Path.glob('**/*')``/``rglob`` on the supported Python versions;
        exclusions are matched by filesystem identity, so they also hold
        when a path differs only in letter case.

        Args:
            folder: Directory to scan.
            recursive: Whether to descend into subdirectories.
            exclude: Directories to skip, typically the output directory. An
                entry that is ``folder`` itself or one of its ancestors is
                ignored.

        Returns:
            Paths of supported image files, sorted by directory then name.
        """
        try:
            root = Path(folder).resolve()
        except (TypeError, ValueError, OSError):
            return []
        excludes = _resolve_excludes(root, exclude)

        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(folder):
            current = Path(dirpath)
            if recursive:
                kept = []
                for name in sorted(dirnames):
                    if name.startswith("."):
                        continue
                    subdir = current / name
                    if (subdir / OUTPUT_MARKER).exists():
                        logger.info("Skipping earlier output directory: %s", subdir)
                        continue
                    if subdir.is_symlink():
                        # Same as the previous Path.glob('**/*') scan and the
                        # pre-flight rglob count: symlinked dirs are not entered.
                        logger.info("Not following symlinked directory: %s", subdir)
                        continue
                    if excludes and _dir_identity(subdir) in excludes:
                        logger.debug("Skipping excluded directory: %s", subdir)
                        continue
                    kept.append(name)
                dirnames[:] = kept
            else:
                dirnames[:] = []

            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                path = current / name
                if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    found.append(path)
        return found

    def scan_images(
        self,
        folder: Path,
        recursive: bool = True,
        exclude: Iterable[Path] = (),
    ) -> list[Image]:
        """Scan a folder for supported images (see :meth:`iter_image_paths`).

        Args:
            folder: Path to folder to scan.
            recursive: Whether to scan subdirectories recursively.
            exclude: Directories to skip, typically the output directory.

        Returns:
            List of Image objects (path, md5 hash and file size populated),
            sorted by directory then name.
        """
        if not folder.exists() or not folder.is_dir():
            logger.warning("Folder does not exist or is not a directory: %s", folder)
            return []

        images: list[Image] = []
        for path in self.iter_image_paths(folder, recursive=recursive, exclude=exclude):
            try:
                img = Image(path=str(path))
                img.md5_hash = self.duplicate_detector.generate_hash(path)
                img.file_size = path.stat().st_size
                images.append(img)
                logger.debug("Found image: %s", path)
            except Exception as e:
                logger.warning("Failed to process image %s: %s", path, e)

        logger.info("Scanned %d images from %s", len(images), folder)
        return images

    def group_by_bottle(
        self,
        images: list[Image],
        similarity_threshold: float | None = None,
        *,
        gap_seconds: float | None = None,
        max_group_size: int | None = None,
    ) -> list[list[Image]]:
        """Group images that belong to the same bottle, by capture time.

        Photos taken by one camera at most ``gap_seconds`` apart form a group;
        groups longer than ``max_group_size`` are split at their largest
        time gap (see :mod:`image_processor.capture_grouping`). Photos
        without a capture time are single-photo groups unless they share an
        angle-suffix base name (``IMG_1_front`` / ``IMG_1_back``). Exact
        duplicates (same file hash) join their original's group, marked
        ``is_duplicate``. File names are otherwise never compared, so
        consecutive numbers like ``IMG_4653``/``IMG_4654`` do not merge.

        Sets ``capture_time``, ``capture_time_source`` and (for burst
        retakes) ``near_duplicate_of`` on the images.

        Args:
            images: Image objects to group.
            similarity_threshold: Ignored; kept for API compatibility.
            gap_seconds: Override of the organizer's gap.
            max_group_size: Override of the organizer's cap.

        Returns:
            Groups in capture order; each group is a list of Image objects for
            one bottle.
        """
        if not images:
            return []
        if similarity_threshold is not None:
            logger.debug("group_by_bottle: similarity_threshold is ignored (grouping is by capture time)")
        gap = self.gap_seconds if gap_seconds is None else gap_seconds
        cap = self.max_group_size if max_group_size is None else max_group_size

        self.detect_duplicates(images)
        reps = [img for img in images if not img.is_duplicate]
        infos = resolve_capture_times([img.path for img in reps])
        for img, info in zip(reps, infos, strict=True):
            img.capture_time = info.time
            img.capture_time_source = info.source

        # Timed photos: chain within each camera's own stream, so two devices
        # shooting at once never interleave into mixed groups. Photos with no
        # camera tags (e.g. timed by a trusted mtime) join the camera stream
        # when there is exactly one camera; with several they form their own.
        cameras = {info.camera for info in infos if info.time is not None and info.camera}
        only_camera = next(iter(cameras)) if len(cameras) == 1 else ""
        streams: dict[str, list[Image]] = {}
        for img, info in zip(reps, infos, strict=True):
            if info.time is not None:
                streams.setdefault(info.camera or only_camera, []).append(img)
        groups: list[list[Image]] = []
        for stream in streams.values():
            stream.sort(key=lambda img: (img.capture_time, natural_key(img.path)))
            segments = segment_by_time([img.capture_time for img in stream], gap, cap)
            groups.extend([stream[k] for k in seg] for seg in segments)

        # Untimed photos: alone, unless explicit angle suffixes pair them.
        by_base: dict[str, list[Image]] = {}
        for img in sorted((i for i in reps if i.capture_time is None), key=lambda i: natural_key(i.path)):
            base = angle_base(img.path)
            if base is None:
                groups.append([img])
            else:
                if base not in by_base:
                    by_base[base] = []
                    groups.append(by_base[base])
                by_base[base].append(img)

        groups.sort(
            key=lambda g: (g[0].capture_time is None, g[0].capture_time or 0.0, natural_key(g[0].path))
        )

        # Exact duplicates follow their original.
        group_of = {str(img.path): group for group in groups for img in group}
        for img in images:
            if not img.is_duplicate:
                continue
            group = group_of.get(str(img.duplicate_of_id))
            if group is None:
                groups.append([img])
                continue
            position = next(k for k, g in enumerate(group) if str(g.path) == str(img.duplicate_of_id))
            while position + 1 < len(group) and group[position + 1].is_duplicate:
                position += 1  # after earlier copies of the same original
            group.insert(position + 1, img)

        if self.detect_near_duplicates:
            self._mark_near_duplicates(groups)

        sizes = Counter(len(group) for group in groups)
        sources = Counter(img.capture_time_source for img in reps)
        logger.info(
            "Grouped %d images into %d bottle groups by capture time (gap %.1fs, max %s per group); "
            "group sizes %s; time sources %s",
            len(images), len(groups), gap, cap if cap and cap > 0 else "unlimited",
            dict(sorted(sizes.items())), dict(sources),
        )
        for i, group in enumerate(groups, 1):
            logger.debug("Group %d: %s", i, [Path(img.path).name for img in group])
        return groups

    @staticmethod
    def _mark_near_duplicates(groups: list[list[Image]]) -> None:
        """Set ``near_duplicate_of`` on burst retakes within each group."""
        for group in groups:
            timed = [img for img in group if not img.is_duplicate and img.capture_time is not None]
            hashes: dict[str, int | None] = {}
            for prev, cur in pairwise(timed):
                gap = cur.capture_time - prev.capture_time  # type: ignore[operator]
                if gap > NEAR_DUP_MAX_GAP_SECONDS:
                    continue
                for img in (prev, cur):
                    if str(img.path) not in hashes:
                        hashes[str(img.path)] = dhash64(img.path)
                if is_near_duplicate(hashes[str(prev.path)], hashes[str(cur.path)], gap):
                    cur.near_duplicate_of = str(prev.near_duplicate_of or prev.path)
                    logger.info("Burst retake: %s is a near copy of %s", Path(cur.path).name, Path(prev.path).name)

    def detect_duplicates(
        self,
        images: list[Image],
    ) -> None:
        """Detect exact duplicate images using file hash comparison.

        Updates Image.is_duplicate and Image.duplicate_of_id fields.

        Args:
            images: List of Image objects to check for duplicates.
        """
        hash_to_images: dict[str, list[Image]] = {}

        # Group by hash
        for img in images:
            if img.md5_hash:
                if img.md5_hash not in hash_to_images:
                    hash_to_images[img.md5_hash] = []
                hash_to_images[img.md5_hash].append(img)

        # Mark duplicates
        duplicate_count = 0
        for hash_images in hash_to_images.values():
            if len(hash_images) > 1:
                # First image is the original, rest are duplicates
                original = hash_images[0]
                for duplicate in hash_images[1:]:
                    duplicate.is_duplicate = True
                    duplicate.duplicate_of_id = original.path
                    duplicate_count += 1

        logger.info("Found %d duplicate images", duplicate_count)

    def is_duplicate_image(
        self,
        img1: Image,
        img2: Image,
    ) -> bool:
        """Check if two images are exact duplicates by comparing file content hash.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            True if images are exact duplicates, False otherwise.
        """
        if not img1.md5_hash or not img2.md5_hash:
            return False

        return img1.md5_hash == img2.md5_hash

    def _images_similar(
        self,
        img1: Image,
        img2: Image,
        threshold: float = 0.6,  # noqa: ARG002 - kept for API compatibility
    ) -> bool:
        """Whether two images are the same bottle by file evidence alone.

        Deprecated helper: true for exact duplicates (same hash) and for
        explicit angle-suffix pairs in one folder (``IMG_1_front`` /
        ``IMG_1_back``). Name prefixes are not compared: consecutive camera
        numbers say nothing about the bottle.

        Args:
            img1: First image.
            img2: Second image.
            threshold: Ignored.

        Returns:
            True if the images belong together.
        """
        if self.is_duplicate_image(img1, img2):
            return True
        base = angle_base(img1.path)
        return base is not None and base == angle_base(img2.path)

    def filter_duplicate_images(
        self,
        images: list[Image],
        keep_original: bool = True,
    ) -> list[Image]:
        """Filter out duplicate images, keeping only one copy.

        Args:
            images: List of Image objects to filter.
            keep_original: If True, keep the first occurrence as original.

        Returns:
            Filtered list of images with duplicates removed.
        """
        seen_hashes: set[str] = set()
        filtered: list[Image] = []

        for img in images:
            if not img.md5_hash:
                # No hash available, keep it
                filtered.append(img)
                continue

            if img.md5_hash not in seen_hashes:
                # First occurrence
                seen_hashes.add(img.md5_hash)
                filtered.append(img)
            elif not keep_original:
                # If we want to keep the "best" version, we could implement
                # logic to keep highest resolution, etc.
                # For now, just skip duplicates
                pass

        removed = len(images) - len(filtered)
        if removed > 0:
            logger.info("Filtered out %d duplicate images", removed)

        return filtered
