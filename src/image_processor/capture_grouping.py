"""Group bottle photos by capture time.

Pure helpers used by :meth:`ImageOrganizer.group_by_bottle`. Pillow only (no
OpenCV), so it works the same on macOS and Linux.

The rule, measured on 98 real photos of one shooting session:

1. Read each photo's capture time from its EXIF header (DateTimeOriginal plus
   sub-seconds, then DateTimeDigitized, then DateTime). File mtimes are used
   only for photos without EXIF, and only when other photos of the same set
   show that the mtimes are preserved camera times (a copy or download
   rewrites them).
2. Within each camera (EXIF Make/Model/serial), order photos by capture time
   and chain consecutive photos taken at most ``gap_seconds`` apart.
3. Split any chain longer than ``max_group_size`` at its largest internal
   gap until every group fits.
4. Photos without a usable time are single-photo groups, except explicit
   angle-suffix names (``IMG_1_front`` / ``IMG_1_back``), which join.

File names are never compared otherwise: consecutive camera numbers
(``IMG_4653``/``IMG_4654``) say nothing about the bottle. Visual similarity
is not used to link photos either (sibling bottles with identical labels
match best); a 64-bit difference hash only flags burst retakes so the vision
step does not spend an image slot on a near copy.

The defaults encode one photographer's habit (mostly front+back pairs, a few
seconds apart); ``--group-gap-seconds`` and ``--group-max-size`` adjust them.
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Most seconds between consecutive photos of one bottle.
DEFAULT_GAP_SECONDS = 12.0
#: Most photos in one group (0 = no cap).
DEFAULT_MAX_GROUP_SIZE = 3
#: dHash Hamming distance at or below which two photos are near copies.
NEAR_DUP_DHASH_MAX = 6
#: Near copies must also be taken at most this many seconds apart.
NEAR_DUP_MAX_GAP_SECONDS = 3.0
#: Gaps within max(GAP_TIE_SECONDS, GAP_TIE_FRACTION x largest) of the
#: largest one count as ties when splitting a chain. B8 required fix #4 asked
#: for max(0.5 s, 10%); that is waived (fraction 0) because on the real
#: session it pairs two different bottles (see split_at_largest_gap).
GAP_TIE_SECONDS = 0.05
GAP_TIE_FRACTION = 0.0
#: mtimes count as camera times when (mtime - EXIF time) varies at most this much.
MTIME_MAX_OFFSET_SPREAD = 2.0

_EXIF_IFD = 0x8769
_MAKE, _MODEL, _DATETIME = 0x010F, 0x0110, 0x0132
_DT_ORIGINAL, _DT_DIGITIZED = 0x9003, 0x9004
_OFFSET_ORIGINAL, _OFFSET_DIGITIZED, _OFFSET_TIME = 0x9011, 0x9012, 0x9010
_SUBSEC_ORIGINAL, _SUBSEC_DIGITIZED, _SUBSEC_TIME = 0x9291, 0x9292, 0x9290
_BODY_SERIAL = 0xA431
_ANGLE_SUFFIXES = (
    "_front", "_back", "_left", "_right", "_top", "_bottom",
    "_side", "_detail", "_box", "_label", "_cap",
)
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


@dataclass(frozen=True)
class CaptureInfo:
    """Capture time of one photo.

    Attributes:
        time: Seconds since the epoch (EXIF local time read as UTC, or the
            true UTC time when an EXIF offset is recorded), or None.
        source: 'exif_original', 'exif_digitized', 'exif_datetime', 'mtime'
            or 'none'.
        camera: Camera identity from EXIF (make/model/serial), '' if unknown.
    """

    time: float | None
    source: str
    camera: str = ""


def _parse_exif_datetime(value: object, subsec: object = None, offset: object = None) -> float | None:
    """EXIF 'YYYY:MM:DD HH:MM:SS' (+ sub-seconds, + offset) as epoch seconds.

    Uses :func:`calendar.timegm`, so the result does not depend on the
    machine's time zone or daylight-saving rules (only differences matter
    unless an offset is recorded, in which case it is true UTC).
    """
    if not value:
        return None
    try:
        # EXIF times carry no zone; timegm below reads them as UTC on purpose.
        parsed = dt.datetime.strptime(str(value).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007
    except ValueError:
        return None
    seconds = float(calendar.timegm(parsed.timetuple()))
    if subsec:
        digits = re.sub(r"\D", "", str(subsec))
        if digits:
            seconds += float("0." + digits)
    match = _OFFSET_RE.match(str(offset or "").strip().rstrip("\x00"))
    if match:
        sign = 1 if match.group(1) == "+" else -1
        seconds -= sign * (int(match.group(2)) * 3600 + int(match.group(3)) * 60)
    return seconds


def read_exif_capture_time(path: str | os.PathLike) -> CaptureInfo:
    """Read a photo's capture time and camera from its EXIF header (no pixel decode).

    Unreadable files, and HEIC without the optional ``pillow-heif`` plugin,
    give ``CaptureInfo(None, 'none')``.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            exif = image.getexif()
            sub = exif.get_ifd(_EXIF_IFD)
            camera = " ".join(
                str(v).strip().rstrip("\x00")
                for v in (exif.get(_MAKE), exif.get(_MODEL), sub.get(_BODY_SERIAL))
                if v
            )
            for tag, subsec, offset, source in (
                (_DT_ORIGINAL, _SUBSEC_ORIGINAL, _OFFSET_ORIGINAL, "exif_original"),
                (_DT_DIGITIZED, _SUBSEC_DIGITIZED, _OFFSET_DIGITIZED, "exif_digitized"),
            ):
                seconds = _parse_exif_datetime(sub.get(tag), sub.get(subsec), sub.get(offset))
                if seconds is not None:
                    return CaptureInfo(seconds, source, camera)
            seconds = _parse_exif_datetime(
                exif.get(_DATETIME), sub.get(_SUBSEC_TIME), sub.get(_OFFSET_TIME)
            )
            if seconds is not None:
                return CaptureInfo(seconds, "exif_datetime", camera)
            return CaptureInfo(None, "none", camera)
    except Exception as e:  # unreadable, corrupt, or HEIC without a plugin
        logger.debug("No EXIF capture time for %s: %s", path, e)
    return CaptureInfo(None, "none")


def _mtimes_trustworthy(exif_times: Sequence[float | None], mtimes: Sequence[float | None]) -> bool:
    """True if file mtimes are preserved camera times.

    Needs at least two photos with both an EXIF time and an mtime whose
    difference is (nearly) constant. Without such evidence mtimes are not
    trusted: a sequential copy or download spaces them about a second apart,
    which would chain unrelated photos.
    """
    both = [(e, m) for e, m in zip(exif_times, mtimes, strict=True) if e is not None and m is not None]
    if len(both) < 2:  # noqa: PLR2004 - a spread needs two points
        return False
    offsets = [m - e for e, m in both]
    return (max(offsets) - min(offsets)) <= MTIME_MAX_OFFSET_SPREAD


def resolve_capture_times(paths: Sequence[str | os.PathLike]) -> list[CaptureInfo]:
    """Capture time of every photo: EXIF, else a trusted mtime, else none.

    A photo without EXIF gets its file mtime (moved onto the EXIF clock)
    only when the mtimes of photos that do have EXIF prove to be preserved
    camera times. One warning is logged when photos stay untimed.
    """
    infos = [read_exif_capture_time(p) for p in paths]
    if all(info.time is not None for info in infos):
        return infos
    mtimes: list[float | None] = []
    for p in paths:
        try:
            mtimes.append(Path(p).stat().st_mtime)
        except OSError:
            mtimes.append(None)
    if _mtimes_trustworthy([i.time for i in infos], mtimes):
        offsets = sorted(
            m - i.time for i, m in zip(infos, mtimes, strict=True) if i.time is not None and m is not None
        )
        offset = offsets[len(offsets) // 2]
        return [
            info if info.time is not None or mtime is None else CaptureInfo(mtime - offset, "mtime", info.camera)
            for info, mtime in zip(infos, mtimes, strict=True)
        ]
    untimed = sum(info.time is None for info in infos)
    logger.warning(
        "%d photo(s) have no EXIF capture time; each becomes its own bottle "
        "(photos named like IMG_1_front / IMG_1_back still pair up)",
        untimed,
    )
    return infos


def natural_key(path: str | os.PathLike) -> tuple:
    """Sort key that orders 'IMG_9' before 'IMG_10'."""
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path)))


def angle_base(path: str | os.PathLike) -> str | None:
    """Folder plus name without an angle suffix ('IMG_1_front' -> '<dir>/img_1'), else None."""
    stem = Path(path).stem.lower()
    for suffix in _ANGLE_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return str(Path(path).parent / stem[: -len(suffix)])
    return None


def split_at_largest_gap(seg: list[int], times: Sequence[float | None]) -> tuple[list[int], list[int]]:
    """Split a segment (at least two photos) at its largest capture gap.

    Gaps within ``max(GAP_TIE_SECONDS, GAP_TIE_FRACTION * largest)`` of the
    largest count as ties (EXIF times without sub-seconds, and a steady
    shooting pace, tie often); among them the one nearest the middle wins,
    so an exactly evenly paced chain splits in half instead of shedding
    single photos.

    Known limitation (B8 required fix #4 waived): only near-exact ties
    (0.05 s) count. A band of max(0.5 s, 10% of the largest gap) was
    measured: on jittered (+-0.5 s), evenly paced 4-7 photo chains it cuts
    "more than one single photo split off" from 15% to 0.7%, but on the
    98-photo session it turns the correct [IMG_4682] | [IMG_4683, 4684,
    4685] into the mixed pair [IMG_4682, IMG_4683] (4682 is a different
    bottle from the Blanton's box 4683; gaps 8.9 s vs 8.8 s), lowering
    pairwise F1 from 0.827 to 0.796, below the 0.80 regression floor
    (fix #8). Re-enable it with ``GAP_TIE_SECONDS = 0.5`` and
    ``GAP_TIE_FRACTION = 0.10`` only if the owner accepts that trade.
    """
    gaps = [float(times[seg[k + 1]] - times[seg[k]]) for k in range(len(seg) - 1)]  # type: ignore[operator]
    largest = max(gaps)
    tolerance = max(GAP_TIE_SECONDS, GAP_TIE_FRACTION * largest)
    middle = (len(seg) - 2) / 2
    candidates = [k for k, gap in enumerate(gaps) if gap >= largest - tolerance]
    k = min(candidates, key=lambda c: (abs(c - middle), c))
    return seg[: k + 1], seg[k + 1:]


def segment_by_time(
    times: Sequence[float | None],
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
) -> list[list[int]]:
    """Chain photos by capture gap and cap the chain length.

    Args:
        times: Capture times, already in capture order (untimed last).
        gap_seconds: Most seconds between consecutive photos of one group.
        max_group_size: Most photos per group; 0 or less disables the cap.

    Returns:
        Contiguous index segments covering ``range(len(times))``.
    """
    if not times:
        return []
    segments: list[list[int]] = []
    current = [0]
    for i in range(1, len(times)):
        a, b = times[i - 1], times[i]
        if a is not None and b is not None and 0 <= b - a <= gap_seconds:
            current.append(i)
        else:
            segments.append(current)
            current = [i]
    segments.append(current)
    if max_group_size and max_group_size > 0:
        capped: list[list[int]] = []
        stack = segments
        while stack:
            seg = stack.pop(0)
            if len(seg) <= max_group_size:
                capped.append(seg)
            else:
                left, right = split_at_largest_gap(seg, times)
                stack[:0] = [left, right]
        segments = capped
    return segments


def dhash64(path: str | os.PathLike) -> int | None:
    """Orientation-corrected 64-bit difference hash (JPEG draft decode), or None."""
    try:
        from PIL import Image as PILImage
        from PIL import ImageOps

        with PILImage.open(path) as image:
            image.draft("L", (64, 64))
            grey = ImageOps.exif_transpose(image).convert("L").resize((9, 8), PILImage.BILINEAR)
            px = grey.tobytes()
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (px[row * 9 + col + 1] > px[row * 9 + col])
        return bits
    except Exception:
        return None


def is_near_duplicate(h1: int | None, h2: int | None, gap: float | None) -> bool:
    """True for a burst retake: near-identical hashes taken at most a few seconds apart."""
    return (
        h1 is not None
        and h2 is not None
        and gap is not None
        and 0 <= gap <= NEAR_DUP_MAX_GAP_SECONDS
        and bin(h1 ^ h2).count("1") <= NEAR_DUP_DHASH_MAX
    )
