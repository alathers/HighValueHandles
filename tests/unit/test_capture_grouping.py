"""Unit tests for image_processor.capture_grouping (synthetic JPEGs with EXIF)."""

import calendar
import datetime as dt
import itertools
import json
import os
import random
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from image_processor import capture_grouping as cg

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "grouping" / "highvalueroom_capture_times.json"
T0 = dt.datetime(2026, 11, 7, 16, 0, 0)


def _exif_bytes(seconds=None, subsec=None, orientation=None, make=None, model=None, offset=None):
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if seconds is not None:
        ifd = exif.get_ifd(0x8769)
        ifd[0x9003] = (T0 + dt.timedelta(seconds=seconds)).strftime("%Y:%m:%d %H:%M:%S")
        if subsec is not None:
            ifd[0x9291] = subsec
        if offset is not None:
            ifd[0x9011] = offset
    if orientation:
        exif[0x0112] = orientation
    return exif.tobytes()


def make_jpeg(path, seconds=None, color=(180, 120, 60), text="A", size=(120, 160), **exif):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 30, 100, 130], outline=(0, 0, 0), width=4)
    draw.text((40, 60), text, fill=(255, 255, 255))
    image.save(path, "JPEG", exif=_exif_bytes(seconds, **exif))
    return str(path)


def group_paths(paths, **kwargs):
    infos = cg.resolve_capture_times(paths)
    order = sorted(
        range(len(paths)),
        key=lambda i: (infos[i].time is None, infos[i].time or 0, cg.natural_key(paths[i])),
    )
    segments = cg.segment_by_time([infos[i].time for i in order], **kwargs)
    return [[os.path.basename(paths[order[k]]) for k in seg] for seg in segments]


def _epoch(seconds):
    return calendar.timegm((T0 + dt.timedelta(seconds=seconds)).timetuple())


def test_reads_exif_original_with_subsec(tmp_path):
    info = cg.read_exif_capture_time(make_jpeg(tmp_path / "a.jpg", 5, subsec="250"))
    assert info.source == "exif_original"
    assert info.time == pytest.approx(_epoch(5) + 0.25)


def test_time_does_not_depend_on_local_timezone(tmp_path, monkeypatch):
    """timegm, not datetime.timestamp(): no DST jumps, same value under any TZ."""
    path = make_jpeg(tmp_path / "a.jpg", 0)
    if not hasattr(__import__("time"), "tzset"):
        pytest.skip("time.tzset unavailable")
    import time

    values = []
    for tz in ("UTC", "America/New_York", "Asia/Tokyo"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        values.append(cg.read_exif_capture_time(path).time)
    monkeypatch.delenv("TZ")
    time.tzset()
    assert values == [_epoch(0)] * 3


def test_offset_time_original_gives_utc(tmp_path):
    info = cg.read_exif_capture_time(make_jpeg(tmp_path / "a.jpg", 0, offset="-06:00"))
    assert info.time == _epoch(0) + 6 * 3600


def test_gap_threshold_splits_bottles(tmp_path):
    paths = [make_jpeg(tmp_path / f"IMG_{i}.jpg", s) for i, s in enumerate([0, 5, 30, 34, 80])]
    assert group_paths(paths) == [["IMG_0.jpg", "IMG_1.jpg"], ["IMG_2.jpg", "IMG_3.jpg"], ["IMG_4.jpg"]]


def test_gap_boundary_inclusive(tmp_path):
    paths = [make_jpeg(tmp_path / "IMG_1.jpg", 0), make_jpeg(tmp_path / "IMG_2.jpg", 12)]
    assert len(group_paths(paths, gap_seconds=12)) == 1
    assert len(group_paths(paths, gap_seconds=11.9)) == 2


def test_orders_by_capture_time_not_filename(tmp_path):
    paths = [
        make_jpeg(tmp_path / "IMG_9.jpg", 0),
        make_jpeg(tmp_path / "IMG_1.jpg", 100),
        make_jpeg(tmp_path / "IMG_5.jpg", 3),
    ]
    assert group_paths(paths) == [["IMG_9.jpg", "IMG_5.jpg"], ["IMG_1.jpg"]]


def test_long_chain_capped_and_split_at_largest_gap(tmp_path):
    paths = [make_jpeg(tmp_path / f"IMG_{i}.jpg", s) for i, s in enumerate([0, 2, 11, 13])]
    assert group_paths(paths) == [["IMG_0.jpg", "IMG_1.jpg"], ["IMG_2.jpg", "IMG_3.jpg"]]
    assert group_paths(paths, max_group_size=0) == [["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]]


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_evenly_spaced_chain_splits_near_the_middle(n):
    """Tied gaps must not peel off single photos one by one."""
    segments = cg.segment_by_time([5.0 * i for i in range(n)])
    assert sum(len(seg) == 1 for seg in segments) <= 1
    assert all(len(seg) <= cg.DEFAULT_MAX_GROUP_SIZE for seg in segments)


def test_split_tie_band_is_near_exact_only():
    """B8 fix #4's wider band (max(0.5 s, 10%)) is waived; see split_at_largest_gap.

    Real case IMG_4682 | IMG_4683-4685 (gaps 8.9 s vs 8.8 s): the largest gap
    wins, so 4682 stays alone instead of pairing with the Blanton's box 4683.
    Known cost: a jittered, evenly paced chain can still shed single photos.
    """
    assert (cg.GAP_TIE_SECONDS, cg.GAP_TIE_FRACTION) == (0.05, 0.0)
    assert cg.split_at_largest_gap([0, 1, 2, 3], [0.0, 8.9, 17.7, 18.0]) == ([0], [1, 2, 3])
    # Exact ties (EXIF without sub-seconds) still go to the middle gap.
    assert cg.split_at_largest_gap([0, 1, 2, 3], [0.0, 5.0, 10.0, 15.0]) == ([0, 1], [2, 3])
    assert cg.split_at_largest_gap([0, 1, 2, 3], [0.0, 5.0, 10.03, 15.0]) == ([0, 1], [2, 3])


def test_decade_filenames_do_not_chain(tmp_path):
    """Regression: IMG_4660..IMG_4669 used to be one 'bottle' by name prefix."""
    paths = [make_jpeg(tmp_path / f"IMG_466{i}.jpg", i * 60) for i in range(10)]
    assert all(len(group) == 1 for group in group_paths(paths))


def test_missing_exif_bulk_copy_gives_singletons(tmp_path):
    paths = [make_jpeg(tmp_path / f"IMG_{i}.jpg") for i in range(3)]
    for p in paths:
        os.utime(p, (1_700_000_000, 1_700_000_000))
    assert group_paths(paths) == [["IMG_0.jpg"], ["IMG_1.jpg"], ["IMG_2.jpg"]]


def test_missing_exif_sequential_copy_gives_singletons(tmp_path):
    """A download/copy at 1-2 s per file looks like a shooting rhythm: never chain on it."""
    rng = random.Random(7)
    paths = [make_jpeg(tmp_path / f"IMG_{i}.jpg") for i in range(8)]
    stamp = 1_700_000_000.0
    for p in paths:
        stamp += rng.uniform(1.0, 2.0)
        os.utime(p, (stamp, stamp))
    infos = cg.resolve_capture_times(paths)
    assert {info.source for info in infos} == {"none"}
    assert all(len(group) == 1 for group in group_paths(paths))


def test_preserved_mtimes_time_photos_without_exif(tmp_path):
    """Photos with EXIF prove the mtimes are camera times (constant offset): use them."""
    a = make_jpeg(tmp_path / "IMG_1.jpg", 0)
    b = make_jpeg(tmp_path / "IMG_2.jpg", 100)
    c = make_jpeg(tmp_path / "IMG_3.jpg")  # no EXIF, taken 4 s after IMG_2
    offset = 1_700_000_000 - _epoch(0)
    for p, s in ((a, 0), (b, 100), (c, 104)):
        os.utime(p, (_epoch(s) + offset,) * 2)
    infos = cg.resolve_capture_times([a, b, c])
    assert infos[2].source == "mtime"
    assert infos[2].time == pytest.approx(_epoch(104))
    assert group_paths([a, b, c]) == [["IMG_1.jpg"], ["IMG_2.jpg", "IMG_3.jpg"]]


def test_mixed_exif_and_copied_mtime_not_trusted(tmp_path):
    a = make_jpeg(tmp_path / "IMG_1.jpg", 0)
    b = make_jpeg(tmp_path / "IMG_2.jpg", 500)
    c = make_jpeg(tmp_path / "IMG_3.jpg")
    for k, p in enumerate((a, b, c)):
        os.utime(p, (1_700_000_000 + k * 0.1,) * 2)
    assert cg.resolve_capture_times([a, b, c])[2].source == "none"


def test_camera_identity_read(tmp_path):
    info = cg.read_exif_capture_time(make_jpeg(tmp_path / "a.jpg", 0, make="Apple", model="iPhone 17"))
    assert info.camera == "Apple iPhone 17"


def test_dhash_is_orientation_corrected(tmp_path):
    a = make_jpeg(tmp_path / "IMG_1.jpg", 0, size=(160, 120))
    b = tmp_path / "IMG_2.jpg"
    with Image.open(a) as image:
        image.rotate(-90, expand=True).save(b, "JPEG", exif=_exif_bytes(1, orientation=8))
    assert cg.read_exif_capture_time(b).time is not None
    assert bin(cg.dhash64(a) ^ cg.dhash64(b)).count("1") <= cg.NEAR_DUP_DHASH_MAX


def test_near_duplicate_burst_detected_but_different_label_not(tmp_path):
    a = make_jpeg(tmp_path / "IMG_1.jpg", 0, text="FRONT")
    b = tmp_path / "IMG_2.jpg"
    shutil.copy(a, b)
    c = make_jpeg(tmp_path / "IMG_3.jpg", 1, color=(20, 40, 160), text="BACK", size=(160, 120))
    ha, hb, hc = (cg.dhash64(p) for p in (a, b, c))
    assert cg.is_near_duplicate(ha, hb, 0.3)
    assert not cg.is_near_duplicate(ha, hc, 1.0)
    assert not cg.is_near_duplicate(ha, hb, 9.0)  # same look far apart: a sibling bottle


def test_unreadable_or_heic_without_plugin_is_untimed(tmp_path):
    p = tmp_path / "IMG_1.heic"
    p.write_bytes(b"\x00\x00\x00\x18ftypheic")
    assert cg.read_exif_capture_time(p).source == "none"
    assert cg.dhash64(p) is None


def test_angle_suffix_base():
    assert cg.angle_base("/x/IMG_1234_front.jpg") == cg.angle_base("/x/IMG_1234_back.jpg")
    assert cg.angle_base("/x/IMG_1234.jpg") is None
    assert cg.angle_base("/x/IMG_1234_front.jpg") != cg.angle_base("/y/IMG_1234_back.jpg")


def _score(segments, names, groups, dont_care):
    pid = {names[k]: g for g, seg in enumerate(segments) for k in seg}
    label = {f: g for g, fs in enumerate(groups) for f in fs}
    tp = fp = fn = 0
    for a, b in itertools.combinations(names, 2):
        if frozenset((a, b)) in dont_care:
            continue
        same, together = label[a] == label[b], pid[a] == pid[b]
        tp += same and together
        fp += together and not same
        fn += same and not together
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    exact = sum(
        1
        for g in groups
        if len({pid[f] for f in g}) == 1
        and not any(
            pid[n] == pid[g[0]] and n not in g and not all(frozenset((n, x)) in dont_care for x in g)
            for n in names
        )
    ) / len(groups)
    return f1, exact


def test_regression_fixture_real_capture_times():
    """Regression lock on the 98-photo session the defaults were tuned on (not a quality proof)."""
    fixture = json.loads(FIXTURE.read_text())
    names = [p["name"] for p in fixture["photos"]]
    segments = cg.segment_by_time([p["t"] for p in fixture["photos"]])
    dont_care = {frozenset(p) for p in fixture["dont_care_pairs"]}
    f1, exact = _score(segments, names, fixture["groups"], dont_care)
    assert f1 >= fixture["expected"]["min_pairwise_f1"]
    assert exact >= fixture["expected"]["min_exact_group_accuracy"]
    assert max(len(seg) for seg in segments) <= fixture["expected"]["max_group_size"]
    # The old filename rule merged whole IMG_46x0-46x9 decades; nothing like that now.
    assert len(segments) >= 45


def _synthetic_session(sizes, within, between, seed):
    rng = random.Random(seed)
    times, groups, t, k = [], [], 0.0, 0
    for size in sizes:
        group = []
        for j in range(size):
            if j:
                t += rng.uniform(*within)
            times.append(t)
            group.append(str(k))
            k += 1
        groups.append(group)
        t += rng.uniform(*between)
    return times, groups


@pytest.mark.parametrize(
    ("sizes", "cap", "min_f1"),
    [
        ([2] * 40, 3, 0.9),  # quick front/back pairs
        ([4] * 30, 4, 0.75),  # always four angles, with the cap raised to match
        ([1] * 40, 3, 0.0),  # all singles: F1 undefined, check exact below
    ],
)
def test_synthetic_cadences(sizes, cap, min_f1):
    """Loose floors so changes to the cap or tie-break cannot silently break other habits."""
    times, groups = _synthetic_session(sizes, within=(2, 6), between=(15, 60), seed=3)
    names = [n for g in groups for n in g]
    segments = cg.segment_by_time(times, max_group_size=cap)
    f1, exact = _score(segments, names, groups, set())
    assert f1 >= min_f1
    assert exact >= 0.75
