"""Tests for lib.web_cache."""

import os
from pathlib import Path

import pytest

from lib.web_cache import WebCache, default_cache_dir


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_put_get_and_ttl(tmp_path: Path):
    clock = FakeClock()
    cache = WebCache(tmp_path / "web", clock=clock)
    assert cache.put("search", "brave\nq", [{"url": "https://a.example"}])
    assert cache.get("search", "brave\nq", 60) == [{"url": "https://a.example"}]
    clock.t += 59
    assert cache.get("search", "brave\nq", 60) is not None
    clock.t += 2
    assert cache.get("search", "brave\nq", 60) is None
    assert cache.get("search", "brave\nq", None) is not None  # no expiry


def test_miss_and_unknown_namespace(tmp_path: Path):
    cache = WebCache(tmp_path)
    assert cache.get("pages", "nope", 10) is None
    with pytest.raises(ValueError, match="namespace"):
        cache.get("llm", "x", 10)


def test_keys_are_hashed_and_write_is_atomic(tmp_path: Path):
    cache = WebCache(tmp_path)
    cache.put("pages", "https://example.com/a?b=c", {"x": 1})
    files = list((tmp_path / "pages").iterdir())
    assert len(files) == 1
    assert "example" not in files[0].name
    assert files[0].suffix == ".json"
    assert not [f for f in files if f.name.startswith(".tmp-")]


def test_failed_write_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = WebCache(tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    assert cache.put("pages", "k", {"x": 1}) is False
    assert list((tmp_path / "pages").iterdir()) == []


def test_corrupt_entry_is_a_miss(tmp_path: Path):
    cache = WebCache(tmp_path)
    cache.put("robots", "k", {"a": 1})
    cache.path_for("robots", "k").write_text("{not json", encoding="utf-8")
    assert cache.get("robots", "k", 100) is None
    cache.path_for("robots", "k").write_text('{"payload": 1}', encoding="utf-8")  # no stored_at
    assert cache.get("robots", "k", 100) is None


def test_refresh_skips_reads_but_writes(tmp_path: Path):
    WebCache(tmp_path).put("search", "k", ["old"])
    fresh = WebCache(tmp_path, refresh=True)
    assert fresh.get("search", "k", 100) is None
    fresh.put("search", "k", ["new"])
    assert WebCache(tmp_path).get("search", "k", 100) == ["new"]


def test_unserializable_payload_returns_false(tmp_path: Path):
    assert WebCache(tmp_path).put("search", "k", {"x": object()}) is False


def test_bytes_round_trip_and_ttl(tmp_path: Path):
    clock = FakeClock()
    cache = WebCache(tmp_path, clock=clock)
    path = cache.put_bytes("images", "https://x.example/a.jpg", b"\xff\xd8data", ext="jpg",
                           meta={"content_type": "image/jpeg"})  # fmt: skip
    assert path is not None
    assert path.suffix == ".jpg"
    data, meta, got_path = cache.get_bytes("images", "https://x.example/a.jpg", 10)
    assert data == b"\xff\xd8data"
    assert meta["content_type"] == "image/jpeg"
    assert got_path == path
    clock.t += 11
    assert cache.get_bytes("images", "https://x.example/a.jpg", 10) is None
    assert cache.put_bytes("images", "k", b"x", ext="../evil") is None


def test_default_cache_dir_honours_xdg(tmp_path: Path):
    env = {"XDG_CACHE_HOME": str(tmp_path / "xc"), "HOME": str(tmp_path / "home")}
    assert default_cache_dir(env) == tmp_path / "xc" / "bourbon-process" / "web"


def test_default_cache_dir_falls_back_to_home_cache(tmp_path: Path):
    env = {"HOME": str(tmp_path / "home")}
    assert default_cache_dir(env) == tmp_path / "home" / ".cache" / "bourbon-process" / "web"
    # A relative XDG_CACHE_HOME is ignored (XDG spec).
    env["XDG_CACHE_HOME"] = "relative/dir"
    assert default_cache_dir(env) == tmp_path / "home" / ".cache" / "bourbon-process" / "web"


def test_default_cache_dir_uses_process_env(tmp_path: Path):
    # conftest points XDG_CACHE_HOME under tmp_path
    assert default_cache_dir() == tmp_path / "xdg_cache" / "bourbon-process" / "web"
