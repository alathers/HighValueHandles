"""On-disk cache for the web layer (search results, pages, robots.txt, images).

Layout under the root (by default ``${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web``)::

    search/<sha256(key)>.json      {"stored_at": <epoch>, "payload": ...}
    pages/<sha256(key)>.json
    robots/<sha256(key)>.json
    images/<sha256(key)>.<ext>     raw bytes
    images/<sha256(key)>.meta.json {"stored_at": <epoch>, "payload": {"ext": ..., ...}}

Rules:

- Keys are hashed, so file names never contain URLs or queries.
- Writes are atomic (a temp file in the same directory, then ``os.replace``).
- A missing, expired, corrupt or unreadable entry is a miss; reads never raise.
- ``refresh=True`` skips every read but still writes.
- Only public web data and the search queries are stored: never local paths,
  bottle ids, API keys or request headers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Cache namespaces (sub-directories).
NAMESPACES: tuple[str, ...] = ("search", "pages", "robots", "images")

_EXT_RE = re.compile(r"^[a-z0-9]{1,5}$")


def default_cache_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web``.

    A relative ``XDG_CACHE_HOME`` is ignored, as the XDG spec requires. The
    same rule applies on Linux and macOS.
    """
    env = os.environ if environ is None else environ
    xdg = (env.get("XDG_CACHE_HOME") or "").strip()
    if xdg and Path(xdg).expanduser().is_absolute():
        base = Path(xdg).expanduser()
    else:
        home = (env.get("HOME") or "").strip()
        base = (Path(home) if home else Path.home()) / ".cache"
    return base / "bourbon-process" / "web"


def cache_key_hash(key: str) -> str:
    """Hex sha256 of a cache key (the entry's file stem)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class WebCache:
    """Small TTL cache of JSON payloads and image bytes.

    Thread-safe for concurrent readers and writers: each write is an atomic
    rename, so a reader sees either the old or the new entry.

    Args:
        root: Cache root directory (created on first write).
        clock: Wall-clock seconds (injected in tests).
        refresh: Skip all reads (``--refresh-web``) but keep writing.
    """

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
        refresh: bool = False,
    ) -> None:
        self.root = Path(root)
        self._clock = clock
        self.refresh = refresh

    def __repr__(self) -> str:
        return f"WebCache(root={str(self.root)!r}, refresh={self.refresh})"

    # -- paths ---------------------------------------------------------------

    def _dir(self, ns: str) -> Path:
        if ns not in NAMESPACES:
            raise ValueError(f"unknown cache namespace {ns!r}; expected one of {NAMESPACES}")
        return self.root / ns

    def path_for(self, ns: str, key: str, suffix: str = ".json") -> Path:
        """Path of the entry for ``key`` in namespace ``ns``."""
        return self._dir(ns) / f"{cache_key_hash(key)}{suffix}"

    # -- JSON entries ----------------------------------------------------------

    def _fresh(self, entry: Any, ttl_seconds: Optional[float]) -> bool:  # noqa: ANN401
        if not isinstance(entry, dict) or "payload" not in entry:
            return False
        stored_at = entry.get("stored_at")
        if isinstance(stored_at, bool) or not isinstance(stored_at, (int, float)):
            return False
        if ttl_seconds is None:
            return True
        age = self._clock() - float(stored_at)
        return 0 <= age <= ttl_seconds or (age < 0 and abs(age) < 60)

    def _read_json(self, path: Path) -> Any:  # noqa: ANN401
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None

    def get(self, ns: str, key: str, ttl_seconds: Optional[float]) -> Any:  # noqa: ANN401
        """Cached payload, or None on a miss, an expired or a corrupt entry.

        ``ttl_seconds=None`` means no expiry. Always None when ``refresh``.
        """
        if self.refresh:
            return None
        entry = self._read_json(self.path_for(ns, key))
        if not self._fresh(entry, ttl_seconds):
            return None
        return entry["payload"]

    def put(self, ns: str, key: str, payload: Any) -> bool:  # noqa: ANN401
        """Store a JSON-serializable payload atomically.

        Returns:
            True when written; False (logged) on any error. Never raises.
        """
        try:
            data = json.dumps(
                {"stored_at": self._clock(), "payload": payload}, ensure_ascii=False
            ).encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.warning("web cache: payload for %s/ not serializable: %s", ns, e)
            return False
        return self._atomic_write(self.path_for(ns, key), data)

    # -- byte entries (images) -------------------------------------------------

    def get_bytes(
        self, ns: str, key: str, ttl_seconds: Optional[float]
    ) -> Optional[tuple[bytes, dict[str, Any], Path]]:
        """Cached bytes with their metadata and file path, or None on a miss."""
        if self.refresh:
            return None
        entry = self._read_json(self.path_for(ns, key, ".meta.json"))
        if not self._fresh(entry, ttl_seconds):
            return None
        meta = entry["payload"]
        if not isinstance(meta, dict):
            return None
        ext = str(meta.get("ext", ""))
        if not _EXT_RE.match(ext):
            return None
        path = self.path_for(ns, key, f".{ext}")
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if "size" in meta and meta["size"] != len(data):
            return None
        return data, meta, path

    def put_bytes(
        self, ns: str, key: str, data: bytes, *, ext: str, meta: Optional[dict[str, Any]] = None
    ) -> Optional[Path]:
        """Store raw bytes (plus a ``.meta.json``) atomically.

        Returns:
            Path of the stored file, or None (logged) on error. Never raises.
        """
        ext = ext.lower().lstrip(".")
        if not _EXT_RE.match(ext):
            logger.warning("web cache: refusing extension %r", ext)
            return None
        path = self.path_for(ns, key, f".{ext}")
        if not self._atomic_write(path, data):
            return None
        info = dict(meta or {})
        info.update({"ext": ext, "size": len(data)})
        try:
            blob = json.dumps({"stored_at": self._clock(), "payload": info}).encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.warning("web cache: image metadata not serializable: %s", e)
            return None
        if not self._atomic_write(self.path_for(ns, key, ".meta.json"), blob):
            return None
        return path

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> bool:
        tmp_name: Optional[str] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, path)
            tmp_name = None
            return True
        except OSError as e:
            logger.warning("web cache: could not write %s: %s", path.name, e)
            return False
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
