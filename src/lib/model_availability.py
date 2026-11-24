"""Model availability helpers: name parsing, download-size estimates, disk checks.

Pure functions with no server state, used by :mod:`lib.model_server` to decide
whether a missing Ollama model may be pulled:

- :func:`parse_model_ref` / :func:`normalize_model_name` follow Ollama's
  ``[host/][namespace/]model[:tag]`` naming, so ``qwen3-vl:32b``,
  ``registry.ollama.ai/library/qwen3-vl:32b`` and ``QWEN3-VL:32b`` compare
  equal while ``qwen3`` does not match ``qwen3-coder:latest``.
- :func:`fetch_pull_estimate` reads the registry manifest (a small JSON
  document; nothing is downloaded) and sums the layer sizes, minus blobs that
  are already complete in the local store.
- :func:`ollama_models_dir`, :func:`check_disk` and :func:`make_disk_guard`
  locate the model store and compare free space with the download plus a
  safety margin.
- :func:`total_memory_bytes` and :func:`memory_verdict` give an advisory
  check of a model's size against this machine's physical memory.

Only POSIX-portable facilities are used (``shutil.disk_usage``,
``platform``), so this works on macOS and Linux.

Ollama downloads into ``<store>/blobs/sha256-<hex>-partial`` and renames the
file when it is complete. A ``-partial`` file therefore never counts as
present, and this module never deletes anything: Ollama resumes partial
downloads on the next pull.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import shutil
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

#: Decimal gigabyte (what ``ollama list`` and registries report).
GB = 1_000_000_000
#: Binary gibibyte (used for the ``--min-free-gb`` floor).
GiB = 1024**3
#: Free space that must remain on the model store after a pull.
DEFAULT_MIN_FREE_BYTES = 5 * GiB
#: Extra margin as a fraction of the download (covers size inaccuracy).
DEFAULT_MARGIN_FRACTION = 0.10

#: Registry used for names without an explicit host.
DEFAULT_REGISTRY = "registry.ollama.ai"
#: Hosts that mean "the default registry" in model names.
DEFAULT_REGISTRY_ALIASES = frozenset({"registry.ollama.ai", "ollama.com"})
#: Namespace of official library models.
LIBRARY_NAMESPACE = "library"
#: Accept header for single (Docker v2) manifests.
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
#: Media type of an Ollama manifest list (one child manifest per runner).
MANIFEST_LIST_TYPE = "application/vnd.ollama.manifest.list.v2+json"
#: Most child manifests fetched when no variant matches this platform. Every
#: child is compared (the largest wins); a list with more children than this
#: is reported as "size unknown" instead of checking only a subset.
MAX_VARIANT_FETCHES = 8

#: Model store of the Linux systemd service created by Ollama's install script
#: (the service runs as the ``ollama`` user whose home is ``/usr/share/ollama``).
LINUX_SERVICE_MODELS_DIR = Path("/usr/share/ollama/.ollama/models")

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})
_USER_AGENT = "bourbon-image-processor"


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelRef:
    """A parsed Ollama model reference.

    Attributes:
        host: Registry host (``registry.ollama.ai`` by default; may carry a port).
        namespace: Namespace (``library`` for official models).
        repo: Model repository name.
        tag: Tag (``latest`` by default).
    """

    host: str
    namespace: str
    repo: str
    tag: str

    @property
    def is_default_registry(self) -> bool:
        """Whether the reference points at the default Ollama registry."""
        return self.host.lower() in DEFAULT_REGISTRY_ALIASES

    @property
    def canonical(self) -> str:
        """Shortest unambiguous form, as ``/api/tags`` lists it.

        ``repo:tag`` for library models, ``ns/repo:tag`` for other models on
        the default registry, ``host/ns/repo:tag`` otherwise.
        """
        if self.is_default_registry:
            if self.namespace == LIBRARY_NAMESPACE:
                return f"{self.repo}:{self.tag}"
            return f"{self.namespace}/{self.repo}:{self.tag}"
        return f"{self.host}/{self.namespace}/{self.repo}:{self.tag}"

    @property
    def _registry_base(self) -> str:
        host = DEFAULT_REGISTRY if self.is_default_registry else self.host
        return f"https://{host}/v2/{self.namespace}/{self.repo}"

    @property
    def manifest_url(self) -> str:
        """Registry URL of this tag's manifest."""
        return self.manifest_url_for(self.tag)

    def manifest_url_for(self, reference: str) -> str:
        """Registry URL of a manifest by tag or digest (for manifest-list children)."""
        return f"{self._registry_base}/manifests/{reference}"


def parse_model_ref(name: str) -> ModelRef:
    """Parse ``[host/][namespace/]model[:tag]`` the way Ollama does.

    The tag is taken from the last path segment only, so a ``host:port`` is
    preserved.

    Args:
        name: Model name, e.g. ``qwen3.6:35b-a3b`` or
            ``hf.co/user/repo-GGUF:Q4_K_M``.

    Returns:
        The parsed reference.

    Raises:
        ValueError: If the name is empty, uses a digest (``@``) or has more
            than three path segments.
    """
    text = name.strip() if isinstance(name, str) else ""
    if not text or "@" in text:
        raise ValueError(
            f"Unrecognised model name {name!r}: expected [host/][namespace/]model[:tag]"
        )
    path, tag = text, "latest"
    head, sep, last = text.rpartition("/")
    if ":" in last:
        last, tag = last.rsplit(":", 1)
        path = f"{head}{sep}{last}"
    parts = path.split("/")
    if any(not p for p in parts) or not tag:
        raise ValueError(
            f"Unrecognised model name {name!r}: expected [host/][namespace/]model[:tag]"
        )
    if len(parts) == 1:
        return ModelRef(DEFAULT_REGISTRY, LIBRARY_NAMESPACE, parts[0], tag)
    if len(parts) == 2:
        return ModelRef(DEFAULT_REGISTRY, parts[0], parts[1], tag)
    if len(parts) == 3:
        return ModelRef(parts[0], parts[1], parts[2], tag)
    raise ValueError(f"Unrecognised model name {name!r}: expected [host/][namespace/]model[:tag]")


def normalize_model_name(name: str) -> str:
    """Return a case-folded canonical name for exact comparisons.

    Unparseable names fall back to the stripped, case-folded input.
    """
    try:
        return parse_model_ref(name).canonical.casefold()
    except ValueError:
        return name.strip().casefold() if isinstance(name, str) else ""


# ---------------------------------------------------------------------------
# Download size
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PullEstimate:
    """How much a pull would download.

    Attributes:
        model: Requested model name.
        total_bytes: Size of all blobs in the manifest (None if unknown).
        missing_bytes: Bytes not yet complete in the local store (None if
            unknown). This is what the pull downloads.
        source: Manifest URL the estimate came from.
        error: Why the size is unknown (None if known).
        not_found: The registry definitively reported the model as missing.
    """

    model: str
    total_bytes: int | None = None
    missing_bytes: int | None = None
    source: str = ""
    error: str | None = None
    not_found: bool = False

    @property
    def known(self) -> bool:
        """Whether the download size is known."""
        return self.missing_bytes is not None


def manifest_blob_sizes(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Map digest to size over the manifest's config and layers, deduplicated.

    Ollama manifests may list the same blob twice (e.g. a license layer); it
    is stored and downloaded once.
    """
    sizes: dict[str, int] = {}
    entries: list[Any] = [manifest.get("config")]
    layers = manifest.get("layers")
    if isinstance(layers, list):
        entries.extend(layers)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        digest = entry.get("digest")
        size = entry.get("size")
        if isinstance(digest, str) and digest and isinstance(size, int) and not isinstance(size, bool):
            sizes[digest] = max(size, 0)
    return sizes


def blob_path(blobs_dir: Path, digest: str) -> Path:
    """Path of a completed blob in an Ollama store (``sha256:<hex>`` -> ``sha256-<hex>``)."""
    return blobs_dir / digest.replace(":", "-")


def blob_is_present(blobs_dir: Path, digest: str) -> bool:
    """Whether a blob is completely downloaded.

    Only the final file name counts; a ``<blob>-partial`` file (a download in
    progress or interrupted) does not. Never raises and never modifies the
    store.
    """
    try:
        return blob_path(blobs_dir, digest).is_file()
    except OSError:
        return False


def sum_manifest_sizes(manifest: Mapping[str, Any], blobs_dir: Path | None) -> tuple[int, int]:
    """Return ``(total, missing)`` bytes for a manifest.

    Args:
        manifest: A single (Docker v2 style) manifest.
        blobs_dir: Local ``blobs`` directory, or None if it cannot be
            inspected (everything counts as missing).

    Returns:
        Total size of the distinct blobs, and the part not already present.
    """
    sizes = manifest_blob_sizes(manifest)
    total = sum(sizes.values())
    if blobs_dir is None:
        return total, total
    missing = sum(size for digest, size in sizes.items() if not blob_is_present(blobs_dir, digest))
    return total, missing


def _variant_traits(entry: Mapping[str, Any]) -> set[str]:
    """Lower-cased 'runner'/'format' values of a manifest-list entry.

    The list format is not documented, so the entry itself, its ``platform``
    and its ``annotations`` are all searched, including namespaced keys such
    as ``org.ollama.runner``.
    """
    traits: set[str] = set()
    for container in (entry, entry.get("platform"), entry.get("annotations")):
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            leaf = key.rsplit(".", 1)[-1].lower()
            if leaf in ("runner", "format"):
                traits.add(value.strip().lower())
    return traits


def _choose_manifest_variant(
    entries: Iterable[Mapping[str, Any]],
    *,
    system: str | None,
    machine: str | None,
) -> Mapping[str, Any] | None:
    """Pick the child manifest Ollama would pull on this platform.

    Ollama prefers its MLX runner on Apple Silicon and llama.cpp (GGUF)
    elsewhere.

    Args:
        entries: ``manifests`` entries of a manifest list.
        system: ``platform.system()`` of the machine running Ollama, or None
            if unknown (e.g. a remote server).
        machine: ``platform.machine()`` of that machine, or None.

    Returns:
        The matching entry, or None if nothing matches (callers then take the
        largest child, which is the safe choice for a disk check).
    """
    candidates = [e for e in entries if isinstance(e, Mapping) and isinstance(e.get("digest"), str)]
    if system is None:
        return None
    apple_silicon = system == "Darwin" and (machine or "").lower() in ("arm64", "aarch64")
    preferences = (["mlx"] if apple_silicon else []) + ["llamacpp", "gguf"]
    for wanted in preferences:
        for entry in candidates:
            if wanted in _variant_traits(entry):
                return entry
    return None


def _get_manifest(session: Any, url: str, timeout: float) -> tuple[dict[str, Any] | None, PullEstimate | None]:
    """GET a manifest. Returns ``(manifest, None)`` or ``(None, failure_estimate)``.

    The failure estimate carries ``error``/``not_found`` but no model name;
    the caller fills that in.
    """
    try:
        response = session.get(
            url,
            headers={"Accept": MANIFEST_ACCEPT, "User-Agent": _USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, PullEstimate("", source=url, error=f"could not reach the registry: {e}")
    status = getattr(response, "status_code", None)
    if status == 404:
        return None, PullEstimate(
            "", source=url, error=f"not found in registry (HTTP 404 from {url})", not_found=True
        )
    if status in (401, 403):
        return None, PullEstimate(
            "", source=url, error=f"registry requires authentication (HTTP {status})"
        )
    if status != 200:
        return None, PullEstimate("", source=url, error=f"registry returned HTTP {status}")
    try:
        payload = response.json()
    except ValueError as e:
        return None, PullEstimate("", source=url, error=f"invalid manifest JSON: {e}")
    if not isinstance(payload, dict):
        return None, PullEstimate("", source=url, error="invalid manifest: not a JSON object")
    return payload, None


def _is_manifest_list(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("mediaType") == MANIFEST_LIST_TYPE or isinstance(manifest.get("manifests"), list)


def fetch_pull_estimate(
    name: str,
    *,
    blobs_dir: Path | None = None,
    session: Any = None,
    timeout: float = 10.0,
    system: str | None = "auto",
    machine: str | None = "auto",
) -> PullEstimate:
    """Estimate the download size of ``name`` from its registry manifest.

    Only manifests (small JSON documents) are fetched; no model data is
    downloaded. Never raises.

    Args:
        name: Model name.
        blobs_dir: Local ``<store>/blobs`` directory, used to subtract blobs
            that are already complete. None counts every blob as missing.
        session: Object with a ``requests``-style ``get`` (default: the
            ``requests`` module).
        timeout: Per-request timeout in seconds.
        system: Platform of the machine running Ollama, used to choose a
            manifest-list variant. ``'auto'`` means this machine; None means
            unknown (the largest variant is used).
        machine: CPU architecture, as for ``system``.

    Returns:
        A PullEstimate; ``known`` is False (with ``error`` set) on any failure.
    """
    try:
        ref = parse_model_ref(name)
    except ValueError as e:
        return PullEstimate(name, error=str(e))
    http = session if session is not None else requests
    if system == "auto":
        system = platform.system()
    if machine == "auto":
        machine = platform.machine()

    url = ref.manifest_url
    manifest, failure = _get_manifest(http, url, timeout)
    if manifest is None:
        failure = failure or PullEstimate(name, error="no manifest")
        return PullEstimate(name, source=url, error=failure.error, not_found=failure.not_found)

    if _is_manifest_list(manifest):
        entries = [e for e in manifest.get("manifests") or [] if isinstance(e, Mapping)]
        chosen = _choose_manifest_variant(entries, system=system, machine=machine)
        if chosen is None and len(entries) > MAX_VARIANT_FETCHES:
            return PullEstimate(
                name,
                source=url,
                error=(
                    f"manifest list has {len(entries)} variants and none matches this platform; "
                    "download size unknown"
                ),
            )
        children = [chosen] if chosen is not None else entries
        best: tuple[int, int] | None = None
        for child in children:
            digest = child.get("digest")
            if not isinstance(digest, str):
                continue
            child_manifest, child_failure = _get_manifest(http, ref.manifest_url_for(digest), timeout)
            if child_manifest is None:
                reason = child_failure.error if child_failure else "no manifest"
                return PullEstimate(name, source=url, error=f"manifest variant {digest}: {reason}")
            sizes = sum_manifest_sizes(child_manifest, blobs_dir)
            if best is None or sizes[1] > best[1]:
                best = sizes
        if best is None:
            return PullEstimate(name, source=url, error="manifest list has no usable variants")
        total, missing = best
    else:
        total, missing = sum_manifest_sizes(manifest, blobs_dir)

    if total <= 0:
        return PullEstimate(name, source=url, error="manifest lists no layers with sizes")
    return PullEstimate(name, total_bytes=total, missing_bytes=missing, source=url)


# ---------------------------------------------------------------------------
# Model store and free space
# ---------------------------------------------------------------------------

def _path_exists(path: Path) -> bool:
    """Whether ``path`` exists and can be stat'ed by this user.

    Uses :func:`os.stat` directly so the behaviour does not depend on the
    Python version (newer :mod:`pathlib` swallows some errors in
    ``Path.exists()``). Any ``OSError`` (missing, permission denied, not a
    directory, loops) counts as "not there".
    """
    try:
        os.stat(path)
    except (OSError, ValueError):
        return False
    return True


def _path_exists_or_hidden(path: Path) -> bool:
    """Whether ``path`` exists, counting "permission denied" as existing.

    The Linux service store lives under ``/usr/share/ollama``, which is often
    not traversable by other users. ``stat`` then fails with ``EACCES``
    (rather than ``ENOENT``), which means a parent exists but is private to
    the ``ollama`` user, so the store is almost certainly there.

    :func:`os.stat` is called directly because ``Path.exists()`` returns
    ``False`` on ``EACCES`` in newer Python versions instead of raising.
    """
    try:
        os.stat(path)
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def ollama_models_dir(
    env: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    home: Path | None = None,
) -> Path:
    """Locate the local Ollama model store.

    Resolution order:

    1. ``$OLLAMA_MODELS`` if set and non-blank (``~`` is expanded).
    2. ``~/.ollama/models`` (desktop app, ``ollama serve`` as this user).
    3. On Linux only, when ``~/.ollama/models`` does not exist but
       :data:`LINUX_SERVICE_MODELS_DIR` does (or cannot be inspected because
       the service user's home is private): the systemd service's store.

    The CLI's environment is used; a server started with a different
    ``OLLAMA_MODELS`` cannot be detected (Ollama has no API for it). Free
    space is measured on the nearest accessible ancestor (see
    :func:`existing_ancestor`), which is on the same filesystem.

    Args:
        env: Environment mapping (default: ``os.environ``).
        system: ``platform.system()`` override (for tests).
        home: Home directory override (for tests).

    Returns:
        The store path (it may not exist yet).
    """
    environ = os.environ if env is None else env
    configured = (environ.get("OLLAMA_MODELS") or "").strip()
    if configured:
        return Path(configured).expanduser()
    user_store = (Path.home() if home is None else home) / ".ollama" / "models"
    system = platform.system() if system is None else system
    if (
        system == "Linux"
        and not _path_exists(user_store)
        and _path_exists_or_hidden(LINUX_SERVICE_MODELS_DIR)
    ):
        return LINUX_SERVICE_MODELS_DIR
    return user_store


def existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its nearest existing, accessible ancestor (stops at the root)."""
    current = path.expanduser()
    while not _path_exists(current) and current != current.parent:
        current = current.parent
    return current


@dataclass(frozen=True)
class DiskCheck:
    """Free space on the model store compared with a pull.

    Attributes:
        path: Model store path.
        measured_path: Existing directory whose filesystem was measured.
        free_bytes: Free bytes on that filesystem.
        download_bytes: Bytes the pull downloads.
        margin_bytes: Safety margin that must stay free.
        min_free_bytes: The ``--min-free-gb`` floor the margin was computed
            from (0 if unknown).
    """

    path: Path
    measured_path: Path
    free_bytes: int
    download_bytes: int
    margin_bytes: int
    min_free_bytes: int = 0

    @property
    def margin_text(self) -> str:
        """Human-readable margin, in the unit the user chose it in.

        ``--min-free-gb`` is in GiB, so when that floor is the larger part
        of the margin it is shown as ``'5 GiB (--min-free-gb)'`` rather than
        the confusing decimal ``'5.4 GB'``; otherwise the 10 % share is shown
        in decimal GB like the download size.
        """
        if self.min_free_bytes > 0 and self.margin_bytes == self.min_free_bytes:
            return f"{format_gib(self.min_free_bytes)} (--min-free-gb)"
        return f"{format_gb(self.margin_bytes)} (10% of the download)"

    @property
    def required_bytes(self) -> int:
        """Download plus margin."""
        return self.download_bytes + self.margin_bytes

    @property
    def ok(self) -> bool:
        """Whether the pull fits with the margin to spare."""
        return self.free_bytes >= self.required_bytes

    @property
    def shortfall_bytes(self) -> int:
        """How much more free space is needed (0 if it fits)."""
        return max(0, self.required_bytes - self.free_bytes)


def required_margin_bytes(
    download_bytes: int,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
) -> int:
    """Safety margin: ``max(min_free_bytes, ceil(download * margin_fraction))``."""
    return max(int(min_free_bytes), math.ceil(max(download_bytes, 0) * margin_fraction))


def check_disk(
    store: Path,
    download_bytes: int,
    *,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    disk_usage: Callable[[Path], Any] | None = None,
) -> DiskCheck:
    """Measure free space for a pull into ``store``.

    Args:
        store: Model store (may not exist yet; its nearest existing ancestor
            is measured).
        download_bytes: Bytes the pull downloads.
        min_free_bytes: Minimum free space to keep after the pull.
        margin_fraction: Margin as a fraction of the download.
        disk_usage: ``shutil.disk_usage``-compatible function (default:
            ``shutil.disk_usage``, looked up at call time).

    Returns:
        The comparison.

    Raises:
        OSError: If free space cannot be measured.
    """
    measure = shutil.disk_usage if disk_usage is None else disk_usage
    measured = existing_ancestor(store)
    free = int(measure(measured).free)
    return DiskCheck(
        path=store,
        measured_path=measured,
        free_bytes=free,
        download_bytes=int(download_bytes),
        margin_bytes=required_margin_bytes(download_bytes, min_free_bytes, margin_fraction),
        min_free_bytes=int(min_free_bytes),
    )


def make_disk_guard(
    store: Path,
    min_free_bytes: int,
    *,
    disk_usage: Callable[[Path], Any] | None = None,
) -> Callable[[], str | None]:
    """Build a ``should_abort`` callback for a running pull.

    The callback returns a reason once free space on the store falls below
    ``min_free_bytes`` (e.g. because something else filled the disk), else
    None. A measurement error never aborts the pull. ``disk_usage`` defaults
    to ``shutil.disk_usage`` (looked up at call time).
    """

    def guard() -> str | None:
        measure = shutil.disk_usage if disk_usage is None else disk_usage
        measured = existing_ancestor(store)
        try:
            free = int(measure(measured).free)
        except OSError as e:
            logger.debug("Disk guard could not measure %s: %s", measured, e)
            return None
        if free < min_free_bytes:
            return (
                f"free space on {measured} fell to {format_gb(free)} "
                f"(--min-free-gb is {format_gib(min_free_bytes)})"
            )
        return None

    return guard


# ---------------------------------------------------------------------------
# Memory (advisory)
# ---------------------------------------------------------------------------

#: Share of physical memory above which a model triggers a warning.
MEMORY_WARN_FRACTION = 0.75


def total_memory_bytes() -> int | None:
    """Physical memory of this machine in bytes, or None if unknown.

    Uses ``os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')``, which
    works on Linux and macOS.
    """
    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError, TypeError):
        return None
    if page <= 0 or pages <= 0:
        return None
    return page * pages


def memory_verdict(model_bytes: int, total: int | None) -> tuple[str, str]:
    """Compare a model's size with physical memory.

    The size is the model's weights as ``/api/tags`` reports them; the KV
    cache and the OS need more, so this is advisory only.

    Args:
        model_bytes: Model size in bytes.
        total: Physical memory in bytes (None if unknown).

    Returns:
        ``(verdict, message)`` where verdict is ``'error'`` (the model is
        larger than memory), ``'warn'`` (more than 75 % of memory), ``'ok'``
        or ``'unknown'`` (memory could not be measured).
    """
    if total is None or total <= 0:
        return "unknown", f"model needs {format_gb(model_bytes)}; physical memory could not be measured"
    share = model_bytes / total
    detail = f"model needs {format_gb(model_bytes)} of {format_gb(total)} physical memory ({share:.0%})"
    if model_bytes > total:
        return "error", detail
    if share > MEMORY_WARN_FRACTION:
        return "warn", detail
    return "ok", detail


# ---------------------------------------------------------------------------
# Formatting and host classification
# ---------------------------------------------------------------------------

def format_gb(n: int | float) -> str:
    """Format a byte count in decimal gigabytes, e.g. ``'20.9 GB'``."""
    return f"{n / GB:.1f} GB"


def format_gib(n: int | float) -> str:
    """Format a byte count in binary gibibytes, e.g. ``'5 GiB'`` or ``'2.5 GiB'``.

    Used for ``--min-free-gb`` values, which are given in GiB.
    """
    return f"{round(n / GiB, 1):g} GiB"


def is_local_url(base_url: str) -> bool:
    """Whether ``base_url`` points at this machine (so its disk is ours to measure)."""
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    if host in _LOCAL_HOSTS:
        return True
    try:
        return host == socket.gethostname().lower()
    except OSError:
        return False
