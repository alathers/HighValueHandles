"""Model server lifecycle management.

Starts (or adopts), health-checks, warms and shuts down the local model server
the pipeline talks to. The only backend is :class:`OllamaServer`, which talks
to Ollama's REST endpoints (``/api/version``, ``/api/tags``, ``/api/ps``,
``/api/generate``) and, if Ollama is not running on a local URL, can spawn
``ollama serve``. Ollama runs on macOS and Linux, and the lifecycle code here
only uses POSIX facilities.

Ownership rules (see :class:`Ownership`):

- A server that already answers its identity probe is adopted as
  ``EXTERNAL``. It is never stopped or signalled by this program.
- A server this program spawns is ``OWNED``: its output goes to a log file in
  :func:`lib.process_supervisor.default_log_dir`, it runs in its own process
  group, it is reaped at interpreter exit, and :meth:`BaseModelServer.stop`
  terminates the whole group (SIGTERM, then SIGKILL after a grace period).

Models are never downloaded implicitly: :meth:`ModelServerManager.ensure_model`
only pulls when called with ``allow_pull=True``, and then only after checking
the download size (from the registry manifest) against free space on the model
store (see :mod:`lib.model_availability`). Missing, non-vision or too-large
models produce an error that lists the installed vision-capable models and the
exact ``ollama pull`` command.

Usage::

    from lib.model_server import ModelServerManager

    manager = ModelServerManager()
    if manager.ensure_running() and manager.ensure_model(
        "qwen3.6:35b-a3b", capability="vision", warm=True
    ):
        client = manager.get_client()
        result = client.chat_with_images(prompt, [image_path], model="qwen3.6:35b-a3b")
        print(result.content)
    manager.close()  # stops the server only if this program started it
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from lib.exceptions import ModelServerError
from lib.model_availability import (
    DEFAULT_MIN_FREE_BYTES,
    LINUX_SERVICE_MODELS_DIR,
    DiskCheck,
    PullEstimate,
    check_disk,
    fetch_pull_estimate,
    format_gb,
    is_local_url,
    make_disk_guard,
    normalize_model_name,
    ollama_models_dir,
)
from lib.process_supervisor import (
    ManagedProcess,
    ProbeResult,
    default_log_dir,
    port_in_use,
    wait_until_ready,
)

logger = logging.getLogger(__name__)

#: Default Ollama endpoint.
DEFAULT_OLLAMA_URL = "http://localhost:11434"
#: Per-request timeout (seconds) for inference clients. One warm vision call on
#: a full-resolution photo was measured at ~43s; a cold load adds minutes.
DEFAULT_REQUEST_TIMEOUT = 600

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ServerStartError(RuntimeError):
    """A server cannot be spawned as configured (raised by ``_spawn_argv``)."""


# ---------------------------------------------------------------------------
# Enumerations and state
# ---------------------------------------------------------------------------

class Ownership(Enum):
    """Who is responsible for a server process.

    Attributes:
        NONE: No server is known to be running.
        EXTERNAL: A server was already running; it is never stopped by us.
        OWNED: We spawned the server; we stop it on close/exit.
    """

    NONE = "none"
    EXTERNAL = "external"
    OWNED = "owned"


@dataclass
class ServerState:
    """Snapshot of the model server's status.

    Attributes:
        is_running: Whether the server answered its identity probe.
        base_url: HTTP URL for API requests.
        pid: Process ID if we own the server (0 otherwise).
        models_loaded: Models currently resident in memory.
        memory_usage_mb: Approximate (V)RAM used by loaded models (0 if unknown).
        cold_start_seconds: Time the last start took (0 if adopted).
        uptime_seconds: Seconds since we started the server (0 if adopted).
        error_message: Last lifecycle error, if any.
        owned: Whether this program started (and will stop) the server.
        version: Server version string, if reported.
        log_path: Log file of an owned server ('' otherwise).
    """

    is_running: bool = False
    base_url: str = ""
    pid: int = 0
    models_loaded: list[str] = field(default_factory=list)
    memory_usage_mb: int = 0
    cold_start_seconds: float = 0.0
    uptime_seconds: float = 0.0
    error_message: Optional[str] = None
    owned: bool = False
    version: str = ""
    log_path: str = ""


# ---------------------------------------------------------------------------
# Abstract server
# ---------------------------------------------------------------------------

class BaseModelServer(ABC):
    """Template for the model server (lifecycle algorithm plus hooks).

    :meth:`start` implements one algorithm: adopt a server
    that already answers, otherwise (if allowed and the port is free) spawn
    one, wait for it to become ready and verify it. Subclasses provide the
    hooks (``_identity_probe``, ``_can_spawn``, ``_spawn_argv``,
    ``_spawn_env``) and the model operations. Ollama (:class:`OllamaServer`)
    is the only implementation; the split keeps the lifecycle algorithm
    separate from the Ollama HTTP API.

    Liveness is :meth:`health_check`; there is deliberately no cached
    ``is_running`` attribute on servers.

    Attributes:
        host: Host the server listens on.
        port: TCP port.
        base_url: HTTP base URL.
        startup_timeout: Seconds to wait for a spawned server to be ready.
        stop_grace_seconds: Seconds between SIGTERM and SIGKILL on stop.
        ownership: Whether the server is external or owned by us.
        last_error: Actionable description of the last failure (or None).
        model_listing_failed: True if the last listing of installed models
            failed (server error), as opposed to an empty listing.
    """

    _display_name = "model server"
    _port_hint = "the port option"
    _install_hint = ""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        base_url: str,
        startup_timeout: float,
        stop_grace_seconds: float,
    ):
        """Initialize lifecycle bookkeeping."""
        self.host = host
        self.port = port
        self.base_url = base_url.rstrip("/")
        self.startup_timeout = startup_timeout
        self.stop_grace_seconds = stop_grace_seconds
        self.ownership = Ownership.NONE
        self.last_error: Optional[str] = None
        # True when the last attempt to list installed models failed (server
        # error or unreachable), as opposed to an empty/complete listing.
        self.model_listing_failed = False
        self.version = ""
        self.start_time = 0.0
        self.cold_start_seconds = 0.0
        self._proc: Optional[ManagedProcess] = None
        self._atexit_registered = False

    # -- identity ----------------------------------------------------------

    @property
    def process(self) -> Optional[subprocess.Popen]:
        """Popen handle of an owned server, or None."""
        return self._proc.popen if self._proc is not None else None

    @property
    def pid(self) -> int:
        """PID of an owned server (0 if none)."""
        return self._proc.pid if self._proc is not None else 0

    @property
    def log_path(self) -> str:
        """Log file of an owned server ('' if none)."""
        return str(self._proc.log_path) if self._proc is not None else ""

    # -- hooks -------------------------------------------------------------

    @abstractmethod
    def _identity_probe(self) -> ProbeResult:
        """Probe ``base_url`` and report whether it is this kind of server."""

    @abstractmethod
    def _can_spawn(self) -> tuple[bool, str]:
        """Return ``(True, '')`` if a local server may be spawned, else ``(False, why)``."""

    @abstractmethod
    def _spawn_argv(self) -> list[str]:
        """Command line for spawning the server.

        Raises:
            ServerStartError: If the server cannot be launched as configured.
        """

    def _spawn_env(self) -> dict[str, str]:
        """Environment for the spawned server (inherits ours; HOME untouched)."""
        return os.environ.copy()

    def _stop_hint(self) -> str:
        """How a user can stop a detached owned server."""
        return f"kill -TERM -{self.pid}"

    # -- lifecycle ---------------------------------------------------------

    def can_start(self) -> tuple[bool, str]:
        """Whether :meth:`start` may launch a local server: ``(True, '')`` or ``(False, why)``.

        Never starts anything; used by dry runs to predict a real run.
        """
        try:
            return self._can_spawn()
        except Exception as e:
            return False, f"cannot check whether a server can be started: {e}"

    def health_check(self) -> bool:
        """Return True if the server answers its identity probe. Never raises."""
        try:
            return bool(self._identity_probe().ready)
        except Exception:
            return False

    def start(self) -> bool:
        """Make sure the server is running, adopting or spawning it.

        Idempotent and never raises. On failure :attr:`last_error` holds an
        actionable message.

        Returns:
            True if the server is ready for requests.
        """
        self.last_error = None

        if self.ownership is Ownership.OWNED and self._proc is not None:
            if self._proc.is_alive():
                return self.health_check() or self._await_owned_ready()
            logger.warning(
                "%s (pid %d) exited unexpectedly; see %s",
                self._display_name, self._proc.pid, self._proc.log_path,
            )
            self._proc.terminate(self.stop_grace_seconds)  # reap stragglers, close the log
            self._forget_owned()

        probe = self._safe_probe()
        if probe.ready:
            if self.ownership is not Ownership.EXTERNAL:
                logger.info(
                    "Using already-running %s at %s (not managed by this run)",
                    self._display_name, self.base_url,
                )
            self.ownership = Ownership.EXTERNAL
            return True

        allowed, why = self._can_spawn()
        if not allowed:
            self.last_error = (
                f"{self._display_name} is not reachable at {self.base_url} ({probe.detail}) and {why}"
            )
            return False

        if port_in_use(self.host, self.port):
            self.last_error = (
                f"Port {self.port} on {self.host} is in use by a process that is not answering "
                f"as {self._display_name} ({probe.detail}). Stop that process or choose another "
                f"port with {self._port_hint}."
            )
            return False

        try:
            argv = self._spawn_argv()
        except ServerStartError as e:
            self.last_error = str(e)
            return False

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = default_log_dir() / f"ollama-{self.port}-{timestamp}.log"
        proc = ManagedProcess(argv, log_path=log_path, env=self._spawn_env(), name=self._display_name)
        logger.info("Starting %s: %s", self._display_name, " ".join(argv))
        try:
            proc.spawn()
        except FileNotFoundError:
            self.last_error = f"{self._display_name} executable not found ({argv[0]}). {self._install_hint}"
            return False
        except OSError as e:
            self.last_error = f"Could not start {self._display_name}: {e}"
            return False

        self._proc = proc
        self.ownership = Ownership.OWNED
        self.start_time = time.time()
        self._register_atexit()
        logger.info(
            "Started %s (pid %d); log: %s", self._display_name, proc.pid, proc.log_path
        )
        return self._await_owned_ready()

    def _await_owned_ready(self) -> bool:
        """Wait for an owned server to become ready; stop it on failure."""
        proc = self._proc
        if proc is None:
            return False
        result = wait_until_ready(
            self._safe_probe, timeout=self.startup_timeout, is_alive=proc.is_alive
        )
        if result.ready:
            self.cold_start_seconds = time.time() - self.start_time
            logger.info(
                "%s is ready at %s after %.1fs",
                self._display_name, self.base_url, self.cold_start_seconds,
            )
            return True

        exit_code = proc.returncode
        tail = proc.tail_log().rstrip()
        exit_note = f" (exit code {exit_code})" if exit_code is not None else ""
        self.last_error = (
            f"{self._display_name} failed to become ready: {result.detail}{exit_note}. "
            f"Last log lines:\n{tail or '(log is empty)'}\nFull log: {proc.log_path}"
        )
        self.stop()
        return False

    def stop(self) -> bool:
        """Stop the server if this program owns it; otherwise do nothing.

        An external server is never contacted or signalled.

        Returns:
            True unless an owned server could not be reaped.
        """
        if self.ownership is not Ownership.OWNED or self._proc is None:
            if self.ownership is Ownership.EXTERNAL:
                logger.debug(
                    "Leaving external %s at %s running", self._display_name, self.base_url
                )
            return True
        logger.info("Stopping %s (pid %d)...", self._display_name, self._proc.pid)
        ok = self._proc.terminate(self.stop_grace_seconds)
        self._forget_owned()
        return ok

    def detach(self) -> None:
        """Leave an owned server running after this program exits (``--keep-server``)."""
        if self.ownership is not Ownership.OWNED or self._proc is None:
            return
        logger.info(
            "Leaving %s running (pid %d, log %s); stop it later with: %s",
            self._display_name, self._proc.pid, self._proc.log_path, self._stop_hint(),
        )
        self._proc.detach()
        self._proc = None
        self._unregister_atexit()
        self.ownership = Ownership.EXTERNAL

    def _forget_owned(self) -> None:
        self._unregister_atexit()
        self._proc = None
        self.ownership = Ownership.NONE
        self.start_time = 0.0

    def _safe_probe(self) -> ProbeResult:
        try:
            return self._identity_probe()
        except Exception as e:  # probes must never break the lifecycle
            return ProbeResult(False, f"probe failed: {e}")

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_stop)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_stop)
            self._atexit_registered = False

    def _atexit_stop(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    # -- model operations --------------------------------------------------

    @abstractmethod
    def get_state(self) -> ServerState:
        """Return a status snapshot (never starts anything)."""

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """Return True if ``model_name`` can be served without downloading."""

    @abstractmethod
    def pull_model(
        self,
        model_name: str,
        *,
        should_abort: Optional[Callable[[], Optional[str]]] = None,
    ) -> bool:
        """Download a model. Only called when the caller explicitly allows it.

        ``should_abort() -> str | None`` is polled during the download and
        returns a reason to stop (e.g. the disk guard).
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """List models available without downloading."""

    # Non-abstract defaults: a server that cannot answer these reports
    # "unknown", which makes ModelServerManager refuse automatic pulls.

    def model_store_path(self) -> Optional[Path]:
        """Local directory the server stores models in, or None if unknown/remote."""
        return None

    def estimate_pull(self, model_name: str) -> PullEstimate:
        """Estimate how much pulling ``model_name`` would download (default: unknown)."""
        return PullEstimate(
            model_name,
            error=f"size estimation is not supported by this {self._display_name}",
        )

    def get_capabilities(self, model_name: str) -> Optional[set[str]]:  # noqa: ARG002
        """Capabilities of an installed model (e.g. ``{'completion', 'vision'}``), or None if unknown."""
        return None

    def installed_model_size(self, model_name: str) -> Optional[int]:  # noqa: ARG002
        """Size in bytes of an installed model, or None if unknown."""
        return None

    def list_vision_models(self) -> list[str]:
        """Installed models that accept images (default: none known)."""
        return []

    def manual_pull_command(self, model_name: str) -> str:
        """Shell command a user can run to download ``model_name`` themselves."""
        return f"ollama pull {shlex.quote(model_name)}"

    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """Load a model into memory now (warm start)."""

    @abstractmethod
    def unload_model(self, model_name: str) -> bool:
        """Release a model from memory."""

    @abstractmethod
    def get_client(self) -> Any:  # noqa: ANN401
        """Return an API client for this server (e.g. an OllamaClient)."""


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaServer(BaseModelServer):
    """Ollama backend.

    Adopts a running Ollama (the desktop app, a systemd service or a manual
    ``ollama serve``) as external, or spawns
    ``ollama serve`` when the URL is local and auto-start is enabled.

    Attributes:
        timeout: Per-request timeout for inference clients (seconds).
        model: Default model for clients.
        auto_start: Whether ``ollama serve`` may be spawned.
        keep_alive: How long a warmed model stays resident (Ollama duration).
        probe_timeout: Timeout of the ``/api/version`` identity probe.
    """

    _display_name = "Ollama"
    _port_hint = "--ollama-url"
    _install_hint = "Install it from https://ollama.com/download (macOS and Linux)."

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        model: Optional[str] = None,
        auto_start: bool = True,
        keep_alive: str = "30m",
        startup_timeout: float = 30.0,
        probe_timeout: float = 2.0,
    ):
        """Initialize the Ollama backend.

        Args:
            base_url: Ollama URL.
            timeout: Per-request timeout for inference clients (seconds).
            model: Default model for clients.
            auto_start: Allow spawning ``ollama serve`` for a local URL.
            keep_alive: Residency duration used when warming a model.
            startup_timeout: Seconds to wait for a spawned server.
            probe_timeout: Identity probe timeout.

        Raises:
            ValueError: If ``base_url`` has an invalid port.
        """
        base_url = base_url.rstrip("/")
        parsed = urlparse(base_url)
        super().__init__(
            host=parsed.hostname or "localhost",
            port=parsed.port or 11434,
            base_url=base_url,
            startup_timeout=startup_timeout,
            stop_grace_seconds=10.0,
        )
        self.timeout = timeout
        self.model = model
        self.auto_start = auto_start
        self.keep_alive = keep_alive
        self.probe_timeout = probe_timeout
        self._binary: Optional[str] = None

    # -- lifecycle hooks ---------------------------------------------------

    def _identity_probe(self) -> ProbeResult:
        """``GET /api/version``: 200 with a ``version`` field means Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=self.probe_timeout)
        except requests.Timeout:
            return ProbeResult(False, "probe timed out")
        except requests.RequestException:
            return ProbeResult(False, "connection refused")
        if response.status_code != 200:
            return ProbeResult(False, f"HTTP {response.status_code} from /api/version")
        try:
            payload = response.json()
        except ValueError:
            return ProbeResult(False, "unexpected /api/version response")
        if not isinstance(payload, dict) or "version" not in payload:
            return ProbeResult(False, "unexpected /api/version response")
        self.version = str(payload.get("version", ""))
        return ProbeResult(True, f"Ollama {self.version}", payload)

    def _can_spawn(self) -> tuple[bool, str]:
        if not self.auto_start:
            return False, "auto-start is disabled (--no-server-autostart)"
        if self.host not in _LOOPBACK_HOSTS:
            return False, (
                f"{self.host} is not this machine, so it cannot be started automatically; "
                "start Ollama there or fix --ollama-url"
            )
        self._binary = shutil.which(os.environ.get("OLLAMA_BIN", "ollama"))
        if self._binary is None:
            return False, (
                "the 'ollama' executable is not on PATH; install it from "
                "https://ollama.com/download, or start Ollama yourself (desktop app, "
                "`ollama serve`, or the systemd service on Linux)"
            )
        if self._store_is_linux_service():
            # An `ollama serve` started by this user would use ~/.ollama/models,
            # not the service's store: installed models would look missing, a
            # --pull would download a second copy and the disk check would
            # measure the wrong filesystem.
            return False, (
                "Ollama is installed as a systemd service (models in "
                f"{LINUX_SERVICE_MODELS_DIR}); start it with `sudo systemctl start ollama`, "
                "or set OLLAMA_MODELS / run `ollama serve` yourself"
            )
        return True, ""

    @staticmethod
    def _store_is_linux_service() -> bool:
        """True if the store in use is the Linux service's and OLLAMA_MODELS is unset."""
        if (os.environ.get("OLLAMA_MODELS") or "").strip():
            return False
        try:
            return ollama_models_dir() == LINUX_SERVICE_MODELS_DIR
        except Exception:
            return False

    def _spawn_argv(self) -> list[str]:
        binary = self._binary or shutil.which(os.environ.get("OLLAMA_BIN", "ollama")) or "ollama"
        return [binary, "serve"]

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        host = f"[{self.host}]" if ":" in self.host else self.host
        env["OLLAMA_HOST"] = f"{host}:{self.port}"  # bind where we will probe
        return env

    def _stop_hint(self) -> str:
        return f"kill {self.pid}"

    # -- state -------------------------------------------------------------

    def _ps(self) -> list[dict[str, Any]]:
        """Models resident in memory (``GET /api/ps``); [] on error."""
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m for m in models if isinstance(m, dict)]
        except (requests.RequestException, ValueError, AttributeError):
            pass
        return []

    def get_state(self) -> ServerState:
        """Return a status snapshot; ``models_loaded`` comes from ``/api/ps``."""
        state = ServerState(
            is_running=self.health_check(),
            base_url=self.base_url,
            pid=self.pid,
            error_message=self.last_error,
            owned=self.ownership is Ownership.OWNED,
            version=self.version,
            log_path=self.log_path,
            cold_start_seconds=self.cold_start_seconds,
        )
        if state.is_running:
            loaded = self._ps()
            state.models_loaded = [str(m.get("name", "")) for m in loaded]
            state.memory_usage_mb = int(self._vram_mb(loaded))
            if self.start_time > 0:
                state.uptime_seconds = time.time() - self.start_time
        return state

    @staticmethod
    def _vram_mb(loaded: list[dict[str, Any]]) -> float:
        total = 0
        for m in loaded:
            value = m.get("size_vram", 0)
            if isinstance(value, (int, float)):
                total += value
        return total / 2**20

    # -- models ------------------------------------------------------------

    def _tags(self) -> list[dict[str, Any]]:
        """Installed models (``GET /api/tags``); [] on any error.

        Sets :attr:`model_listing_failed` so callers can tell "the server
        could not be asked" apart from "the model is not installed".
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if isinstance(models, list):
                    self.model_listing_failed = False
                    return [m for m in models if isinstance(m, dict)]
        except (requests.RequestException, ValueError, AttributeError):
            pass
        self.model_listing_failed = True
        return []

    @staticmethod
    def _tag_matches(entry: dict[str, Any], target: str) -> bool:
        """Whether a ``/api/tags`` entry is the model whose normalized name is ``target``."""
        return any(
            isinstance(entry.get(key), str) and normalize_model_name(entry[key]) == target
            for key in ("name", "model")
        )

    def _find_tag(self, model_name: str, tags: Optional[list[dict[str, Any]]] = None) -> Optional[dict[str, Any]]:
        target = normalize_model_name(model_name)
        for entry in self._tags() if tags is None else tags:
            if self._tag_matches(entry, target):
                return entry
        return None

    def is_model_available(self, model_name: str) -> bool:
        """Check whether a model is installed (``GET /api/tags``).

        Matching is exact after normalization: ``Drews54/llama3.2-vision``
        matches ``Drews54/llama3.2-vision:latest`` but ``qwen3`` does not
        match ``qwen3-coder:latest``.

        Args:
            model_name: Model tag to check.

        Returns:
            True if the model is installed.
        """
        return self._find_tag(model_name) is not None

    def _show_capabilities(self, model_name: str) -> Optional[set[str]]:
        """Capabilities from ``POST /api/show``; None on error or if not reported."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show", json={"model": model_name}, timeout=10
            )
            if response.status_code != 200:
                return None
            caps = response.json().get("capabilities")
        except (requests.RequestException, ValueError, AttributeError):
            return None
        if isinstance(caps, list):
            return {str(c).lower() for c in caps}
        return None

    @staticmethod
    def _entry_capabilities(entry: dict[str, Any]) -> Optional[set[str]]:
        caps = entry.get("capabilities")
        if isinstance(caps, list):
            return {str(c).lower() for c in caps}
        return None

    def get_capabilities(self, model_name: str) -> Optional[set[str]]:
        """Capabilities of an installed model.

        Uses the ``capabilities`` field of ``/api/tags`` (recent Ollama
        versions) and falls back to ``POST /api/show``.

        Args:
            model_name: Installed model name.

        Returns:
            Lower-case capability names (e.g. ``{'completion', 'vision'}``),
            or None if Ollama does not report them.
        """
        entry = self._find_tag(model_name)
        caps = self._entry_capabilities(entry) if entry is not None else None
        if caps is not None:
            return caps
        return self._show_capabilities(str(entry.get("name")) if entry and entry.get("name") else model_name)

    def installed_model_size(self, model_name: str) -> Optional[int]:
        """Size in bytes of an installed model (the ``/api/tags`` ``size``).

        Args:
            model_name: Model tag.

        Returns:
            The size, or None if the model is not installed, the server
            cannot be asked, or it does not report a size.
        """
        entry = self._find_tag(model_name)
        size = entry.get("size") if entry is not None else None
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
            return None
        return int(size)

    def list_vision_models(self) -> list[str]:
        """Installed models whose capabilities include ``vision``, in ``/api/tags`` order."""
        names: list[str] = []
        for entry in self._tags():
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            caps = self._entry_capabilities(entry)
            if caps is None:
                caps = self._show_capabilities(name)
            if caps and "vision" in caps:
                names.append(name)
        return names

    def model_store_path(self) -> Optional[Path]:
        """The local Ollama model store, or None when the server is remote.

        See :func:`lib.model_availability.ollama_models_dir` for how the
        path is resolved (``$OLLAMA_MODELS``, ``~/.ollama/models``, the Linux
        systemd service store).
        """
        if not is_local_url(self.base_url):
            return None
        return ollama_models_dir()

    def estimate_pull(self, model_name: str) -> PullEstimate:
        """Download size of ``model_name`` from its registry manifest.

        Blobs already complete in the local store are subtracted. For a
        remote server the local store says nothing, so every blob counts and
        the manifest variant is not guessed from this machine's platform.
        """
        store = self.model_store_path()
        if store is None:
            return fetch_pull_estimate(model_name, system=None, machine=None)
        return fetch_pull_estimate(model_name, blobs_dir=store / "blobs")

    def pull_model(
        self,
        model_name: str,
        *,
        progress: Optional[Callable[[str, int, int], None]] = None,
        should_abort: Optional[Callable[[], Optional[str]]] = None,
        read_timeout: float = 600.0,
        progress_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> bool:
        """Download a model from its registry with a streaming ``POST /api/pull``.

        Only reached through ``ModelServerManager.ensure_model(allow_pull=True)``.
        Progress is reported at most every ``progress_interval`` seconds and
        on status changes. Closing the connection (on abort, error or Ctrl-C)
        makes Ollama stop the download; it keeps the partial blobs and
        resumes them on the next pull. Nothing is ever deleted here.

        Args:
            model_name: Model name to pull.
            progress: ``progress(status, completed_bytes, total_bytes)``;
                default logs at INFO.
            should_abort: Polled every ``progress_interval`` seconds; a
                returned reason aborts the pull (e.g. the disk guard).
            read_timeout: Seconds without data before giving up (digest
                verification of a large blob is silent for a while).
            progress_interval: Seconds between progress reports and abort polls.
            clock: Monotonic clock (for tests).

        Returns:
            True only if Ollama reported ``success``.

        Raises:
            KeyboardInterrupt: Re-raised after closing the connection.
        """
        response = None
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": model_name, "stream": True},
                stream=True,
                timeout=(10, read_timeout),
            )
            if response.status_code != 200:
                logger.error(
                    "Pull of '%s' failed: HTTP %s %s",
                    model_name, response.status_code, (response.text or "")[:500],
                )
                return False
            return self._consume_pull_stream(
                model_name, response, progress, should_abort, progress_interval, clock
            )
        except requests.RequestException as e:
            logger.error("Pull of '%s' failed: %s", model_name, e)
            return False
        except KeyboardInterrupt:
            logger.warning(
                "Pull of '%s' interrupted; the partial download is kept and will resume next time",
                model_name,
            )
            raise
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def _consume_pull_stream(
        self,
        model_name: str,
        response: Any,  # noqa: ANN401
        progress: Optional[Callable[[str, int, int], None]],
        should_abort: Optional[Callable[[], Optional[str]]],
        interval: float,
        clock: Callable[[], float],
    ) -> bool:
        """Read ``/api/pull`` NDJSON events until success, error or end of stream."""
        totals: dict[str, int] = {}
        completed: dict[str, int] = {}
        last_status = ""
        start = clock()
        last_report = start
        last_guard = start

        def report(status: str) -> None:
            done, total = sum(completed.values()), sum(totals.values())
            if progress is not None:
                progress(status, done, total)
            elif total:
                logger.info(
                    "Pulling %s: %s / %s (%d%%) [%s]",
                    model_name, format_gb(done), format_gb(total), int(100 * done / total), status,
                )
            else:
                logger.info("Pulling %s: %s", model_name, status)

        for raw in response.iter_lines():
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                logger.debug("Ignoring malformed pull event: %r", raw)
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                logger.error("Pull of '%s' failed: %s", model_name, event["error"])
                return False
            status = str(event.get("status", ""))
            digest = event.get("digest")
            if isinstance(digest, str) and digest:
                if isinstance(event.get("total"), int):
                    totals[digest] = event["total"]
                if isinstance(event.get("completed"), int):
                    completed[digest] = event["completed"]
            if status == "success":
                report(status)
                return True

            now = clock()
            if status != last_status or now - last_report >= interval:
                report(status)
                last_status, last_report = status, now
            if should_abort is not None and now - last_guard >= interval:
                last_guard = now
                reason = should_abort()
                if reason:
                    logger.error(
                        "Aborting pull of '%s': %s (the partial download is kept)", model_name, reason
                    )
                    return False

        logger.error("Pull of '%s' ended without a success status", model_name)
        return False

    def list_models(self) -> list[str]:
        """List installed models (``GET /api/tags``).

        Returns:
            List of model names.
        """
        return [str(m["name"]) for m in self._tags() if m.get("name")]

    def load_model(self, model_name: str) -> bool:
        """Load a model now so the first image does not pay the cold start.

        Sends ``POST /api/generate {model, keep_alive, stream: false}`` with no
        prompt, which loads the model and returns immediately once resident.

        Args:
            model_name: Model to load.

        Returns:
            True if Ollama reported success.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": self.keep_alive, "stream": False},
                timeout=max(self.timeout, 300),
            )
        except requests.RequestException as e:
            logger.warning("Failed to pre-load model %s: %s", model_name, e)
            return False
        if response.status_code != 200:
            logger.warning(
                "Failed to pre-load model %s: HTTP %s %s",
                model_name, response.status_code, response.text[:200],
            )
            return False
        return True

    def unload_model(self, model_name: str) -> bool:
        """Release a model from memory (``POST /api/generate`` with ``keep_alive: 0``).

        Args:
            model_name: Model to unload.

        Returns:
            True if Ollama reported success.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0, "stream": False},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_client(self) -> Any:  # noqa: ANN401
        """Return an OllamaClient using this server's URL, timeout and default model.

        ``timeout`` (the request timeout, ``--request-timeout``) is used both
        for metadata calls and as the read timeout of vision calls, which must
        cover a cold model load; connecting gets a short 10s timeout.
        """
        from lib.ollama_client import DEFAULT_CONNECT_TIMEOUT, OllamaClient

        return OllamaClient(
            base_url=self.base_url,
            timeout=self.timeout,
            model=self.model,
            generate_timeout=self.timeout,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Manager facade
# ---------------------------------------------------------------------------

class _InstalledHints:
    """Lazily computed "installed models" text for error messages.

    Listing vision models may call ``/api/show`` per model, so it is done at
    most once per :meth:`ModelServerManager.ensure_models` call, and only
    when an error needs it.
    """

    def __init__(self, server: BaseModelServer, capability: Optional[str]):
        self._server = server
        self._capability = capability
        self._names: Optional[list[str]] = None

    def invalidate(self) -> None:
        """Forget the cached list (after a pull changed what is installed)."""
        self._names = None

    def _list(self) -> list[str]:
        if self._names is None:
            try:
                if self._capability == "vision":
                    self._names = list(self._server.list_vision_models())
                else:
                    self._names = list(self._server.list_models())
            except Exception as e:  # hints must never break the check itself
                logger.debug("Could not list installed models: %s", e)
                self._names = []
        return self._names

    @property
    def _kind(self) -> str:
        return f"{self._capability}-capable models" if self._capability else "models"

    def installed_line(self) -> str:
        """``Installed vision-capable models: a, b`` (or ``(none installed)``)."""
        return f"Installed {self._kind}: {', '.join(self._list()) or '(none installed)'}"

    def choose_line(self) -> str:
        """``Or choose an installed vision-capable model with -m: a, b``.

        The text model (``completion``) is chosen with ``--text-model``.
        """
        names = ", ".join(self._list()) or "(none installed)"
        flag = "--text-model" if self._capability == "completion" else "-m"
        return f"Or choose an installed {self._kind[:-1]} with {flag}: {names}"


def _capability_text(capability: str) -> str:
    """Human wording of a required capability (``'vision input'``, ``'text completion'``)."""
    cap = capability.lower()
    if cap == "vision":
        return "vision input"
    if cap == "completion":
        return "text completion"
    return f"the '{cap}' capability"


def _size_line(estimate: PullEstimate, disk: Optional[DiskCheck], disk_error: Optional[str]) -> str:
    """Describe the download size (and free space, when measured) for an error message."""
    if not estimate.known:
        free = f"; {format_gb(disk.free_bytes)} free on {disk.measured_path}" if disk else ""
        return f"Download size unknown ({estimate.error}){free}."
    download = format_gb(estimate.missing_bytes or 0)
    if disk is not None:
        return (
            f"Download size: {download}; free space on {disk.measured_path}: "
            f"{format_gb(disk.free_bytes)} (needs {format_gb(disk.required_bytes)} incl. "
            f"{disk.margin_text} safety margin)."
        )
    if disk_error:
        return f"Download size: {download} ({disk_error})."
    return f"Download size: {download}."


class ModelServerManager:
    """Pipeline-facing facade over the Ollama model server.

    1. :meth:`ensure_running` adopts or starts the server.
    2. :meth:`ensure_models` checks (and, only if allowed, pulls / warms) models.
    3. :meth:`get_client` returns a cached API client.
    4. :meth:`close` releases the client and stops the server only if owned.

    Attributes:
        keep_running: Detach instead of stopping an owned server on close.
    """

    def __init__(self, *, keep_running: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the manager (does not start anything).

        Args:
            keep_running: Leave an owned server running on close.
            **kwargs: :class:`OllamaServer` constructor parameters
                (``base_url``, ``model``, ``timeout``, ``startup_timeout``,
                ``auto_start``, ...).
        """
        self.keep_running = keep_running
        self._client: Any = None
        self._closed = False
        self._server: BaseModelServer | None = OllamaServer(**kwargs)

    # -- accessors ---------------------------------------------------------

    @property
    def server(self) -> BaseModelServer:
        """The concrete server (an OllamaServer)."""
        if self._server is None:
            raise RuntimeError("ModelServerManager not initialized")
        return self._server

    @property
    def state(self) -> ServerState:
        """Current server state (never starts anything)."""
        return self.server.get_state()

    @property
    def last_error(self) -> Optional[str]:
        """The server's last lifecycle error, if any."""
        return self.server.last_error

    def get_state(self) -> ServerState:
        """Return the current server state."""
        return self.state

    def probe(self) -> ServerState:
        """Return the server state without starting anything. Never raises."""
        try:
            return self.server.get_state()
        except Exception as e:
            return ServerState(
                base_url=getattr(self._server, "base_url", ""),
                error_message=str(e),
            )

    def list_models(self) -> list[str]:
        """List models available on the server without downloading."""
        return self.server.list_models()

    # -- lifecycle ---------------------------------------------------------

    def ensure_running(self) -> bool:
        """Adopt or start the server. Idempotent; never raises.

        Returns:
            True if the server is ready; otherwise False with :attr:`last_error` set.
        """
        server = self.server
        try:
            ok = server.start()
        except Exception as e:
            logger.debug("Unexpected error starting the model server", exc_info=True)
            server.last_error = f"Unexpected error starting Ollama: {e}"
            ok = False
        if ok:
            self._closed = False
        else:
            logger.debug("Ollama model server unavailable: %s", server.last_error)
        return ok

    def require_running(self) -> None:
        """Ensure the server is running or raise.

        Raises:
            ModelServerError: If the server cannot be reached or started.
        """
        if not self.ensure_running():
            raise ModelServerError(
                "Ollama model server is not running",
                details=self.last_error or "",
            )

    def ensure_model(
        self,
        model_name: str,
        *,
        allow_pull: bool = False,
        dry_run: bool = False,
        warm: bool = False,
        capability: Optional[str] = None,
        min_free_bytes: Optional[int] = None,
    ) -> bool:
        """Make sure a model can be served, without implicit downloads.

        1. The server must be reachable: in a dry run it is only probed
           (never started); otherwise :meth:`ensure_running` adopts or
           starts it.
        2. An installed model must have ``capability`` (if given).
        3. A missing model is downloaded only with ``allow_pull=True``, and
           only if its size is known and fits on the model store with a
           safety margin (``download + max(min_free_bytes, 10%)``). A pull
           into a remote server's store, whose disk cannot be measured, is
           allowed with a warning. While pulling, the download is aborted if
           free space falls below ``min_free_bytes``.

        Every failure is logged as an ERROR that names the installed
        vision-capable models and the exact manual pull command.

        Args:
            model_name: Model to check.
            allow_pull: Download the model if it is missing. With False
                (the default) nothing is ever downloaded.
            dry_run: Only check: never start the server, pull or load. The
                server must already be healthy. Returns what a real run with
                the same arguments would do.
            warm: Load the model into memory now (ignored in dry-run). A
                failed warm-up is only a warning.
            capability: Required capability (e.g. ``'vision'``); None skips
                the check.
            min_free_bytes: Free space to keep on the model store after a
                pull (None means :data:`DEFAULT_MIN_FREE_BYTES`).

        Returns:
            True if the model is (or, in a dry run with ``allow_pull``, could
            be) available.
        """
        if not self._ensure_reachable(dry_run):
            return False
        return self._ensure_one(
            model_name,
            allow_pull=allow_pull,
            dry_run=dry_run,
            warm=warm,
            capability=capability,
            min_free_bytes=DEFAULT_MIN_FREE_BYTES if min_free_bytes is None else int(min_free_bytes),
            hints=_InstalledHints(self.server, capability),
        )

    def ensure_models(
        self,
        models: Iterable[str | tuple[str, Optional[str]]],
        *,
        allow_pull: bool = False,
        dry_run: bool = False,
        warm: bool = False,
        capability: Optional[str] = None,
        min_free_bytes: Optional[int] = None,
        warm_models: Optional[Iterable[str]] = None,
    ) -> bool:
        """Apply :meth:`ensure_model` to each distinct model (no short-circuit).

        The server is checked once; every model is then checked so all
        problems are reported in one run.

        When pulls are allowed and more than one model is missing, their
        download sizes are added up and checked against the free space on
        the model store once, *before any pull*: if the total does not fit,
        a per-model breakdown is logged and nothing is pulled (this avoids
        downloading 23 GB only to find that the next 65 GB do not fit).

        Args:
            models: Model names, or ``(name, capability)`` pairs. A pair's
                capability overrides ``capability`` (None in a pair falls
                back to it). Names that normalize to the same model are
                checked once; if their capabilities differ, ``'vision'``
                (the stricter one) is kept. An empty list succeeds without
                contacting the server.
            allow_pull: See :meth:`ensure_model`.
            dry_run: See :meth:`ensure_model`.
            warm: See :meth:`ensure_model`.
            capability: Default required capability; see :meth:`ensure_model`.
            min_free_bytes: See :meth:`ensure_model`.
            warm_models: With ``warm``, only these models are loaded (e.g.
                the vision model, so the text model does not sit beside it
                during the vision phase). None warms every model.

        Returns:
            True if every model is available.
        """
        specs: list[tuple[str, Optional[str]]] = []
        index: dict[str, int] = {}
        for item in models:
            if isinstance(item, tuple):
                name, item_cap = item[0], (item[1] if len(item) > 1 else None)
            else:
                name, item_cap = item, None
            cap = item_cap if item_cap is not None else capability
            key = normalize_model_name(name)
            if key not in index:
                index[key] = len(specs)
                specs.append((name, cap))
            elif cap == "vision" or specs[index[key]][1] is None:
                specs[index[key]] = (specs[index[key]][0], cap)
        if not specs:
            return True
        if not self._ensure_reachable(dry_run):
            return False

        warm_set = (
            None if warm_models is None else {normalize_model_name(n) for n in warm_models}
        )
        hints_by_cap: dict[Optional[str], _InstalledHints] = {}

        def hints_for(cap: Optional[str]) -> _InstalledHints:
            if cap not in hints_by_cap:
                hints_by_cap[cap] = _InstalledHints(self.server, cap)
            return hints_by_cap[cap]

        floor = DEFAULT_MIN_FREE_BYTES if min_free_bytes is None else int(min_free_bytes)
        estimates: dict[str, PullEstimate] = {}
        availability: dict[str, tuple[bool, bool]] = {}
        if allow_pull and len(specs) > 1:
            availability = {name: self._availability(name) for name, _ in specs}
            fits, estimates = self._combined_disk_check(availability, floor, dry_run)
            if not fits:
                # Still report capability problems of the installed models,
                # but pull and warm nothing.
                for name, cap in specs:
                    if availability[name][0]:
                        self._check_capability(name, cap, hints_for(cap))
                return False

        results = [
            self._ensure_one(
                name,
                allow_pull=allow_pull,
                dry_run=dry_run,
                warm=warm and (warm_set is None or normalize_model_name(name) in warm_set),
                capability=cap,
                min_free_bytes=floor,
                hints=hints_for(cap),
                estimate=estimates.get(name),
                availability=availability.get(name),
            )
            for name, cap in specs
        ]
        return all(results)

    def _availability(self, name: str) -> tuple[bool, bool]:
        """``(installed, listing_failed)`` for one model, from one ``/api/tags`` read."""
        installed = bool(self.server.is_model_available(name))
        return installed, getattr(self.server, "model_listing_failed", False) is True

    def _combined_disk_check(
        self, availability: dict[str, tuple[bool, bool]], min_free_bytes: int, dry_run: bool
    ) -> tuple[bool, dict[str, PullEstimate]]:
        """Check that all missing models fit on the model store together.

        Only applies when more than one model is missing and the local store
        can be measured; otherwise each model is checked on its own.
        Models whose size is unknown are left to the per-model check (which
        refuses to pull them).

        Returns:
            ``(fits, estimates)``: False if the combined download does not
            fit (an error with the per-model breakdown is logged), and the
            estimates already fetched, by model name, for reuse.
        """
        server = self.server
        if any(listing_failed for _, listing_failed in availability.values()):
            return True, {}
        missing = [name for name, (installed, _) in availability.items() if not installed]
        if len(missing) < 2:
            return True, {}
        estimates = {name: server.estimate_pull(name) for name in missing}
        store = server.model_store_path()
        if store is None:
            return True, estimates
        known = {n: e for n, e in estimates.items() if e.known and not e.not_found}
        if len(known) < 2:
            return True, estimates
        total = sum(e.missing_bytes or 0 for e in known.values())
        try:
            disk = check_disk(store, total, min_free_bytes=min_free_bytes)
        except OSError as e:
            logger.debug("Combined disk check could not measure %s: %s", store, e)
            return True, estimates
        if disk.ok:
            return True, estimates
        breakdown = "\n".join(
            f"    {name}: {format_gb(est.missing_bytes or 0)} (pull manually: "
            f"{server.manual_pull_command(name)})"
            for name, est in known.items()
        )
        logger.error(
            "%sNot enough disk space to pull all missing models: need %s (%s download + %s "
            "safety margin), %s free on %s. Nothing was pulled.\n"
            "%s\n"
            "  Free up at least %s, lower the margin with --min-free-gb, or pull the "
            "models one at a time when there is room.",
            "Dry run: " if dry_run else "",
            format_gb(disk.required_bytes), format_gb(total), disk.margin_text,
            format_gb(disk.free_bytes), disk.measured_path, breakdown,
            format_gb(disk.shortfall_bytes),
        )
        return False, estimates

    def _ensure_reachable(self, dry_run: bool) -> bool:
        """Probe (dry run) or adopt/start (real run) the server."""
        if dry_run:
            if not self.server.health_check():
                logger.error("Dry run: the Ollama server at %s is not running", self.server.base_url)
                return False
            return True
        return self.ensure_running()

    def _ensure_one(
        self,
        name: str,
        *,
        allow_pull: bool,
        dry_run: bool,
        warm: bool,
        capability: Optional[str],
        min_free_bytes: int,
        hints: _InstalledHints,
        estimate: Optional[PullEstimate] = None,
        availability: Optional[tuple[bool, bool]] = None,
    ) -> bool:
        """Check (and, if allowed, pull / warm) one model; see :meth:`ensure_model`.

        ``estimate`` and ``availability`` (``(installed, listing_failed)``)
        reuse what the combined disk check already asked the server.
        """
        server = self.server
        prefix = "Dry run: " if dry_run else ""

        installed, listing_failed = availability if availability is not None else self._availability(name)
        if installed:
            if not self._check_capability(name, capability, hints):
                return False
            if warm and not dry_run:
                self._warm(name)
            logger.info("%sModel '%s' is installed", prefix, name)
            return True
        if listing_failed:
            # The server stopped answering after the reachability check: do
            # not claim the model is missing or suggest downloading it.
            logger.error(
                "%sCould not list the models installed on the Ollama server at %s "
                "(is it still running?); cannot check '%s'.",
                prefix, server.base_url, name,
            )
            return False

        what = f"{capability.capitalize()} model" if capability else "Model"
        cmd = server.manual_pull_command(name)
        if estimate is None:
            estimate = server.estimate_pull(name)
        if estimate.not_found:
            logger.error(
                "Model '%s' was not found in the registry (%s returned 404). "
                "Check the name and tag at https://ollama.com/library.\n"
                "  %s",
                name, estimate.source or "the registry", hints.installed_line(),
            )
            return False

        store = server.model_store_path()
        disk: Optional[DiskCheck] = None
        disk_error: Optional[str] = None
        if store is not None:
            # With an unknown size this only reports free space (the pull is
            # refused below before ``disk.ok`` would matter).
            try:
                disk = check_disk(store, estimate.missing_bytes or 0, min_free_bytes=min_free_bytes)
            except OSError as e:
                disk_error = f"cannot measure free space on {store}: {e}"

        if not allow_pull:
            logger.error(
                "%s '%s' is not installed on the Ollama server at %s.\n"
                "  %s\n"
                "  To download it, re-run with --pull, or run: %s\n"
                "  %s",
                what, name, server.base_url,
                _size_line(estimate, disk, disk_error), cmd, hints.choose_line(),
            )
            return False

        if not estimate.known:
            free_note = f" ({format_gb(disk.free_bytes)} free on {disk.measured_path})" if disk else ""
            logger.error(
                "Could not determine the download size of '%s' (%s); refusing to pull "
                "automatically%s.\n"
                "  If you are sure it fits, pull it manually: %s\n"
                "  %s",
                name, estimate.error, free_note, cmd, hints.choose_line(),
            )
            return False

        download = estimate.missing_bytes or 0
        if store is None:
            logger.warning(
                "Cannot verify free disk space on remote server %s; pulling '%s' needs about %s there.",
                server.base_url, name, format_gb(download),
            )
        elif disk is None:
            logger.error(
                "Refusing to pull '%s' (%s download): %s.\n"
                "  If you are sure it fits, pull it manually: %s\n"
                "  %s",
                name, format_gb(download), disk_error, cmd, hints.choose_line(),
            )
            return False
        elif not disk.ok:
            logger.error(
                "Not enough disk space to pull '%s': need %s (%s download + %s safety margin), "
                "%s free on %s.\n"
                "  Free up at least %s, lower the margin with --min-free-gb, or pull it "
                "manually when there is room: %s\n"
                "  %s",
                name, format_gb(disk.required_bytes), format_gb(download),
                disk.margin_text, format_gb(disk.free_bytes), disk.measured_path,
                format_gb(disk.shortfall_bytes), cmd, hints.choose_line(),
            )
            return False

        if dry_run:
            where = (
                f"{format_gb(disk.free_bytes)} free on {disk.measured_path}"
                if disk is not None else f"free space on {server.base_url} not verifiable"
            )
            logger.info("Dry run: would pull '%s' (%s download; %s)", name, format_gb(download), where)
            return True

        logger.info("Pulling '%s' (%s download, --pull)...", name, format_gb(download))
        guard = make_disk_guard(store, min_free_bytes) if store is not None else None
        if not server.pull_model(name, should_abort=guard):
            logger.error("Failed to pull model '%s'. You can retry, or run: %s", name, cmd)
            return False
        if not server.is_model_available(name):
            logger.error(
                "Pull of '%s' reported success but the model is not listed by the server", name
            )
            return False
        logger.info("Model '%s' pulled successfully", name)
        hints.invalidate()
        if not self._check_capability(name, capability, hints):
            return False
        if warm:
            self._warm(name)
        return True

    def _check_capability(
        self, name: str, capability: Optional[str], hints: _InstalledHints
    ) -> bool:
        """Return False (with an error) if an installed model lacks ``capability``."""
        if not capability:
            return True
        caps = self.server.get_capabilities(name)
        if caps is None:
            logger.warning(
                "Could not determine the capabilities of '%s'; assuming it supports %s",
                name, _capability_text(capability),
            )
            return True
        if capability.lower() in caps:
            return True
        logger.error(
            "Model '%s' is installed but does not support %s (capabilities: %s).\n"
            "  %s",
            name, _capability_text(capability), ", ".join(sorted(caps)) or "none reported",
            hints.choose_line(),
        )
        return False

    def unload_model(self, name: str) -> bool:
        """Release a model from the server's memory now.

        Sends ``keep_alive: 0`` for the model, which works on adopted
        (external) servers as well as owned ones. Used between pipeline
        phases so the vision and text models are not resident together.

        Args:
            name: Model to unload.

        Returns:
            True if the server reported success. A failure is only logged.
        """
        try:
            ok = bool(self.server.unload_model(name))
        except Exception as e:  # unloading is best-effort
            logger.warning("Could not unload model '%s': %s", name, e)
            return False
        if ok:
            logger.info("Unloaded model '%s'", name)
        else:
            logger.warning("Could not unload model '%s'; Ollama may keep it in memory", name)
        return ok

    def installed_model_size(self, name: str) -> Optional[int]:
        """Size in bytes of an installed model, or None if unknown."""
        try:
            return self.server.installed_model_size(name)
        except Exception as e:  # advisory only
            logger.debug("Could not read the size of '%s': %s", name, e)
            return None

    def _warm(self, name: str) -> None:
        """Load a model now; a failure is only a warning."""
        if not self.server.load_model(name):
            logger.warning(
                "Failed to pre-load model '%s'; it will load on the first request", name
            )

    def get_client(self) -> Any:  # noqa: ANN401
        """Return the (cached) API client, starting the server if needed.

        Raises:
            ModelServerError: If the server is not running and cannot be started.
        """
        if self._client is None:
            self.require_running()
            self._client = self.server.get_client()
        return self._client

    def close(self) -> None:
        """Release the client and stop the server if this program owns it.

        With ``keep_running`` an owned server is detached instead. External
        servers are never stopped. Idempotent; never raises.
        """
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing model client: %s", e)
        if self._server is not None:
            try:
                if self.keep_running:
                    self._server.detach()
                else:
                    self._server.stop()
            except Exception as e:
                logger.warning("Error during model server shutdown: %s", e)

    def __enter__(self) -> ModelServerManager:
        """Context manager entry: require a running server."""
        self.require_running()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Context manager exit: :meth:`close`."""
        self.close()
