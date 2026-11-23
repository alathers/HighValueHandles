"""Supervision primitives for model-server child processes.

Small, dependency-free helpers used by :mod:`lib.model_server` to run a
server it owns (``ollama serve``). Only POSIX facilities are used
(``setsid`` via ``start_new_session`` and ``killpg``), so this works on macOS
and Linux:

- :class:`ManagedProcess` -- a child we own. Its output goes to a log file
  (never an undrained PIPE, which deadlocks chatty servers), it runs in its own
  session/process group, and :meth:`ManagedProcess.terminate` signals the whole
  group so grandchildren are not orphaned.
- :func:`wait_until_ready` -- poll a readiness probe with exponential backoff,
  bailing out early if the child exits.
- :func:`port_in_use` -- is something already listening on host:port?
- :func:`default_log_dir` -- where server logs are written.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional

logger = logging.getLogger(__name__)

#: Seconds stragglers left in the group after the leader exits get between
#: SIGTERM and SIGKILL.
STRAGGLER_GRACE_SECONDS = 2.0


@dataclass
class ProbeResult:
    """Outcome of one readiness/identity probe.

    Attributes:
        ready: Whether the endpoint answered as the expected server.
        detail: Short human-readable reason (used in error messages).
        payload: Parsed JSON body of the probe response, if any.
    """

    ready: bool
    detail: str = ""
    payload: Optional[dict[str, Any]] = None


def default_log_dir() -> Path:
    """Return the directory for spawned-server logs.

    Uses ``$BOURBON_LOG_DIR`` when set, otherwise
    ``$XDG_CACHE_HOME/bourbon-processor/logs`` (``$XDG_CACHE_HOME`` defaults to
    ``~/.cache``, as on Linux; the same path is used on macOS).

    Returns:
        Log directory path (not created here).
    """
    override = os.environ.get("BOURBON_LOG_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "bourbon-processor" / "logs"


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check whether something accepts TCP connections on ``host:port``.

    Args:
        host: Host name or IP address.
        port: TCP port.
        timeout: Connect timeout in seconds.

    Returns:
        True if a connection could be opened, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ManagedProcess:
    """A server child process owned by this program.

    Output (stdout and stderr) is appended to ``log_path``; the child gets its
    own session so it does not receive the terminal's Ctrl-C and so the whole
    process group can be signalled on shutdown.

    Attributes:
        argv: Command line.
        log_path: File receiving the child's combined output.
        env: Environment for the child (None = inherit).
        name: Display name used in log messages.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        log_path: Path,
        env: Optional[dict[str, str]] = None,
        name: str = "server",
    ):
        """Initialize (does not spawn).

        Args:
            argv: Command line to execute.
            log_path: File receiving the child's combined output.
            env: Environment for the child (None = inherit).
            name: Display name used in log messages.
        """
        self.argv = list(argv)
        self.log_path = Path(log_path)
        self.env = env
        self.name = name
        self._popen: Optional[subprocess.Popen] = None
        self._log_fh: Optional[IO[bytes]] = None
        # Whether the process group still had members right after the leader
        # was reaped (None until then). Checked in the same instant as the
        # reap, while the group id cannot have been reused.
        self._group_alive_after_reap: Optional[bool] = None

    # -- lifecycle ---------------------------------------------------------

    def spawn(self) -> None:
        """Start the child process.

        Raises:
            FileNotFoundError: If the executable does not exist.
            OSError: If the process (or its log file) cannot be created.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "ab", buffering=0)  # noqa: SIM115 - kept open for the child
        try:
            self._popen = subprocess.Popen(
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException:
            self._close_log()
            raise

    def terminate(self, grace_seconds: float = 10.0) -> bool:
        """Stop the child and its process group.

        Sends SIGTERM to the group, waits up to ``grace_seconds`` for the
        leader, then escalates to SIGKILL. Members left in the group after
        the leader exits get SIGTERM, then SIGKILL after
        :data:`STRAGGLER_GRACE_SECONDS`. Idempotent.

        The group is signalled only while it is known to be ours: while the
        leader is unreaped (its pid pins the group id), or when the group
        still had members at the moment the leader was reaped (a live group
        id is never handed to a new process). A group whose leader was reaped
        with no members left is never signalled, so a recycled pid cannot
        receive our signal.

        Args:
            grace_seconds: Seconds to wait after SIGTERM before SIGKILL.

        Returns:
            False only if the process could not be reaped even after SIGKILL.
        """
        proc = self._popen
        if proc is None:
            return True
        ok = True
        try:
            if self._poll() is None:
                self._signal_group(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "%s (pid %d) did not exit within %.0fs; sending SIGKILL",
                        self.name, proc.pid, grace_seconds,
                    )
                    self._signal_group(proc.pid, signal.SIGKILL)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        ok = False
                if ok:
                    self._note_reaped(proc.pid)
            if ok and self._group_alive_after_reap:
                self._stop_stragglers(proc.pid)
        finally:
            self._close_log()
            self._popen = None
        return ok

    def _poll(self) -> Optional[int]:
        """``Popen.poll()`` that records the group state when it reaps the leader."""
        proc = self._popen
        if proc is None:
            return None
        already_reaped = proc.returncode is not None
        code = proc.poll()
        if code is not None and not already_reaped:
            self._note_reaped(proc.pid)
        return code

    def _note_reaped(self, pgid: int) -> None:
        """Record, right after the leader was reaped, whether its group lives on."""
        if self._group_alive_after_reap is None:
            self._group_alive_after_reap = self._group_exists(pgid)

    def _stop_stragglers(self, pgid: int) -> None:
        """SIGTERM what is left of the group, then SIGKILL after a short grace."""
        if not self._group_exists(pgid):
            return
        logger.debug("%s: stopping processes left in group %d", self.name, pgid)
        self._signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + STRAGGLER_GRACE_SECONDS
        while time.monotonic() < deadline:
            if not self._group_exists(pgid):
                return
            time.sleep(0.05)
        self._signal_group(pgid, signal.SIGKILL)

    @staticmethod
    def _group_exists(pgid: int) -> bool:
        """Whether process group ``pgid`` has members (signal 0 probe)."""
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, but is not ours to signal; _signal_group ignores it
        except OSError:
            return False
        return True

    def detach(self) -> None:
        """Forget the child without signalling it (it keeps running)."""
        self._close_log()
        self._popen = None

    # -- introspection -----------------------------------------------------

    @property
    def popen(self) -> Optional[subprocess.Popen]:
        """The underlying Popen handle, or None if not running/owned."""
        return self._popen

    @property
    def pid(self) -> int:
        """Process id of the child (0 if none)."""
        return self._popen.pid if self._popen is not None else 0

    @property
    def returncode(self) -> Optional[int]:
        """Exit code if the child has exited, else None."""
        return self._poll()

    def is_alive(self) -> bool:
        """Return True while the child process is running."""
        return self._popen is not None and self._poll() is None

    def tail_log(self, lines: int = 20) -> str:
        """Return the last ``lines`` lines of the child's log ('' if unreadable)."""
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(deque(fh, maxlen=lines))
        except OSError:
            return ""

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        """Send ``sig`` to the process group led by ``pid``, ignoring races."""
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None


def wait_until_ready(
    probe: Callable[[], ProbeResult],
    *,
    timeout: float,
    is_alive: Optional[Callable[[], bool]] = None,
    initial_delay: float = 0.25,
    max_delay: float = 2.0,
    factor: float = 1.6,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProbeResult:
    """Poll ``probe`` until it reports ready, the child dies, or time runs out.

    The probe runs first (no initial sleep); delays then grow geometrically
    from ``initial_delay`` by ``factor`` up to ``max_delay``.

    Args:
        probe: Callable returning a :class:`ProbeResult`.
        timeout: Overall deadline in seconds.
        is_alive: Optional liveness check for the child; when it returns False
            the wait stops immediately.
        initial_delay: First sleep between probes.
        max_delay: Upper bound for a single sleep.
        factor: Backoff multiplier.
        clock: Monotonic clock (injectable for tests).
        sleep: Sleep function (injectable for tests).

    Returns:
        The ready ProbeResult, or a not-ready one whose ``detail`` explains why.
    """
    deadline = clock() + timeout
    delay = initial_delay
    while True:
        last = probe()
        if last.ready:
            return last
        if is_alive is not None and not is_alive():
            return ProbeResult(False, "process exited before becoming ready")
        remaining = deadline - clock()
        if remaining <= 0:
            return ProbeResult(False, f"timed out after {timeout:.0f}s ({last.detail})")
        sleep(min(delay, remaining))
        delay = min(delay * factor, max_delay)
