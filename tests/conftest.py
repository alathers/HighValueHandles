"""Pytest configuration for bourbon-image-processor tests.

Besides putting ``src/`` on ``sys.path``, this makes every test hermetic:

- **Isolated environment** (``_isolate_environment``, autouse): each test runs
  with its cwd set to ``tmp_path``, ``XDG_CONFIG_HOME``/``XDG_CACHE_HOME``
  pointing under ``tmp_path``, every ``BOURBON_*`` variable removed (except
  ``BOURBON_LIVE_*`` and ``BOURBON_LOG_DIR``), and ``BRAVE_API_KEY`` /
  ``OLLAMA_API_KEY`` removed (unless the test is marked ``allow_web``). So no
  user config, cache, API key or stray env var can leak into a test, and
  in-process and subprocess CLI runs resolve no web provider.
- **Model-server guard** (``_hermetic_model_server_guard``, autouse): HTTP
  calls that could download models or run inference, and spawns of the
  ``ollama`` executable, are blocked and fail the test. Opt out with
  ``@pytest.mark.allow_model_server`` (live tests only, which are themselves
  skipped unless ``BOURBON_LIVE_TESTS=1``).
- **Web guard** (same fixture): any HTTP request to a host other than
  loopback, and any ``socket.getaddrinfo`` / ``socket.connect`` to a
  non-loopback target, is blocked and fails the test. AF_UNIX sockets and
  loopback are allowed. Opt out with ``@pytest.mark.allow_web`` (live web
  tests only). Tests must inject fakes for HTTP, search and LLM calls.
- **Repo-root guard** (session hooks): if the repo root gains a new entry
  during the run, the session fails. Pre-existing debris does not count.

The in-process guards cover this process only; CLI subprocess tests must stay
on safe paths (``--dry-run`` or early exits) and point ``--ollama-url`` at a
closed loopback port with ``--no-server-autostart`` so they never reach a real
server (see ``tests/contract/test_cli_contracts.py``).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add src directory to Python path for imports like `from image_processor import ...`
src_path = REPO_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

#: URL fragments that download/delete models or run inference.
BLOCKED_URL_PARTS = (
    "/api/pull",
    "/api/push",
    "/api/create",
    "/api/delete",
    "/api/copy",
    "/api/generate",
    "/api/chat",
    "/api/embed",
    "/v1/chat/completions",
    "/v1/completions",
    "registry.ollama.ai",
    "huggingface.co",
    "hf.co/",
)

#: Executables tests must never launch (model servers / downloaders).
BLOCKED_EXECUTABLES = frozenset({"ollama"})

#: Host names (besides loopback IP literals) that count as local.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Env vars kept even though they start with ``BOURBON_``.
KEPT_BOURBON_PREFIXES = ("BOURBON_LIVE_", "BOURBON_LOG_DIR")

#: Web API keys scrubbed unless the test is marked ``allow_web``.
WEB_API_KEY_VARS = ("BRAVE_API_KEY", "OLLAMA_API_KEY")

#: Tool caches that may appear in the repo root during a run without counting
#: as test debris.
IGNORED_ROOT_ENTRIES = frozenset(
    {".pytest_cache", ".ruff_cache", ".mypy_cache", ".coverage", "__pycache__", ".venv"}
)


class HermeticGuardError(requests.exceptions.ConnectionError):
    """Raised for a blocked call.

    A ConnectionError (and so OSError) subclass so the code under test sees an
    ordinary network failure; the guard still fails the test at teardown even
    if the code swallows it.
    """


def pytest_configure(config: pytest.Config) -> None:
    """Register the guards' opt-out markers."""
    config.addinivalue_line(
        "markers",
        "allow_model_server: disable the hermetic model-server guard (live tests only)",
    )
    config.addinivalue_line(
        "markers",
        "allow_web: allow non-loopback network access and keep web API keys (live web tests only)",
    )
    config.addinivalue_line(
        "markers",
        "live: opt-in test against real servers or the web (skipped unless BOURBON_LIVE_TESTS=1)",
    )


# ---------------------------------------------------------------------------
# Repo-root guard
# ---------------------------------------------------------------------------

_ROOT_SNAPSHOT_KEY = pytest.StashKey[set]()


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config.stash[_ROOT_SNAPSHOT_KEY] = set(os.listdir(REPO_ROOT))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    before = session.config.stash.get(_ROOT_SNAPSHOT_KEY, None)
    if before is None:
        return
    new_entries = sorted(set(os.listdir(REPO_ROOT)) - before - IGNORED_ROOT_ENTRIES)
    if new_entries:
        print(
            "\nRepo-root guard: the test run created new entries in "
            f"{REPO_ROOT}: {', '.join(new_entries)}. Tests must write under tmp_path.",
            file=sys.stderr,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """Run every test in ``tmp_path`` with a scrubbed, sandboxed environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    for name in list(os.environ):
        if name.startswith("BOURBON_") and not name.startswith(KEPT_BOURBON_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    if not request.node.get_closest_marker("allow_web"):
        for name in WEB_API_KEY_VARS:
            monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------------------
# Model-server and web guard
# ---------------------------------------------------------------------------


def _argv0(args: object) -> str:
    if isinstance(args, (str, bytes, Path)):
        first = str(args).split()[0] if str(args).strip() else ""
    else:
        try:
            first = str(next(iter(args)))  # type: ignore[call-overload]
        except (StopIteration, TypeError):
            first = ""
    return Path(first).name


def _is_loopback_host(host: object) -> bool:
    """True for loopback names/addresses (and an empty/None host, i.e. local)."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip().lower()
    if host in ("", *LOOPBACK_HOSTS):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]").split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _hermetic_model_server_guard(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Block model pulls, inference, model-server spawns and web access.

    Yields the list of recorded violations (see ``hermetic_violations``).
    """
    allow_model_server = request.node.get_closest_marker("allow_model_server") is not None
    allow_web = request.node.get_closest_marker("allow_web") is not None

    violations: list[str] = []

    def _block(message: str, hint: str) -> HermeticGuardError:
        violations.append(message)
        return HermeticGuardError(f"hermetic test guard blocked {message}; {hint}")

    if not allow_model_server:
        original_popen_init = subprocess.Popen.__init__

        def guarded_popen_init(self, args, *pargs, **kwargs):
            name = _argv0(args)
            if name in BLOCKED_EXECUTABLES:
                message = f"spawn of {args!r}"
                violations.append(message)
                raise PermissionError(f"hermetic test guard blocked {message}; patch the spawn")
            original_popen_init(self, args, *pargs, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "__init__", guarded_popen_init)

    original_request = requests.sessions.Session.request

    def guarded_request(self, method, url, *args, **kwargs):
        if not allow_model_server and any(part in str(url) for part in BLOCKED_URL_PARTS):
            raise _block(f"HTTP {method} {url}", "patch this call")
        if not allow_web:
            try:
                host = urlsplit(str(url)).hostname
            except ValueError:
                host = None
            if not host or not _is_loopback_host(host):
                raise _block(f"web access {url}", "inject a fake session/provider")
        return original_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", guarded_request)

    if not allow_web:
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex

        def guarded_getaddrinfo(host, *args, **kwargs):
            if not _is_loopback_host(host):
                raise _block(f"web access getaddrinfo({host!r})", "inject a fake resolver")
            return original_getaddrinfo(host, *args, **kwargs)

        def _check_address(sock: socket.socket, address: object) -> None:
            if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
                return
            host = address[0] if isinstance(address, tuple) and address else address
            if not _is_loopback_host(host):
                raise _block(f"web access connect({address!r})", "inject a fake session")

        def guarded_connect(self, address):
            _check_address(self, address)
            return original_connect(self, address)

        def guarded_connect_ex(self, address):
            _check_address(self, address)
            return original_connect_ex(self, address)

        monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
        monkeypatch.setattr(socket.socket, "connect", guarded_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

    yield violations

    if violations:
        pytest.fail(
            "Hermetic test guard: this test tried to reach a model server, registry or "
            "the web: " + "; ".join(violations),
            pytrace=False,
        )


@pytest.fixture
def hermetic_violations(_hermetic_model_server_guard: list[str]) -> list[str]:
    """The guard's violation list, so a test that expects a block can clear it."""
    return _hermetic_model_server_guard
