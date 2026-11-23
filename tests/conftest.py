"""Pytest configuration for bourbon-image-processor tests.

Besides putting ``src/`` on ``sys.path``, this installs a hermetic guard for
every test: HTTP calls that could download models or run inference, and spawns
of the ``ollama`` executable, are blocked and fail the test. Tests
must patch those calls instead. The guard covers this process only; CLI
subprocess tests must stay on safe paths (``--dry-run`` or early exits) and
point ``--ollama-url`` at a closed port (see
``tests/contract/test_cli_contracts.py``) so they never reach a real server.

Opt out with ``@pytest.mark.allow_model_server`` (used only by the live tests,
which are themselves skipped unless ``BOURBON_LIVE_TESTS=1``).
"""

import subprocess
import sys
from pathlib import Path

import pytest
import requests

# Add src directory to Python path for imports like `from image_processor import ...`
src_path = Path(__file__).parent.parent / "src"
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


class HermeticGuardError(requests.exceptions.ConnectionError):
    """Raised for a blocked call.

    A ConnectionError subclass so the code under test sees an ordinary network
    failure; the guard still fails the test at teardown even if the code
    swallows it.
    """


def pytest_configure(config: pytest.Config) -> None:
    """Register the guard's opt-out marker."""
    config.addinivalue_line(
        "markers",
        "allow_model_server: disable the hermetic model-server guard (live tests only)",
    )


def _argv0(args: object) -> str:
    if isinstance(args, (str, bytes, Path)):
        first = str(args).split()[0] if str(args).strip() else ""
    else:
        try:
            first = str(next(iter(args)))  # type: ignore[call-overload]
        except (StopIteration, TypeError):
            first = ""
    return Path(first).name


@pytest.fixture(autouse=True)
def _hermetic_model_server_guard(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Block model pulls, inference and model-server spawns for every test."""
    if request.node.get_closest_marker("allow_model_server"):
        yield
        return

    violations: list[str] = []

    original_request = requests.sessions.Session.request

    def guarded_request(self, method, url, *args, **kwargs):
        if any(part in str(url) for part in BLOCKED_URL_PARTS):
            message = f"HTTP {method} {url}"
            violations.append(message)
            raise HermeticGuardError(f"hermetic test guard blocked {message}; patch this call")
        return original_request(self, method, url, *args, **kwargs)

    original_popen_init = subprocess.Popen.__init__

    def guarded_popen_init(self, args, *pargs, **kwargs):
        name = _argv0(args)
        if name in BLOCKED_EXECUTABLES:
            message = f"spawn of {args!r}"
            violations.append(message)
            raise PermissionError(f"hermetic test guard blocked {message}; patch the spawn")
        original_popen_init(self, args, *pargs, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", guarded_request)
    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_popen_init)

    yield

    if violations:
        pytest.fail(
            "Hermetic test guard: this test tried to reach a model server or registry: "
            + "; ".join(violations),
            pytrace=False,
        )
