"""Opt-in live tests for lib.model_server against the developer's real servers.

Skipped unless ``BOURBON_LIVE_TESTS=1``. They never pull models and never stop
a server they did not start:

- (a) Ollama on localhost:11434 is adopted as EXTERNAL and still answers after
  ``close()``.
- (b) Only with ``BOURBON_LIVE_SPAWN=1``: an owned ``ollama serve`` on
  127.0.0.1:11500 becomes ready and ``close()`` frees the port.

Run::

    BOURBON_LIVE_TESTS=1 uv run --extra dev pytest tests/integration/test_model_server_live.py
"""

import os
import shutil

import pytest
import requests

from lib.model_server import ModelServerManager, Ownership
from lib.process_supervisor import port_in_use

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("BOURBON_LIVE_TESTS") != "1", reason="set BOURBON_LIVE_TESTS=1 to run"
    ),
    pytest.mark.allow_model_server,
    pytest.mark.live,
]


def _answers(url: str) -> bool:
    try:
        return requests.get(url, timeout=2).status_code == 200
    except requests.RequestException:
        return False


def test_ollama_adopted_as_external_and_left_running():
    if not _answers("http://localhost:11434/api/version"):
        pytest.skip("Ollama is not running on localhost:11434")
    manager = ModelServerManager()
    assert manager.ensure_running() is True
    assert manager.server.ownership is Ownership.EXTERNAL
    manager.close()
    assert _answers("http://localhost:11434/api/version")


@pytest.mark.skipif(
    os.environ.get("BOURBON_LIVE_SPAWN") != "1", reason="set BOURBON_LIVE_SPAWN=1 to spawn ollama"
)
def test_owned_ollama_serve_starts_and_stops(tmp_path, monkeypatch):
    if shutil.which("ollama") is None:
        pytest.skip("ollama is not installed")
    if port_in_use("127.0.0.1", 11500):
        pytest.skip("port 11500 is busy")
    monkeypatch.setenv("BOURBON_LOG_DIR", str(tmp_path))
    manager = ModelServerManager(base_url="http://127.0.0.1:11500")
    try:
        assert manager.ensure_running() is True, manager.last_error
        assert manager.server.ownership is Ownership.OWNED
        assert manager.server.pid > 0
    finally:
        manager.close()
    assert not port_in_use("127.0.0.1", 11500)
