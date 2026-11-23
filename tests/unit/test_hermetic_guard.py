"""Checks on the hermetic test guard in tests/conftest.py."""

import os
import socket
from pathlib import Path

import pytest
import requests

from tests.conftest import HermeticGuardError


@pytest.fixture(scope="module")
def _bourbon_foo_set_before_tests():
    """Set BOURBON_FOO before the function-scoped env scrub runs."""
    os.environ["BOURBON_FOO"] = "leaked"
    os.environ["BOURBON_LIVE_KEEP"] = "1"
    yield
    os.environ.pop("BOURBON_FOO", None)
    os.environ.pop("BOURBON_LIVE_KEEP", None)


def test_web_request_is_blocked(hermetic_violations):
    with pytest.raises(HermeticGuardError):
        requests.get("https://example.com", timeout=1)
    assert any("web access https://example.com" in v for v in hermetic_violations)
    hermetic_violations.clear()


def test_socket_resolution_of_web_host_is_blocked(hermetic_violations):
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 443)
    assert hermetic_violations
    hermetic_violations.clear()


def test_loopback_resolution_is_allowed(hermetic_violations):
    assert socket.getaddrinfo("127.0.0.1", 9)
    assert hermetic_violations == []


def test_env_scrub_removes_bourbon_vars(_bourbon_foo_set_before_tests):
    assert "BOURBON_FOO" not in os.environ
    assert os.environ.get("BOURBON_LIVE_KEEP") == "1"
    assert "BRAVE_API_KEY" not in os.environ
    assert "OLLAMA_API_KEY" not in os.environ


def test_cwd_and_xdg_dirs_are_under_tmp_path(tmp_path):
    assert Path.cwd() == tmp_path
    assert Path(os.environ["XDG_CONFIG_HOME"]) == tmp_path / "xdg_config"
    assert Path(os.environ["XDG_CACHE_HOME"]) == tmp_path / "xdg_cache"


def test_socket_connect_to_non_loopback_address_is_blocked(hermetic_violations):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(OSError):
        sock.connect(("192.0.2.1", 80))  # TEST-NET-1: blocked before any packet is sent
    assert any("connect" in v for v in hermetic_violations)
    hermetic_violations.clear()
