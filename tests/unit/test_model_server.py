"""Unit tests for lib.model_server: OllamaServer and ModelServerManager.

Tests cover:
  - ModelServerManager always manages an OllamaServer; ServerState defaults
  - The shared start() template: adopt external, refuse to spawn (remote,
    disabled, missing binary, port taken), spawn + readiness, failure cleanup
  - Ownership: external servers are never stopped; owned ones are terminated
  - OllamaServer: real endpoints only (/api/version, /api/ps, /api/generate)
  - OllamaServer model queries: exact-name matching, capabilities
    (/api/tags, /api/show fallback), vision-model listing, model store path,
    pull estimates, streaming /api/pull with progress and the disk guard
  - ModelServerManager: ensure_* never pull unless allowed, refuse unknown
    sizes and insufficient disk, reject non-vision models, report installed
    vision models and the manual pull command; dry runs never start, pull or
    load; client caching, idempotent close, --keep-server detach

Every test is hermetic: HTTP and process spawning are blocked by the
``_hermetic`` fixture unless a test patches them explicitly.
"""

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from lib.exceptions import ModelServerError
from lib.model_availability import GB, GiB, PullEstimate
from lib.model_server import (
    DEFAULT_REQUEST_TIMEOUT,
    BaseModelServer,
    ModelServerManager,
    OllamaServer,
    Ownership,
    ServerStartError,
    ServerState,
)
from lib.ollama_client import OllamaClient
from lib.process_supervisor import ProbeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hermetic guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """Block real HTTP, process spawns, signals and atexit hooks for every test.

    A real Ollama answers on :11434 on developer machines, so an unpatched
    call would otherwise hit it. Spawn attempts raise and are also
    recorded, so a test cannot pass by swallowing them.
    """
    spawned: list = []

    def no_http(*_args, **_kwargs):
        raise requests.ConnectionError("hermetic test: no network")

    def no_spawn(*args, **_kwargs):
        spawned.append(args)
        raise AssertionError("tests must not spawn processes")

    monkeypatch.setattr("lib.model_server.requests.get", no_http)
    monkeypatch.setattr("lib.model_server.requests.post", no_http)
    # Registry manifest fetches (pull estimates).
    monkeypatch.setattr("lib.model_availability.requests.get", no_http)
    monkeypatch.setattr("lib.model_server.subprocess.run", no_spawn)
    monkeypatch.setattr("lib.model_server.subprocess.Popen", no_spawn)
    monkeypatch.setattr("lib.process_supervisor.subprocess.Popen", no_spawn)
    monkeypatch.setattr("lib.process_supervisor.os.killpg", MagicMock(name="killpg"))
    monkeypatch.setattr("lib.model_server.atexit", MagicMock(name="atexit"))
    monkeypatch.setattr("lib.model_server.port_in_use", lambda *_a, **_k: False)
    monkeypatch.setenv("BOURBON_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    yield
    assert not spawned, f"test tried to spawn: {spawned}"


def _resp(status=200, payload=None, text=""):
    """A fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}
    resp.text = text or json.dumps(payload or {})
    return resp


def _stream_resp(lines, status=200):
    """A fake streaming /api/pull response yielding NDJSON ``lines`` (dicts)."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = "boom"
    resp.iter_lines.return_value = [
        json.dumps(line).encode() if isinstance(line, dict) else line for line in lines
    ]
    return resp


#: /api/tags as reported by Ollama 0.40 on the development machine (trimmed).
TAGS = {
    "models": [
        {"name": "qwen3.6:35b-a3b", "model": "qwen3.6:35b-a3b",
         "capabilities": ["completion", "vision", "tools", "thinking"]},
        {"name": "gpt-oss:120b", "model": "gpt-oss:120b",
         "capabilities": ["completion", "tools", "thinking"]},
        {"name": "qwen3-coder:latest", "model": "qwen3-coder:latest",
         "capabilities": ["completion", "tools"]},
        {"name": "gemma4:26b", "model": "gemma4:26b", "capabilities": ["completion", "vision"]},
        {"name": "Drews54/llama3.2-vision-abliterated:latest",
         "model": "Drews54/llama3.2-vision-abliterated:latest"},
    ]
}


def _fake_proc(alive=True, pid=4321):
    proc = MagicMock(name="ManagedProcess")
    proc.pid = pid
    proc.is_alive.return_value = alive
    proc.returncode = None if alive else 1
    proc.tail_log.return_value = "fatal: model not found\n"
    proc.log_path = "/tmp/logs/x.log"
    proc.terminate.return_value = True
    return proc


# ---------------------------------------------------------------------------
# Manager server / ServerState
# ---------------------------------------------------------------------------


class TestManagerServer:
    """The manager always manages an Ollama server."""

    def test_manager_server_is_ollama(self):
        assert isinstance(ModelServerManager().server, OllamaServer)

    def test_manager_passes_kwargs_to_ollama_server(self):
        manager = ModelServerManager(base_url="http://127.0.0.1:11500", model="m", timeout=42)
        assert manager.server.base_url == "http://127.0.0.1:11500"
        assert manager.server.model == "m"
        assert manager.server.timeout == 42


class TestServerState:
    """Test ServerState dataclass defaults."""

    def test_defaults(self):
        state = ServerState()
        assert state.is_running is False
        assert state.base_url == ""
        assert state.pid == 0
        assert state.models_loaded == []
        assert state.memory_usage_mb == 0
        assert state.cold_start_seconds == 0.0
        assert state.uptime_seconds == 0.0
        assert state.error_message is None
        assert state.owned is False
        assert state.version == ""
        assert state.log_path == ""


# ---------------------------------------------------------------------------
# OllamaServer
# ---------------------------------------------------------------------------


class TestOllamaServer:
    """OllamaServer lifecycle with mocked HTTP."""

    def _make_server(self, **kwargs) -> OllamaServer:
        return OllamaServer(**kwargs)

    def test_default_base_url(self):
        server = self._make_server()
        assert server.base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        server = self._make_server(base_url="http://remote:11434")
        assert server.base_url == "http://remote:11434"
        assert server.host == "remote"
        assert server.port == 11434

    def test_invalid_port_raises(self):
        with pytest.raises(ValueError):
            self._make_server(base_url="http://localhost:99999")

    def test_health_check_success(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get",
                   return_value=_resp(200, {"version": "0.40.16"})) as get:
            assert server.health_check() is True
        assert get.call_args.args[0] == "http://localhost:11434/api/version"
        assert server.version == "0.40.16"

    def test_health_check_rejects_non_ollama(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"ok": True})):
            assert server.health_check() is False

    def test_health_check_failure(self):
        server = self._make_server()
        assert server.health_check() is False

    def test_start_already_running(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"version": "1"})), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is True
        mp.assert_not_called()
        assert server.ownership is Ownership.EXTERNAL

    def test_start_not_auto_start(self):
        server = self._make_server(auto_start=False)
        with patch.object(server, "health_check", return_value=False):
            assert server.start() is False
        assert "--no-server-autostart" in server.last_error

    def test_start_remote_url_does_not_spawn(self):
        server = self._make_server(base_url="http://remote:11434")
        with patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is False
        mp.assert_not_called()
        assert "not this machine" in server.last_error

    def test_start_without_ollama_binary(self):
        server = self._make_server()
        with patch("lib.model_server.shutil.which", return_value=None), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is False
        mp.assert_not_called()
        assert "ollama.com/download" in server.last_error

    def test_start_refuses_when_store_is_linux_service(self, monkeypatch, tmp_path):
        # Linux, service store present, no user store, OLLAMA_MODELS unset: a
        # user `ollama serve` would read/write ~/.ollama/models instead.
        from lib import model_availability as ma

        monkeypatch.delenv("OLLAMA_MODELS", raising=False)
        monkeypatch.setattr(ma.platform, "system", lambda: "Linux")
        monkeypatch.setattr(ma.Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(ma, "_path_exists_or_hidden", lambda p: p == ma.LINUX_SERVICE_MODELS_DIR)
        server = self._make_server()
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch("lib.model_server.port_in_use", return_value=False), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is False
        mp.assert_not_called()
        assert "systemctl start ollama" in server.last_error

    def test_start_allowed_with_linux_service_when_ollama_models_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "models"))
        server = self._make_server()
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch("lib.model_server.ollama_models_dir", return_value=Path("/usr/share/ollama/.ollama/models")):
            assert server._can_spawn() == (True, "")

    def test_start_port_in_use_by_other(self):
        server = self._make_server()
        with patch("lib.model_server.shutil.which", return_value="/usr/local/bin/ollama"), \
                patch("lib.model_server.port_in_use", return_value=True), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is False
        mp.assert_not_called()
        assert "Port 11434" in server.last_error
        assert "in use" in server.last_error

    def test_start_spawns_and_owns_local_ollama(self):
        server = self._make_server()
        proc = _fake_proc()
        probes = [ProbeResult(False, "connection refused"), ProbeResult(True, "Ollama 1", {"version": "1"})]
        with patch("lib.model_server.shutil.which", return_value="/usr/local/bin/ollama"), \
                patch.object(OllamaServer, "_identity_probe", side_effect=probes), \
                patch("lib.model_server.ManagedProcess", return_value=proc) as mp, \
                patch("lib.model_server.wait_until_ready", side_effect=lambda probe, **_: probe()):
            assert server.start() is True
        assert mp.call_args.args[0] == ["/usr/local/bin/ollama", "serve"]
        assert mp.call_args.kwargs["env"]["OLLAMA_HOST"] == "localhost:11434"
        assert server.ownership is Ownership.OWNED
        proc.spawn.assert_called_once()
        server.stop()
        proc.terminate.assert_called_once_with(10.0)
        assert server.ownership is Ownership.NONE

    def test_start_is_idempotent_for_owned_server(self):
        server = self._make_server()
        proc = _fake_proc()
        server._proc = proc
        server.ownership = Ownership.OWNED
        with patch.object(server, "health_check", return_value=True), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is True
        mp.assert_not_called()

    def test_start_after_owned_server_died_reaps_and_readopts(self):
        server = self._make_server()
        dead = _fake_proc(alive=False)
        server._proc = dead
        server.ownership = Ownership.OWNED
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"version": "1"})):
            assert server.start() is True
        dead.terminate.assert_called_once()
        assert server.ownership is Ownership.EXTERNAL  # something else now answers

    def test_server_has_no_is_running_attribute(self):
        assert not hasattr(self._make_server(), "is_running")

    def test_start_spawn_redirects_output_to_log_in_new_session(self):
        """A real ManagedProcess: output goes to a log file, never a PIPE."""
        server = self._make_server()
        popen = MagicMock()
        popen.pid = 12345
        popen.poll.return_value = None
        probes = [ProbeResult(False, "connection refused"), ProbeResult(True, "Ollama 1", {"version": "1"})]
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch.object(OllamaServer, "_identity_probe", side_effect=probes), \
                patch("lib.model_server.wait_until_ready", side_effect=lambda probe, **_: probe()), \
                patch("lib.process_supervisor.subprocess.Popen", return_value=popen) as mock_popen:
            assert server.start() is True
        assert server.ownership is Ownership.OWNED
        assert server.pid == 12345
        assert server.log_path.endswith(".log")
        assert mock_popen.call_args.args[0] == ["/usr/bin/ollama", "serve"]
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["stdout"] not in (subprocess.PIPE, None)
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["start_new_session"] is True
        server.stop()
        assert server.ownership is Ownership.NONE

    def test_start_failure_reports_log_tail_and_stops(self):
        server = self._make_server()
        proc = _fake_proc(alive=False)
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch.object(OllamaServer, "_identity_probe", return_value=ProbeResult(False, "refused")), \
                patch("lib.model_server.ManagedProcess", return_value=proc), \
                patch("lib.model_server.wait_until_ready",
                      return_value=ProbeResult(False, "process exited before becoming ready")):
            assert server.start() is False
        assert "failed to become ready" in server.last_error
        assert "fatal: model not found" in server.last_error  # log tail included
        proc.terminate.assert_called_once()
        assert server.ownership is Ownership.NONE

    def test_start_spawn_executable_missing(self):
        server = self._make_server()
        proc = _fake_proc()
        proc.spawn.side_effect = FileNotFoundError("ollama")
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch("lib.model_server.ManagedProcess", return_value=proc):
            assert server.start() is False
        assert "executable not found" in server.last_error
        assert "ollama.com/download" in server.last_error
        assert server.ownership is Ownership.NONE

    def test_start_spawn_argv_error_does_not_spawn(self):
        server = self._make_server()
        with patch("lib.model_server.shutil.which", return_value="/usr/bin/ollama"), \
                patch.object(OllamaServer, "_spawn_argv", side_effect=ServerStartError("bad config")), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert server.start() is False
        mp.assert_not_called()
        assert server.last_error == "bad config"

    def test_spawn_env_preserves_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        assert self._make_server()._spawn_env()["HOME"] == "/home/someone"

    def test_spawn_env_brackets_ipv6_host(self):
        server = self._make_server(base_url="http://[::1]:11500")
        assert server._spawn_env()["OLLAMA_HOST"] == "[::1]:11500"

    def test_spawn_env_sets_ollama_host(self):
        server = self._make_server(base_url="http://127.0.0.1:11500")
        env = server._spawn_env()
        assert env["OLLAMA_HOST"] == "127.0.0.1:11500"

    def test_stop_external_is_noop(self):
        server = self._make_server()
        server.ownership = Ownership.EXTERNAL
        with patch("lib.model_server.requests.post") as mock_post:
            assert server.stop() is True
        mock_post.assert_not_called()

    def test_stop_owned_terminates(self):
        server = self._make_server()
        proc = _fake_proc()
        server._proc = proc
        server.ownership = Ownership.OWNED
        assert server.stop() is True
        proc.terminate.assert_called_once_with(10.0)
        assert server.ownership is Ownership.NONE
        assert server.process is None

    def test_stop_no_process(self):
        assert self._make_server().stop() is True  # no-op is success

    def test_detach_keeps_process_running(self):
        server = self._make_server()
        proc = _fake_proc()
        server._proc = proc
        server.ownership = Ownership.OWNED
        server.detach()
        proc.detach.assert_called_once()
        proc.terminate.assert_not_called()
        assert server.ownership is Ownership.EXTERNAL

    def test_is_model_available_found(self):
        server = self._make_server()
        payload = {"models": [{"name": "qwen3-vl:32b"}]}
        with patch("lib.model_server.requests.get", return_value=_resp(200, payload)):
            assert server.is_model_available("qwen3-vl:32b") is True

    def test_is_model_available_exact_match(self):
        server = self._make_server()
        payload = {"models": [{"name": "qwen3-coder:latest"}]}
        with patch("lib.model_server.requests.get", return_value=_resp(200, payload)):
            assert server.is_model_available("qwen3") is False
            assert server.is_model_available("qwen3-coder") is True

    def test_is_model_available_implicit_latest(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get", return_value=_resp(200, TAGS)):
            assert server.is_model_available("Drews54/llama3.2-vision-abliterated") is True
            assert server.is_model_available("drews54/LLAMA3.2-vision-abliterated:latest") is True

    def test_is_model_available_server_down(self):
        server = self._make_server()
        assert server.is_model_available("qwen3.6:35b-a3b") is False
        assert server.model_listing_failed is True

    def test_model_listing_failed_tracks_last_listing(self):
        server = self._make_server()
        assert server.model_listing_failed is False
        with patch("lib.model_server.requests.get", return_value=_resp(500, text="boom")):
            assert server.list_models() == []
        assert server.model_listing_failed is True
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"models": []})):
            assert server.list_models() == []  # empty, but a real answer
        assert server.model_listing_failed is False

    def test_get_capabilities_from_tags(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get", return_value=_resp(200, TAGS)), \
                patch("lib.model_server.requests.post") as post:
            assert server.get_capabilities("gpt-oss:120b") == {"completion", "tools", "thinking"}
        post.assert_not_called()

    def test_get_capabilities_falls_back_to_show(self):
        server = self._make_server()
        show = _resp(200, {"capabilities": ["tools", "completion", "vision"]})
        with patch("lib.model_server.requests.get", return_value=_resp(200, TAGS)), \
                patch("lib.model_server.requests.post", return_value=show) as post:
            caps = server.get_capabilities("Drews54/llama3.2-vision-abliterated")
        assert caps == {"tools", "completion", "vision"}
        assert post.call_args.args[0] == "http://localhost:11434/api/show"
        assert post.call_args.kwargs["json"] == {"model": "Drews54/llama3.2-vision-abliterated:latest"}

    def test_get_capabilities_unknown(self):
        server = self._make_server()
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"models": []})), \
                patch("lib.model_server.requests.post", return_value=_resp(404)):
            assert server.get_capabilities("missing:1b") is None

    def test_list_vision_models_filters(self):
        server = self._make_server()
        show = _resp(200, {"capabilities": ["completion", "vision"]})
        with patch("lib.model_server.requests.get", return_value=_resp(200, TAGS)), \
                patch("lib.model_server.requests.post", return_value=show):
            assert server.list_vision_models() == [
                "qwen3.6:35b-a3b", "gemma4:26b", "Drews54/llama3.2-vision-abliterated:latest",
            ]

    def test_model_store_path_local_vs_remote(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "store"))
        assert self._make_server().model_store_path() == tmp_path / "store"
        remote = self._make_server(base_url="http://gpu-box.example:11434")
        assert remote.model_store_path() is None

    def test_estimate_pull_uses_store_blobs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
        expected = PullEstimate("m", 1, 1)
        with patch("lib.model_server.fetch_pull_estimate", return_value=expected) as fetch:
            assert self._make_server().estimate_pull("m") is expected
        assert fetch.call_args.kwargs["blobs_dir"] == tmp_path / "blobs"

    def test_estimate_pull_remote_does_not_guess_platform(self):
        server = self._make_server(base_url="http://gpu-box.example:11434")
        with patch("lib.model_server.fetch_pull_estimate") as fetch:
            server.estimate_pull("m")
        assert fetch.call_args.kwargs.get("blobs_dir") is None
        assert fetch.call_args.kwargs["system"] is None

    def test_manual_pull_command(self):
        assert self._make_server().manual_pull_command("qwen3.6:35b-a3b") == (
            "ollama pull qwen3.6:35b-a3b"
        )

    def test_pull_model_streams_success(self):
        server = self._make_server()
        resp = _stream_resp([
            {"status": "pulling manifest"},
            {"status": "pulling sha256:ab", "digest": "sha256:ab", "total": 100, "completed": 50},
            b"",
            {"status": "success"},
        ])
        progress = MagicMock()
        with patch("lib.model_server.requests.post", return_value=resp) as post:
            assert server.pull_model("qwen3-vl:32b", progress=progress) is True
        assert post.call_args.args[0] == "http://localhost:11434/api/pull"
        assert post.call_args.kwargs["json"] == {"model": "qwen3-vl:32b", "stream": True}
        assert post.call_args.kwargs["stream"] is True
        assert post.call_args.kwargs["timeout"][1] >= 600
        assert progress.call_args_list[-1].args == ("success", 50, 100)
        resp.close.assert_called_once()

    def test_pull_model_error_event_returns_false(self, caplog):
        server = self._make_server()
        resp = _stream_resp([{"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}])
        with patch("lib.model_server.requests.post", return_value=resp):
            assert server.pull_model("nope:1b") is False
        assert "file does not exist" in caplog.text
        resp.close.assert_called_once()

    def test_pull_model_stream_ends_without_success_returns_false(self, caplog):
        server = self._make_server()
        resp = _stream_resp([{"status": "pulling manifest"}])
        with patch("lib.model_server.requests.post", return_value=resp):
            assert server.pull_model("m") is False
        assert "without a success status" in caplog.text

    def test_pull_model_non_200_returns_false(self):
        server = self._make_server()
        resp = _stream_resp([], status=500)
        with patch("lib.model_server.requests.post", return_value=resp):
            assert server.pull_model("m") is False
        resp.close.assert_called_once()

    def test_pull_model_connection_error_returns_false(self):
        # The hermetic fixture makes requests.post raise ConnectionError.
        assert self._make_server().pull_model("m") is False

    def test_pull_model_should_abort_closes_response(self, caplog):
        server = self._make_server()
        lines = [{"status": "pulling sha256:ab", "digest": "sha256:ab", "total": 100, "completed": i}
                 for i in range(5)] + [{"status": "success"}]
        resp = _stream_resp(lines)
        ticks = iter(range(0, 1000, 3))  # each clock() call advances 3s
        should_abort = MagicMock(side_effect=[None, "low disk", None, None])
        with patch("lib.model_server.requests.post", return_value=resp):
            ok = server.pull_model("m", should_abort=should_abort, clock=lambda: next(ticks))
        assert ok is False
        assert should_abort.call_count == 2
        assert "low disk" in caplog.text
        resp.close.assert_called_once()

    def test_pull_model_keyboard_interrupt_closes_and_reraises(self):
        server = self._make_server()
        resp = MagicMock(status_code=200)
        resp.iter_lines.side_effect = KeyboardInterrupt
        with patch("lib.model_server.requests.post", return_value=resp):
            with pytest.raises(KeyboardInterrupt):
                server.pull_model("m")
        resp.close.assert_called_once()

    def test_list_models(self):
        server = self._make_server()
        payload = {"models": [{"name": "qwen3-vl:32b"}]}
        with patch("lib.model_server.requests.get", return_value=_resp(200, payload)):
            assert server.list_models() == ["qwen3-vl:32b"]

    def test_get_state_models_loaded_from_api_ps(self):
        server = self._make_server()

        def fake_get(url, **_kwargs):
            if url.endswith("/api/version"):
                return _resp(200, {"version": "0.40.16"})
            if url.endswith("/api/ps"):
                return _resp(200, {"models": [{"name": "qwen3.6:35b-a3b", "size_vram": 2**30}]})
            raise AssertionError(f"unexpected GET {url}")

        with patch("lib.model_server.requests.get", side_effect=fake_get):
            state = server.get_state()
        assert state.is_running is True
        assert state.models_loaded == ["qwen3.6:35b-a3b"]
        assert state.memory_usage_mb == 1024
        assert state.version == "0.40.16"

    def test_load_model_payload(self):
        server = self._make_server()
        with patch("lib.model_server.requests.post", return_value=_resp(200)) as post:
            assert server.load_model("qwen3.6:35b-a3b") is True
        assert post.call_args.args[0] == "http://localhost:11434/api/generate"
        assert post.call_args.kwargs["json"] == {
            "model": "qwen3.6:35b-a3b", "keep_alive": "30m", "stream": False,
        }
        assert post.call_args.kwargs["timeout"] >= 300

    def test_load_model_http_error(self):
        server = self._make_server()
        with patch("lib.model_server.requests.post", return_value=_resp(500, text="no longer compatible")):
            assert server.load_model("broken") is False

    def test_unload_uses_generate_keep_alive_0(self):
        server = self._make_server()
        with patch("lib.model_server.requests.post", return_value=_resp(200)) as post:
            assert server.unload_model("qwen3.6:35b-a3b") is True
        assert post.call_args.args[0].endswith("/api/generate")
        assert post.call_args.kwargs["json"]["keep_alive"] == 0

    def test_installed_model_size_from_tags(self):
        server = self._make_server()
        tags = {"models": [
            {"name": "gpt-oss:120b", "model": "gpt-oss:120b", "size": 65_290_000_000},
            {"name": "nosize:1b", "model": "nosize:1b"},
        ]}
        with patch("lib.model_server.requests.get", return_value=_resp(200, tags)):
            assert server.installed_model_size("gpt-oss:120b") == 65_290_000_000
            assert server.installed_model_size("nosize:1b") is None
            assert server.installed_model_size("missing:1b") is None
        with patch("lib.model_server.requests.get", return_value=_resp(500, text="boom")):
            assert server.installed_model_size("gpt-oss:120b") is None

    def test_get_client(self):
        server = self._make_server(model="qwen3.6:35b-a3b", timeout=123)
        client = server.get_client()
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 123
        assert client.model == "qwen3.6:35b-a3b"
        # Vision calls read with the request timeout and connect with 10s.
        assert client.generate_timeout == 123
        assert client.connect_timeout == 10

    def test_get_client_timeout_600(self):
        client = self._make_server().get_client()
        assert client.timeout == 600 == DEFAULT_REQUEST_TIMEOUT
        assert client.generate_timeout == 600


# ---------------------------------------------------------------------------
# ModelServerManager
# ---------------------------------------------------------------------------


class TestModelServerManager:
    """Test ModelServerManager facade."""

    def _make_manager(self, **kwargs) -> ModelServerManager:
        return ModelServerManager(**kwargs)

    def test_server_property(self):
        manager = self._make_manager()
        assert manager.server is not None

    def test_state_property(self):
        manager = self._make_manager()
        state = manager.state
        assert isinstance(state, ServerState)

    def test_probe_never_starts(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start") as start:
            state = manager.probe()
        start.assert_not_called()
        assert state.is_running is False

    def test_ensure_running(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", return_value=True):
            assert manager.ensure_running() is True

    def test_ensure_running_already_running(self):
        manager = self._make_manager()
        with patch("lib.model_server.requests.get", return_value=_resp(200, {"version": "1"})), \
                patch("lib.model_server.ManagedProcess") as mp:
            assert manager.ensure_running() is True
        mp.assert_not_called()
        assert manager.server.ownership is Ownership.EXTERNAL

    def test_ensure_running_never_raises(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", side_effect=RuntimeError("kaboom")):
            assert manager.ensure_running() is False
        assert "kaboom" in manager.last_error

    # -- ensure_model / ensure_models ---------------------------------------

    @staticmethod
    def _ready(manager, *, installed=False, estimate=None, store=None, caps=None, vision=()):
        """Patch the manager's server: running, with the given model state.

        ``installed`` may be a list to give successive is_model_available
        results. Returns the patched server methods by name.
        """
        manager.ensure_running = MagicMock(return_value=True)
        server = manager.server
        mocks = {
            "is_model_available": MagicMock(
                side_effect=installed if isinstance(installed, list) else None,
                return_value=installed if not isinstance(installed, list) else None,
            ),
            "estimate_pull": MagicMock(return_value=estimate or PullEstimate("m", error="offline")),
            "model_store_path": MagicMock(return_value=store),
            "get_capabilities": MagicMock(return_value=caps),
            "list_vision_models": MagicMock(return_value=list(vision)),
            "list_models": MagicMock(return_value=list(vision)),
            "pull_model": MagicMock(return_value=True),
            "load_model": MagicMock(return_value=True),
            "health_check": MagicMock(return_value=True),
            "start": MagicMock(return_value=True),
        }
        for name, mock in mocks.items():
            setattr(server, name, mock)
        return mocks

    @staticmethod
    def _disk(free):
        return patch(
            "lib.model_availability.shutil.disk_usage",
            return_value=MagicMock(total=10 * free, used=9 * free, free=free),
        )

    def test_ensure_model_pulls_when_allowed_and_space_ok(self, tmp_path):
        manager = self._make_manager()
        m = self._ready(manager, installed=[False, True], store=tmp_path,
                        estimate=PullEstimate("m", 1 * GB, 1 * GB, source="url"))
        with self._disk(100 * GB):
            assert manager.ensure_model("qwen3-vl:32b", allow_pull=True) is True
        m["pull_model"].assert_called_once()
        assert m["pull_model"].call_args.args == ("qwen3-vl:32b",)
        assert callable(m["pull_model"].call_args.kwargs["should_abort"])  # disk guard
        m["load_model"].assert_not_called()  # warm-load is opt-in

    def test_ensure_model_pull_fails(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("m", 1 * GB, 1 * GB))
        m["pull_model"].return_value = False
        with self._disk(100 * GB):
            assert manager.ensure_model("nonexistent", allow_pull=True) is False
        assert "Failed to pull model 'nonexistent'" in caplog.text
        assert "ollama pull nonexistent" in caplog.text

    def test_ensure_model_pull_succeeds_but_not_listed(self, tmp_path, caplog):
        manager = self._make_manager()
        self._ready(manager, installed=False, store=tmp_path,
                    estimate=PullEstimate("m", 1 * GB, 1 * GB))
        with self._disk(100 * GB):
            assert manager.ensure_model("m:1b", allow_pull=True) is False
        assert "not listed by the server" in caplog.text

    def test_ensure_model_missing_without_pull_does_not_pull(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(
            manager, installed=False, store=tmp_path,
            estimate=PullEstimate("qwen3-vl:32b", 20_910_286_445, 20_910_286_445),
            vision=["qwen3.6:35b-a3b", "gemma4:26b"],
        )
        with self._disk(28_150_000_000):
            assert manager.ensure_model("qwen3-vl:32b", capability="vision") is False
        m["pull_model"].assert_not_called()
        text = caplog.text
        assert "Vision model 'qwen3-vl:32b' is not installed" in text
        assert "--pull" in text
        assert "ollama pull qwen3-vl:32b" in text
        assert "20.9 GB" in text
        assert "28.1 GB" in text  # free space is reported too
        assert "26.3 GB" in text  # download + safety margin
        assert "5 GiB (--min-free-gb) safety margin" in text  # floor in the flag's unit
        assert "qwen3.6:35b-a3b, gemma4:26b" in text

    def test_ensure_model_listing_failure_is_not_reported_as_missing(self, tmp_path, caplog):
        # The server answered the reachability check but /api/tags then
        # fails: do not say "not installed", fetch a manifest, or pull.
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path)
        manager.server.model_listing_failed = True
        assert manager.ensure_model("qwen3.6:35b-a3b", allow_pull=True, capability="vision") is False
        m["estimate_pull"].assert_not_called()
        m["pull_model"].assert_not_called()
        assert "Could not list the models installed" in caplog.text
        assert "not installed" not in caplog.text
        assert "ollama pull" not in caplog.text

    def test_ensure_model_listing_failure_real_tags(self, caplog):
        # End to end through OllamaServer._tags: /api/version OK, /api/tags 500.
        manager = self._make_manager()
        manager.ensure_running = MagicMock(return_value=True)
        estimate = MagicMock(name="estimate_pull")
        manager.server.estimate_pull = estimate

        def fake_get(url, **_kwargs):
            return _resp(500, text="boom") if url.endswith("/api/tags") else _resp(200, {"version": "1"})

        with patch("lib.model_server.requests.get", side_effect=fake_get):
            assert manager.ensure_model("qwen3.6:35b-a3b") is False
        assert manager.server.model_listing_failed is True
        estimate.assert_not_called()
        assert "Could not list the models installed" in caplog.text

    def test_ensure_model_never_pulls_by_default(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=False)
        assert manager.ensure_model("qwen3-vl:32b") is False
        m["pull_model"].assert_not_called()

    def test_ensure_model_refuses_insufficient_disk(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("qwen3-vl:32b", 20_910_286_445, 20_910_286_445),
                        vision=["gemma4:26b"])
        with self._disk(12 * GB):
            assert manager.ensure_model("qwen3-vl:32b", allow_pull=True, capability="vision") is False
        m["pull_model"].assert_not_called()
        assert "Not enough disk space to pull 'qwen3-vl:32b'" in caplog.text
        assert str(tmp_path) in caplog.text
        assert "ollama pull qwen3-vl:32b" in caplog.text
        assert "gemma4:26b" in caplog.text

    def test_ensure_model_min_free_bytes_raises_requirement(self, tmp_path):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("m", 20 * GB, 20 * GB))
        with self._disk(28 * GB):
            assert manager.ensure_model("m", allow_pull=True, min_free_bytes=50 * GiB) is False
            assert manager.ensure_model("m", allow_pull=True, min_free_bytes=0, dry_run=True) is True
        m["pull_model"].assert_not_called()

    def test_ensure_model_refuses_unknown_size(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("m", error="registry requires authentication (HTTP 401)"))
        with self._disk(100 * GB):
            assert manager.ensure_model("hf.co/x/y:Q4", allow_pull=True) is False
        m["pull_model"].assert_not_called()
        assert "Could not determine the download size" in caplog.text
        assert "ollama pull hf.co/x/y:Q4" in caplog.text
        assert f"100.0 GB free on {tmp_path}" in caplog.text

    def test_ensure_model_refuses_when_disk_unmeasurable(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("m", GB, GB))
        with patch("lib.model_availability.shutil.disk_usage", side_effect=OSError("io")):
            assert manager.ensure_model("m", allow_pull=True) is False
        m["pull_model"].assert_not_called()
        assert "cannot measure free space" in caplog.text

    def test_ensure_model_not_found_in_registry(self, caplog):
        manager = self._make_manager()
        m = self._ready(
            manager, installed=False, vision=["gemma4:26b"],
            estimate=PullEstimate("qwen2-vl:7b", source="https://registry/x", not_found=True,
                                  error="not found in registry (HTTP 404 from https://registry/x)"),
        )
        assert manager.ensure_model("qwen2-vl:7b", allow_pull=True, capability="vision") is False
        m["pull_model"].assert_not_called()
        assert "was not found in the registry" in caplog.text
        assert "Installed vision-capable models: gemma4:26b" in caplog.text

    def test_ensure_model_rejects_non_vision(self, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=True, caps={"completion", "tools", "thinking"},
                        vision=["qwen3.6:35b-a3b"])
        assert manager.ensure_model("gpt-oss:120b", capability="vision", warm=True) is False
        m["load_model"].assert_not_called()
        assert "does not support vision input" in caplog.text
        assert "qwen3.6:35b-a3b" in caplog.text

    def test_ensure_model_unknown_capabilities_only_warns(self, caplog):
        manager = self._make_manager()
        self._ready(manager, installed=True, caps=None)
        assert manager.ensure_model("old-model", capability="vision") is True
        assert "Could not determine the capabilities" in caplog.text

    def test_ensure_model_installed_no_pull_no_load(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=True, caps={"vision"})
        assert manager.ensure_model("qwen3.6:35b-a3b", capability="vision", allow_pull=True) is True
        m["pull_model"].assert_not_called()
        m["estimate_pull"].assert_not_called()
        m["load_model"].assert_not_called()

    def test_ensure_model_warm_loads(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        m["load_model"].return_value = False
        # A failed warm-up is only a warning.
        assert manager.ensure_model("qwen3.6:35b-a3b", warm=True) is True
        m["load_model"].assert_called_once_with("qwen3.6:35b-a3b")

    def test_ensure_model_warm_loads_after_pull(self, tmp_path):
        manager = self._make_manager()
        m = self._ready(manager, installed=[False, True], store=tmp_path, caps={"vision"},
                        estimate=PullEstimate("m", GB, GB))
        with self._disk(100 * GB):
            assert manager.ensure_model("m", allow_pull=True, warm=True, capability="vision") is True
        m["load_model"].assert_called_once_with("m")

    def test_ensure_model_not_running(self):
        manager = self._make_manager()
        manager.ensure_running = MagicMock(return_value=False)
        with patch.object(manager.server, "is_model_available") as available:
            assert manager.ensure_model("x") is False
        available.assert_not_called()

    def test_ensure_models_dry_run_never_starts_pulls_or_loads(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="lib.model_server")
        manager = self._make_manager()
        m = self._ready(manager, installed=[True, False], store=tmp_path,
                        estimate=PullEstimate("b", GB, GB))
        with self._disk(100 * GB):
            assert manager.ensure_models(["a", "b"], dry_run=True, warm=True, allow_pull=True) is True
        manager.ensure_running.assert_not_called()
        m["start"].assert_not_called()
        m["pull_model"].assert_not_called()
        m["load_model"].assert_not_called()
        m["health_check"].assert_called_once()
        assert "Dry run: would pull 'b'" in caplog.text

    def test_ensure_model_dry_run_reports_what_a_real_run_would(self, tmp_path):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path, estimate=PullEstimate("b", GB, GB))
        with self._disk(100 * GB):
            assert manager.ensure_model("b", dry_run=True) is False  # no --pull
        with self._disk(2 * GB):
            assert manager.ensure_model("b", dry_run=True, allow_pull=True) is False  # no room
        m["pull_model"].assert_not_called()

    def test_ensure_model_dry_run_unhealthy(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start") as start:
            assert manager.ensure_model("a", dry_run=True) is False
        start.assert_not_called()

    def test_remote_server_pull_warns_and_proceeds(self, caplog):
        manager = self._make_manager(base_url="http://gpu-box.example:11434")
        m = self._ready(manager, installed=[False, True], store=None,
                        estimate=PullEstimate("m", 3 * GB, 3 * GB))
        assert manager.ensure_model("m", allow_pull=True) is True
        assert "Cannot verify free disk space on remote server" in caplog.text
        assert m["pull_model"].call_args.kwargs["should_abort"] is None

    def test_ensure_models(self):
        manager = self._make_manager()
        self._ready(manager, installed=True)
        assert manager.ensure_models(["qwen3-vl:32b", "gpt-oss:120b"]) is True

    def test_ensure_models_one_fails(self, tmp_path):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path, estimate=PullEstimate("m", GB, GB))
        m["pull_model"].return_value = False
        with self._disk(100 * GB):
            assert manager.ensure_models(["qwen3-vl:32b", "nonexistent"], allow_pull=True) is False
        assert m["pull_model"].call_count == 2  # no short-circuit

    def test_ensure_models_dedupes_and_does_not_short_circuit(self):
        manager = self._make_manager()
        manager.ensure_running = MagicMock(return_value=True)
        manager._ensure_one = MagicMock(side_effect=[False, True])
        assert manager.ensure_models(["a", "b", "A:latest"], warm=True) is False
        assert [c.args[0] for c in manager._ensure_one.call_args_list] == ["a", "b"]
        assert manager._ensure_one.call_args.kwargs["warm"] is True
        manager.ensure_running.assert_called_once()

    def test_ensure_models_lists_vision_models_once(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, vision=["gemma4:26b"])
        assert manager.ensure_models(["x", "y"], capability="vision") is False
        m["list_vision_models"].assert_called_once()

    def test_ensure_models_empty(self):
        manager = self._make_manager()
        manager.ensure_running = MagicMock()
        assert manager.ensure_models([]) is True
        manager.ensure_running.assert_not_called()

    def test_ensure_models_per_item_capability_overrides_keyword(self):
        manager = self._make_manager()
        manager.ensure_running = MagicMock(return_value=True)
        manager._ensure_one = MagicMock(return_value=True)
        assert manager.ensure_models(
            [("qwen3.6:35b-a3b", "vision"), ("gpt-oss:120b", "completion"), "plain", ("dflt", None)],
            capability="tools",
        ) is True
        caps = {c.args[0]: c.kwargs["capability"] for c in manager._ensure_one.call_args_list}
        assert caps == {
            "qwen3.6:35b-a3b": "vision",
            "gpt-oss:120b": "completion",
            "plain": "tools",
            "dflt": "tools",
        }

    def test_ensure_models_capability_checked_per_model(self, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        m["get_capabilities"].side_effect = lambda name: {
            "qwen3.6:35b-a3b": {"completion", "vision", "thinking"},
            "gpt-oss:120b": {"completion", "tools", "thinking"},
        }[name]
        assert manager.ensure_models(
            [("qwen3.6:35b-a3b", "vision"), ("gpt-oss:120b", "completion")]
        ) is True
        # gpt-oss is not a vision model, but it is only asked for completion.
        assert manager.ensure_models([("gpt-oss:120b", "vision")]) is False
        assert "does not support vision input" in caplog.text

    def test_ensure_models_missing_completion_is_reported(self, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        m["get_capabilities"].return_value = {"embedding"}
        assert manager.ensure_models([("nomic-embed-text", "completion")]) is False
        assert "does not support text completion" in caplog.text

    def test_ensure_models_hints_built_per_capability(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, vision=["gemma4:26b"])
        assert manager.ensure_models([("x", "vision"), ("y", "completion")]) is False
        m["list_vision_models"].assert_called_once()
        m["list_models"].assert_called_once()

    def test_ensure_models_duplicate_keeps_vision_capability(self):
        manager = self._make_manager()
        manager.ensure_running = MagicMock(return_value=True)
        manager._ensure_one = MagicMock(return_value=True)
        manager.ensure_models([("m:latest", "completion"), ("M", "vision")])
        assert manager._ensure_one.call_count == 1
        assert manager._ensure_one.call_args.args[0] == "m:latest"
        assert manager._ensure_one.call_args.kwargs["capability"] == "vision"

    def test_ensure_models_warm_models_restricts_warming(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        assert manager.ensure_models(
            [("qwen3.6:35b-a3b", "vision"), ("gpt-oss:120b", "completion")],
            warm=True,
            warm_models=["QWEN3.6:35b-a3b"],
        ) is True
        m["load_model"].assert_called_once_with("qwen3.6:35b-a3b")

    def test_ensure_models_warms_all_by_default(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        assert manager.ensure_models(["a", "b"], warm=True) is True
        assert [c.args[0] for c in m["load_model"].call_args_list] == ["a", "b"]

    def test_ensure_models_warm_models_ignored_without_warm(self):
        manager = self._make_manager()
        m = self._ready(manager, installed=True)
        assert manager.ensure_models(["a", "b"], warm=False, warm_models=["a"]) is True
        m["load_model"].assert_not_called()

    def test_combined_disk_check_refuses_before_any_pull(self, tmp_path, caplog):
        manager = self._make_manager()
        estimates = {
            "qwen3.6:35b-a3b": PullEstimate("qwen3.6:35b-a3b", 23 * GB, 23 * GB, source="u1"),
            "gpt-oss:120b": PullEstimate("gpt-oss:120b", 65 * GB, 65 * GB, source="u2"),
        }
        m = self._ready(manager, installed=False, store=tmp_path)
        m["estimate_pull"].side_effect = lambda name: estimates[name]
        # Each model fits on its own (70 GB free), but not both together.
        with self._disk(80 * GB):
            ok = manager.ensure_models(
                [("qwen3.6:35b-a3b", "vision"), ("gpt-oss:120b", "completion")],
                allow_pull=True, warm=True,
            )
        assert ok is False
        m["pull_model"].assert_not_called()
        m["load_model"].assert_not_called()
        text = caplog.text
        assert "Not enough disk space to pull all missing models" in text
        assert "Nothing was pulled" in text
        assert "qwen3.6:35b-a3b: 23.0 GB" in text
        assert "gpt-oss:120b: 65.0 GB" in text
        assert "88.0 GB download" in text
        assert m["estimate_pull"].call_count == 2  # one estimate per model

    def test_combined_disk_check_dry_run_also_refuses(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=False, store=tmp_path,
                        estimate=PullEstimate("m", 40 * GB, 40 * GB))
        with self._disk(60 * GB):
            assert manager.ensure_models(["a", "b"], allow_pull=True, dry_run=True) is False
        assert "Dry run: Not enough disk space to pull all missing models" in caplog.text
        m["pull_model"].assert_not_called()

    def test_combined_disk_check_passes_and_pulls_each_once(self, tmp_path):
        manager = self._make_manager()
        # a missing, b missing (combined check), then each listed after its pull.
        m = self._ready(manager, installed=[False, False, True, True], store=tmp_path,
                        estimate=PullEstimate("m", 10 * GB, 10 * GB))
        with self._disk(100 * GB):
            assert manager.ensure_models(["a", "b"], allow_pull=True) is True
        assert [c.args[0] for c in m["pull_model"].call_args_list] == ["a", "b"]
        assert m["estimate_pull"].call_count == 2  # reused, not fetched again

    def test_combined_disk_check_reports_installed_capability_problems(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=[True, False, False], store=tmp_path,
                        caps={"completion"}, estimate=PullEstimate("m", 40 * GB, 40 * GB))
        with self._disk(60 * GB):
            assert manager.ensure_models(
                [("have", "vision"), ("b", None), ("c", None)], allow_pull=True
            ) is False
        assert "'have' is installed but does not support vision input" in caplog.text
        m["pull_model"].assert_not_called()

    def test_single_missing_model_uses_per_model_check(self, tmp_path, caplog):
        manager = self._make_manager()
        m = self._ready(manager, installed=[True, False], store=tmp_path,
                        estimate=PullEstimate("b", 40 * GB, 40 * GB))
        with self._disk(30 * GB):
            assert manager.ensure_models(["a", "b"], allow_pull=True) is False
        assert "Not enough disk space to pull 'b'" in caplog.text
        assert "all missing models" not in caplog.text
        m["pull_model"].assert_not_called()

    def test_unload_model_delegates(self):
        manager = self._make_manager()
        with patch.object(manager.server, "unload_model", return_value=True) as unload:
            assert manager.unload_model("gpt-oss:120b") is True
        unload.assert_called_once_with("gpt-oss:120b")

    def test_unload_model_failure_is_reported_not_raised(self, caplog):
        manager = self._make_manager()
        with patch.object(manager.server, "unload_model", return_value=False):
            assert manager.unload_model("m") is False
        with patch.object(manager.server, "unload_model", side_effect=RuntimeError("boom")):
            assert manager.unload_model("m") is False
        assert "Could not unload model 'm'" in caplog.text

    def test_unload_model_works_on_adopted_server(self):
        manager = self._make_manager()
        manager.server.ownership = Ownership.EXTERNAL
        with patch("lib.model_server.requests.post", return_value=_resp(200)) as post:
            assert manager.unload_model("qwen3.6:35b-a3b") is True
        assert post.call_args.kwargs["json"] == {
            "model": "qwen3.6:35b-a3b", "keep_alive": 0, "stream": False,
        }

    def test_installed_model_size_delegates(self):
        manager = self._make_manager()
        with patch.object(manager.server, "installed_model_size", return_value=123):
            assert manager.installed_model_size("m") == 123
        with patch.object(manager.server, "installed_model_size", side_effect=RuntimeError("x")):
            assert manager.installed_model_size("m") is None

    def test_get_client(self):
        manager = self._make_manager()
        with patch.object(manager, "ensure_running", return_value=True):
            client = manager.get_client()
            assert isinstance(client, OllamaClient)
            assert hasattr(client, "chat_with_images")
            assert manager.get_client() is client  # cached

    def test_get_client_not_running_raises(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", return_value=False):
            with pytest.raises(RuntimeError, match="not running"):
                manager.get_client()
        with patch.object(manager.server, "start", return_value=False):
            with pytest.raises(ModelServerError):
                manager.get_client()

    def test_get_state(self):
        manager = self._make_manager()
        state = manager.get_state()
        assert isinstance(state, ServerState)

    def test_list_models(self):
        manager = self._make_manager()
        with patch.object(manager.server, "list_models", return_value=["qwen3-vl:32b"]):
            assert manager.list_models() == ["qwen3-vl:32b"]

    def test_close(self):
        manager = self._make_manager()
        with patch.object(manager.server, "stop") as mock_stop:
            manager.close()
            mock_stop.assert_called_once()

    def test_close_idempotent(self):
        manager = self._make_manager()
        client = MagicMock()
        manager._client = client
        with patch.object(manager.server, "stop") as mock_stop:
            manager.close()
            manager.close()
        mock_stop.assert_called_once()
        client.close.assert_called_once()

    def test_close_keep_running_detaches(self):
        manager = self._make_manager(keep_running=True)
        with patch.object(manager.server, "stop") as stop, \
                patch.object(manager.server, "detach") as detach:
            manager.close()
        detach.assert_called_once()
        stop.assert_not_called()

    def test_close_swallows_errors(self):
        manager = self._make_manager()
        with patch.object(manager.server, "stop", side_effect=RuntimeError("boom")):
            manager.close()  # must not raise

    def test_context_manager(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", return_value=True), \
                patch.object(manager.server, "stop") as stop:
            with manager:
                pass  # __exit__ calls close()
        stop.assert_called_once()

    def test_enter_raises_when_unavailable(self):
        manager = self._make_manager()
        with patch("lib.model_server.shutil.which", return_value=None):
            with pytest.raises(ModelServerError, match="not running"):
                with manager:
                    pass



# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TestBaseModelServer:
    """The lifecycle template's abstract hooks and "unknown" defaults."""

    def test_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModelServer()  # type: ignore[abstract]

    def test_base_class_model_query_defaults(self):
        server = OllamaServer()
        # The non-abstract defaults report "unknown" (OllamaServer overrides
        # them), so a server that cannot answer refuses automatic pulls.
        assert BaseModelServer.model_store_path(server) is None
        assert BaseModelServer.get_capabilities(server, "m") is None
        assert BaseModelServer.list_vision_models(server) == []
        estimate = BaseModelServer.estimate_pull(server, "m")
        assert not estimate.known and "not supported" in estimate.error
        assert BaseModelServer.manual_pull_command(server, "m") == "ollama pull m"
