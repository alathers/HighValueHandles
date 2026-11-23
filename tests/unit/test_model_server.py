"""Unit tests for ModelServerManager — MpltxServer, OllamaServer, ModelServerManager.

Tests cover:
  - ServerBackend enum, ServerState dataclass
  - MpltxServer start/stop/health-check/pull/load/unload (mocked HTTP)
  - OllamaServer start/stop/health-check (mocked HTTP)
  - ModelServerManager ensure_model/ensure_models/get_client (mocked)
  - Context manager usage
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from lib.model_server import (
    BaseModelServer,
    MpltxServer,
    ModelServerManager,
    OllamaServer,
    ServerBackend,
    ServerState,
    create_server_manager,
    create_server_manager_from_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ServerBackend / ServerState
# ---------------------------------------------------------------------------


class TestServerBackend:
    """Verify ServerBackend enum."""

    def test_two_backends(self):
        assert len(ServerBackend) == 2

    def test_mpltx_value(self):
        assert ServerBackend.MPLTX.value is not None

    def test_ollama_value(self):
        assert ServerBackend.OLLAMA.value is not None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("ollama", ServerBackend.OLLAMA),
            ("mpltx", ServerBackend.MPLTX),
            ("OLLAMA", ServerBackend.OLLAMA),
            (" Mpltx ", ServerBackend.MPLTX),
            (ServerBackend.OLLAMA, ServerBackend.OLLAMA),
        ],
    )
    def test_parse_accepts_strings_and_members(self, value, expected):
        assert ServerBackend.parse(value) is expected

    @pytest.mark.parametrize("value", ["", "llama", None, 1])
    def test_parse_rejects_unknown(self, value):
        with pytest.raises(ValueError, match="Unknown backend"):
            ServerBackend.parse(value)

    @pytest.mark.parametrize(
        ("value", "server_cls"), [("ollama", OllamaServer), ("mpltx", MpltxServer)]
    )
    def test_manager_accepts_cli_string(self, value, server_cls):
        """Regression: the CLI passes plain strings; they must select the right server."""
        manager = ModelServerManager(backend=value)
        assert isinstance(manager.server, server_cls)


class TestServerState:
    """Test ServerState dataclass defaults."""

    def test_defaults(self):
        state = ServerState(backend=ServerBackend.MPLTX)
        assert state.is_running is False
        assert state.base_url == ""
        assert state.pid == 0
        assert state.models_loaded == []
        assert state.memory_usage_mb == 0
        assert state.gpu_usage_percent == 0.0
        assert state.cold_start_seconds == 0.0
        assert state.uptime_seconds == 0.0
        assert state.error_message is None


# ---------------------------------------------------------------------------
# MpltxServer (mocked HTTP)
# ---------------------------------------------------------------------------


class TestMpltxServer:
    """Test MpltxServer lifecycle with mocked HTTP."""

    def _make_server(self, **kwargs) -> MpltxServer:
        """Create a server with mocked subprocess and requests."""
        return MpltxServer(**kwargs)

    def test_backend_property(self):
        server = self._make_server()
        assert server.backend == ServerBackend.MPLTX

    def test_default_base_url(self):
        server = self._make_server()
        assert server.base_url == "http://localhost:11435"

    def test_custom_port(self):
        server = self._make_server(port=8080)
        assert server.base_url == "http://localhost:8080"

    def test_default_home_dir(self):
        server = self._make_server()
        assert str(server.home_dir).endswith(".mpltx")

    def test_build_env(self):
        server = self._make_server()
        env = server._build_env()
        assert "MPLTX_HOME" in env
        assert "MPLTX_PORT" in env

    def test_build_args(self):
        server = self._make_server(port=8080, gpu_layers=32, context_size=8192)
        args = server._build_args()
        assert "mpltx" in args[0] or args[0].endswith("mpltx")
        assert "--port" in args
        assert "8080" in args
        assert "--gpu-layers" in args
        assert "32" in args
        assert "--ctx-size" in args
        assert "8192" in args

    def test_build_args_with_model(self):
        server = self._make_server(model="qwen3-vl:32b")
        args = server._build_args()
        assert "--model" in args
        assert "qwen3-vl:32b" in args

    @patch.object(MpltxServer, "health_check", return_value=False)
    @patch("subprocess.Popen")
    def test_start_success(self, mock_popen, _mock_health):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Patch _wait_for_healthy to return True
        with patch.object(MpltxServer, "_wait_for_healthy", return_value=True):
            server = self._make_server()
            result = server.start()

        assert result is True
        assert server.process is mock_proc
        assert server.process.pid == 12345

    @patch.object(MpltxServer, "health_check", return_value=False)
    @patch("subprocess.Popen")
    def test_start_failure(self, mock_popen, _mock_health):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Patch _wait_for_healthy to return False
        with patch.object(MpltxServer, "_wait_for_healthy", return_value=False):
            server = self._make_server()
            result = server.start()

        assert result is False

    @patch("subprocess.Popen")
    def test_start_file_not_found(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError("mpltx not found")
        server = self._make_server()
        result = server.start()
        assert result is False

    @patch("os.kill")
    def test_stop_graceful(self, mock_kill):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = None

        server = self._make_server()
        server.process = mock_proc
        server.start_time = 100.0

        result = server.stop()

        assert result is True
        assert server.process is None
        assert server.start_time == 0.0

    def test_stop_no_process(self):
        server = self._make_server()
        assert server.stop() is True  # No-op is success

    def test_health_check_running(self):
        server = self._make_server()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server.process = mock_proc

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp
            assert server.health_check() is True

    def test_health_check_not_running(self):
        server = self._make_server()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Dead process
        server.process = mock_proc
        assert server.health_check() is False

    def test_health_check_request_fails(self):
        server = self._make_server()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server.process = mock_proc

        with patch("requests.get", side_effect=Exception("connection refused")):
            assert server.health_check() is False

    def test_is_model_available_found(self):
        server = self._make_server()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [{"name": "qwen3-vl:32b"}, {"name": "gpt-oss:120b"}]
            }
            mock_get.return_value = mock_resp
            assert server.is_model_available("qwen3-vl:32b") is True
            assert server.is_model_available("nonexistent") is False

    def test_is_model_available_request_fails(self):
        server = self._make_server()
        with patch("requests.get", side_effect=Exception("connection refused")):
            assert server.is_model_available("any-model") is False

    def test_pull_model_success(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.pull_model("qwen3-vl:32b") is True

    def test_pull_model_failure(self):
        server = self._make_server()
        with patch("requests.post", side_effect=Exception("connection refused")):
            assert server.pull_model("qwen3-vl:32b") is False

    def test_list_models(self):
        server = self._make_server()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [{"name": "qwen3-vl:32b"}, {"name": "gpt-oss:120b"}]
            }
            mock_get.return_value = mock_resp
            assert server.list_models() == ["qwen3-vl:32b", "gpt-oss:120b"]

    def test_list_models_request_fails(self):
        server = self._make_server()
        with patch("requests.get", side_effect=Exception("connection refused")):
            assert server.list_models() == []

    def test_get_resource_usage_dead_process(self):
        server = self._make_server()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Dead
        server.process = mock_proc
        usage = server.get_resource_usage()
        assert usage == {"memory_mb": 0.0, "gpu_percent": 0.0, "cpu_percent": 0.0}

    def test_get_resource_usage_alive(self):
        server = self._make_server()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server.process = mock_proc

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "memory_used": 4294967296,  # 4GB
                "gpu_utilization": 75.0,
                "cpu_usage": 25.0,
            }
            mock_get.return_value = mock_resp
            usage = server.get_resource_usage()
            assert usage["memory_mb"] == 4096.0
            assert usage["gpu_percent"] == 75.0
            assert usage["cpu_percent"] == 25.0

    def test_load_model_success(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.load_model("qwen3-vl:32b") is True

    def test_unload_model_success(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.unload_model("qwen3-vl:32b") is True

    def test_get_client(self):
        server = self._make_server()
        # Should return an OllamaClient (we don't need to test its internals)
        client = server.get_client()
        assert client is not None
        assert hasattr(client, "image_to_text")


# ---------------------------------------------------------------------------
# OllamaServer (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOllamaServer:
    """Test OllamaServer lifecycle with mocked HTTP."""

    def _make_server(self, **kwargs) -> OllamaServer:
        return OllamaServer(**kwargs)

    def test_backend_property(self):
        server = self._make_server()
        assert server.backend == ServerBackend.OLLAMA

    def test_default_base_url(self):
        server = self._make_server()
        assert server.base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        server = self._make_server(base_url="http://remote:11434")
        assert server.base_url == "http://remote:11434"

    def test_health_check_success(self):
        server = self._make_server()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp
            assert server.health_check() is True

    def test_health_check_failure(self):
        server = self._make_server()
        with patch("requests.get", side_effect=Exception("connection refused")):
            assert server.health_check() is False

    def test_start_already_running(self):
        server = self._make_server()
        with patch.object(server, "health_check", return_value=True):
            assert server.start() is True

    def test_start_not_auto_start(self):
        server = self._make_server(auto_start=False)
        with patch.object(server, "health_check", return_value=False):
            assert server.start() is False

    def test_stop_success(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.stop() is True

    def test_stop_failure(self):
        server = self._make_server()
        with patch("requests.post", side_effect=Exception("connection refused")):
            assert server.stop() is False

    def test_is_model_available_found(self):
        server = self._make_server()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [{"name": "qwen3-vl:32b"}]
            }
            mock_get.return_value = mock_resp
            assert server.is_model_available("qwen3-vl:32b") is True

    def test_pull_model(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.pull_model("qwen3-vl:32b") is True

    def test_list_models(self):
        server = self._make_server()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [{"name": "qwen3-vl:32b"}]
            }
            mock_get.return_value = mock_resp
            assert server.list_models() == ["qwen3-vl:32b"]

    def test_get_resource_usage_returns_zeros(self):
        server = self._make_server()
        # OllamaServer returns zeros (no resource monitoring implemented)
        assert server.get_resource_usage() == {
            "memory_mb": 0.0,
            "gpu_percent": 0.0,
            "cpu_percent": 0.0,
        }

    def test_load_model(self):
        server = self._make_server()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            assert server.load_model("qwen3-vl:32b") is True

    def test_get_client(self):
        server = self._make_server()
        client = server.get_client()
        assert client is not None


# ---------------------------------------------------------------------------
# ModelServerManager (mocked)
# ---------------------------------------------------------------------------


class TestModelServerManager:
    """Test ModelServerManager facade."""

    def _make_manager(self, backend=ServerBackend.MPLTX, **kwargs) -> ModelServerManager:
        return ModelServerManager(backend=backend, **kwargs)

    def test_init_mpltx(self):
        manager = self._make_manager()
        assert manager._backend == ServerBackend.MPLTX
        assert manager._server is not None

    def test_init_ollama(self):
        manager = self._make_manager(backend=ServerBackend.OLLAMA)
        assert manager._backend == ServerBackend.OLLAMA

    def test_server_property(self):
        manager = self._make_manager()
        assert manager.server is not None

    def test_state_property(self):
        manager = self._make_manager()
        state = manager.state
        assert isinstance(state, ServerState)
        assert state.backend == ServerBackend.MPLTX

    def test_ensure_running_mpltx(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", return_value=True):
            assert manager.ensure_running() is True

    def test_ensure_running_already_running(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_running", True):
            with patch.object(manager.server, "health_check", return_value=True):
                assert manager.ensure_running() is True

    def test_ensure_model(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_model_available", return_value=False):
            with patch.object(manager.server, "pull_model", return_value=True):
                with patch.object(manager.server, "load_model", return_value=True):
                    assert manager.ensure_model("qwen3-vl:32b") is True

    def test_ensure_model_pull_fails(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_model_available", return_value=False):
            with patch.object(manager.server, "pull_model", return_value=False):
                assert manager.ensure_model("nonexistent") is False

    def test_ensure_models(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_model_available", return_value=True):
            assert manager.ensure_models(["qwen3-vl:32b", "gpt-oss:120b"]) is True

    def test_ensure_models_one_fails(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_model_available", return_value=False):
            with patch.object(manager.server, "pull_model", return_value=False):
                assert manager.ensure_models(["qwen3-vl:32b", "nonexistent"]) is False

    def test_get_client(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_running", True):
            client = manager.get_client()
            assert client is not None

    def test_get_client_not_running_raises(self):
        manager = self._make_manager()
        with patch.object(manager.server, "is_running", False):
            with patch.object(manager.server, "start", return_value=False):
                with pytest.raises(RuntimeError, match="not running"):
                    manager.get_client()

    def test_get_state(self):
        manager = self._make_manager()
        state = manager.get_state()
        assert isinstance(state, ServerState)

    def test_get_resource_usage(self):
        manager = self._make_manager()
        usage = manager.get_resource_usage()
        assert isinstance(usage, dict)

    def test_list_models(self):
        manager = self._make_manager()
        with patch.object(manager.server, "list_models", return_value=["qwen3-vl:32b"]):
            assert manager.list_models() == ["qwen3-vl:32b"]

    def test_close(self):
        manager = self._make_manager()
        with patch.object(manager.server, "stop") as mock_stop:
            manager.close()
            mock_stop.assert_called_once()

    def test_context_manager(self):
        manager = self._make_manager()
        with patch.object(manager.server, "start", return_value=True):
            with manager:
                pass  # __exit__ calls close()

    def test_init_from_config(self):
        config = MagicMock()
        config.mpltx_home = "/tmp/mpltx"
        config.mpltx_port = 8080
        config.model = "qwen3-vl:32b"
        config.mpltx_gpu_layers = 32
        config.mpltx_context_size = 8192

        manager = ModelServerManager(backend=ServerBackend.MPLTX, config=config)
        assert manager._server is not None

    def test_init_from_config_ollama(self):
        config = MagicMock()
        config.ollama_url = "http://remote:11434"
        config.model = "qwen3-vl:32b"

        manager = ModelServerManager(backend=ServerBackend.OLLAMA, config=config)
        assert manager._server is not None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Test create_server_manager() and create_server_manager_from_config()."""

    def test_create_server_manager(self):
        manager = create_server_manager()
        assert isinstance(manager, ModelServerManager)

    def test_create_server_manager_custom_backend(self):
        manager = create_server_manager(backend=ServerBackend.OLLAMA)
        assert manager._backend == ServerBackend.OLLAMA

    def test_create_server_manager_from_config(self):
        config = MagicMock()
        config.mpltx_home = "/tmp/mpltx"
        config.mpltx_port = 8080
        config.model = "qwen3-vl:32b"
        config.mpltx_gpu_layers = 16
        config.mpltx_context_size = 4096

        manager = create_server_manager_from_config(config)
        assert isinstance(manager, ModelServerManager)
        assert manager._server is not None