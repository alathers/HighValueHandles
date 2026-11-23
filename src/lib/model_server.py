"""MPLTX model serving manager.

Manages the full lifecycle of model serving via MPLTX: start, stop,
health-check, model pull, resource monitoring, and cold-start management.

Replaces the current external-Ollama assumption with codebase-managed
model serving. The serving manager is pluggable — swap backends by
changing the concrete implementation.

Current backends:
    - MpltxServer: Native MPLTX serving (process management)
    - OllamaServer: Ollama with auto-start (backward compatible)

Usage::

    from lib.model_server import ModelServerManager, ServerBackend

    manager = ModelServerManager(backend=ServerBackend.MPLTX)

    # Ensure model is loaded and ready
    manager.ensure_model("qwen3-vl:32b")

    # Use the client (serving is guaranteed running)
    client = manager.get_client()
    result = client.image_to_text(image_path, prompt, model="qwen3-vl:32b")

    # Clean shutdown
    manager.close()

"""

import json
import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend enumeration
# ---------------------------------------------------------------------------

class ServerBackend(Enum):
    """Model serving backend.

    Attributes:
        MPLTX: Native MPLTX serving (preferred).
        OLLAMA: Ollama with auto-start (backward compatible).
    """

    MPLTX = auto()
    """Native MPLTX model serving."""

    OLLAMA = auto()
    """Ollama with automatic start/stop."""


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    """Current state of the model server.

    Attributes:
        backend: Serving backend in use.
        is_running: Whether the server process is alive.
        base_url: HTTP URL for API requests.
        pid: Process ID (0 if not managed by us).
        models_loaded: List of currently loaded model names.
        memory_usage_mb: Approximate memory usage (0 if unknown).
        gpu_usage_percent: GPU utilization (0 if no GPU).
        cold_start_seconds: Time since last cold start (0 if warm).
        uptime_seconds: How long the server has been running.
    """

    backend: ServerBackend
    is_running: bool = False
    base_url: str = ""
    pid: int = 0
    models_loaded: list[str] = field(default_factory=list)
    memory_usage_mb: int = 0
    gpu_usage_percent: float = 0.0
    cold_start_seconds: float = 0.0
    uptime_seconds: float = 0.0
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract server interface
# ---------------------------------------------------------------------------

class BaseModelServer(ABC):
    """Abstract interface for model serving backends.

    All concrete backends (MPLTX, Ollama, etc.) implement this interface.
    The ModelSelector uses this abstraction to interact with any backend.
    """

    @property
    @abstractmethod
    def backend(self) -> ServerBackend:
        """Return the backend enum value."""

    @abstractmethod
    def start(self) -> bool:
        """Start the model server.

        Returns:
            True if server started successfully, False otherwise.
        """

    @abstractmethod
    def stop(self) -> bool:
        """Stop the model server gracefully.

        Returns:
            True if server stopped successfully, False otherwise.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the server is healthy and responsive.

        Returns:
            True if server is healthy, False otherwise.
        """

    @abstractmethod
    def get_state(self) -> ServerState:
        """Get current server state.

        Returns:
            ServerState with current status information.
        """

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available on this server.

        Args:
            model_name: Model name to check.

        Returns:
            True if model is available, False otherwise.
        """

    @abstractmethod
    def pull_model(self, model_name: str, **kwargs: Any) -> bool:
        """Pull/download a model from the registry.

        Args:
            model_name: Model name to pull.
            **kwargs: Backend-specific options (e.g., insecure flag).

        Returns:
            True if model pulled successfully, False otherwise.
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """List all models available on this server.

        Returns:
            List of model names.
        """

    @abstractmethod
    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage (memory, GPU, CPU).

        Returns:
            Dictionary with resource metrics.
        """

    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """Explicitly load a model into memory (warm start).

        Some servers unload idle models to free resources. This forces
        a model to stay loaded.

        Args:
            model_name: Model to load.

        Returns:
            True if model loaded successfully, False otherwise.
        """

    @abstractmethod
    def unload_model(self, model_name: str) -> bool:
        """Explicitly unload a model from memory.

        Args:
            model_name: Model to unload.

        Returns:
            True if model unloaded successfully, False otherwise.
        """

    @abstractmethod
    def get_client(self) -> Any:  # noqa: ANN401
        """Get an API client for making requests to this server.

        Returns:
            API client instance (e.g., OllamaClient, MpltxClient).
        """


# ---------------------------------------------------------------------------
# MPLTX server implementation
# ---------------------------------------------------------------------------

class MpltxServer(BaseModelServer):
    """MPLTX model serving backend.

    Manages a local MPLTX server process: start, stop, health-check,
    model pull, resource monitoring, and cold-start management.

    Configuration (environment variables):
        MPLTX_HOME: Path to MPLTX data directory (default: ~/.mpltx)
        MPLTX_PORT: Port for the HTTP API (default: 11435)
        MPLTX_MODEL: Default model to load on startup
        MPLTX_GPU_LAYERS: Number of layers to offload to GPU (default: -1 = all)
        MPLTX_CONTEXT_SIZE: Context window size (default: 4096)

    Attributes:
        home_dir: Path to MPLTX data directory.
        port: HTTP API port.
        model: Default model to load.
        gpu_layers: GPU offload configuration.
        context_size: Context window size.
        process: Subprocess handle (None if not started).
        start_time: When the server was started (0 if not started).
    """

    def __init__(
        self,
        home_dir: Optional[str] = None,
        port: int = 11435,
        model: Optional[str] = None,
        gpu_layers: int = -1,
        context_size: int = 4096,
        keep_alive: str = "-1",  # Never auto-unload
    ):
        """Initialize MPLTX server manager.

        Args:
            home_dir: Path to MPLTX data directory.
            port: HTTP API port.
            model: Default model to load on startup.
            gpu_layers: Number of layers to offload to GPU (-1 = all).
            context_size: Context window size.
            keep_alive: How long to keep model loaded ("0" = unload after use,
                       "-1" = never unload, "5m" = 5 minutes).
        """
        self.home_dir = Path(home_dir) if home_dir else Path.home() / ".mpltx"
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.model = model
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.keep_alive = keep_alive
        self.process: Optional[subprocess.Popen] = None
        self.start_time: float = 0.0
        self._env: dict[str, str] = self._build_env()

    @property
    def backend(self) -> ServerBackend:
        """Return the backend enum value."""
        return ServerBackend.MPLTX

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for the MPLTX process.

        Returns:
            Dictionary of environment variables.
        """
        env = os.environ.copy()
        env["HOME"] = str(self.home_dir)
        env.setdefault("MPLTX_HOME", str(self.home_dir))
        env.setdefault("MPLTX_PORT", str(self.port))
        return env

    def _build_args(self) -> list[str]:
        """Build command-line arguments for the MPLTX process.

        Returns:
            List of command-line arguments.
        """
        args = ["mpltx"]

        # Check if mpltx binary is available
        mpltx_path = self._find_mpltx_binary()
        if mpltx_path:
            args = [mpltx_path]
        else:
            # If binary not found, assume it's in PATH
            args = ["mpltx"]

        args.extend([
            "--port", str(self.port),
            "--ctx-size", str(self.context_size),
            "--gpu-layers", str(self.gpu_layers),
            "--keep-alive", self.keep_alive,
        ])

        if self.model:
            args.extend(["--model", self.model])

        return args

    def _find_mpltx_binary(self) -> Optional[str]:
        """Find the MPLTX binary on the system.

        Searches:
        1. MPLTX_HOME/bin/
        2. PATH

        Returns:
            Path to binary, or None if not found.
        """
        # Check MPLTX_HOME/bin first
        bin_path = self.home_dir / "bin" / "mpltx"
        if bin_path.exists() and bin_path.is_file():
            return str(bin_path)

        # Check PATH
        try:
            result = subprocess.run(
                ["which", "mpltx"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def start(self) -> bool:
        """Start the MPLTX server process.

        Attempts to start the server and waits for it to become healthy.
        If the server is already running, returns True without starting.

        Returns:
            True if server is running and healthy, False otherwise.
        """
        if self.is_running:
            return self.health_check()

        logger.info("Starting MPLTX server on port %d...", self.port)

        try:
            args = self._build_args()
            logger.debug("MPLTX command: %s", " ".join(args))

            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                start_new_session=True,  # Separate process group
            )

            self.start_time = time.time()
            logger.info("MPLTX server started (PID: %d)", self.process.pid)

            # Wait for server to become healthy (up to 60 seconds)
            if self._wait_for_healthy(timeout=60):
                logger.info("MPLTX server is healthy")
                return True
            else:
                logger.error("MPLTX server failed to become healthy within 60 seconds")
                self.stop()
                return False

        except FileNotFoundError:
            logger.error("MPLTX binary not found. Install MPLTX or set MPLTX_HOME.")
            return False
        except Exception as e:
            logger.exception("Failed to start MPLTX server: %s", e)
            return False

    def stop(self) -> bool:
        """Stop the MPLTX server gracefully.

        Sends SIGTERM and waits up to 10 seconds for clean shutdown.
        If the process doesn't exit, sends SIGKILL.

        Returns:
            True if server stopped successfully, False otherwise.
        """
        if self.process is None:
            return True

        logger.info("Stopping MPLTX server (PID: %d)...", self.process.pid)

        try:
            # Send SIGTERM for graceful shutdown
            if hasattr(signal, "SIGTERM"):
                os.kill(self.process.pid, signal.SIGTERM)

            # Wait up to 10 seconds for clean exit
            self.process.wait(timeout=10)

            self.process = None
            self.start_time = 0.0
            logger.info("MPLTX server stopped")
            return True

        except subprocess.TimeoutExpired:
            logger.warning("MPLTX server did not stop gracefully, sending SIGKILL")
            try:
                os.kill(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except Exception:
                pass

            self.process = None
            self.start_time = 0.0
            return True

        except Exception as e:
            logger.exception("Error stopping MPLTX server: %s", e)
            return False

    def health_check(self) -> bool:
        """Check if the server is healthy and responsive.

        Returns:
            True if server is healthy, False otherwise.
        """
        if self.process is None or self.process.poll() is not None:
            return False

        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_healthy(self, timeout: int = 60) -> bool:
        """Wait for the server to become healthy.

        Polls the health endpoint every 2 seconds until healthy or timeout.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if server became healthy within timeout.
        """
        import requests

        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=5,
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)

        return False

    def get_state(self) -> ServerState:
        """Get current server state.

        Returns:
            ServerState with current status information.
        """
        state = ServerState(
            backend=self.backend,
            is_running=self.process is not None and self.process.poll() is None,
            base_url=self.base_url,
            pid=self.process.pid if self.process else 0,
        )

        if state.is_running:
            try:
                import requests

                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    state.models_loaded = [m["name"] for m in data.get("models", [])]
            except Exception:
                pass

            resource = self.get_resource_usage()
            state.memory_usage_mb = int(resource.get("memory_mb", 0))
            state.gpu_usage_percent = resource.get("gpu_percent", 0.0)

            if self.start_time > 0:
                state.uptime_seconds = time.time() - self.start_time

        return state

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available on this server.

        Args:
            model_name: Model name to check.

        Returns:
            True if model is available, False otherwise.
        """
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name") == model_name for m in models)
        except Exception:
            pass

        return False

    def pull_model(self, model_name: str, **kwargs: Any) -> bool:
        """Pull/download a model from the registry.

        Args:
            model_name: Model name to pull.
            **kwargs: Backend-specific options.

        Returns:
            True if model pulled successfully, False otherwise.
        """
        try:
            import requests

            payload = {
                "name": model_name,
                "stream": False,
            }

            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300,  # Model pulls can take a while
            )
            return response.status_code in (200, 201)

        except Exception as e:
            logger.exception("Failed to pull model %s: %s", model_name, e)
            return False

    def list_models(self) -> list[str]:
        """List all models available on this server.

        Returns:
            List of model names.
        """
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            pass

        return []

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage (memory, GPU, CPU).

        Returns:
            Dictionary with resource metrics.
        """
        usage: dict[str, float] = {"memory_mb": 0.0, "gpu_percent": 0.0, "cpu_percent": 0.0}

        if self.process is None or self.process.poll() is not None:
            return usage

        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/usage",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                usage["memory_mb"] = float(data.get("memory_used", 0) / (1024 * 1024))
                usage["gpu_percent"] = float(data.get("gpu_utilization", 0))
                usage["cpu_percent"] = float(data.get("cpu_usage", 0))
        except Exception:
            # If /api/usage doesn't exist, try to get from process stats
            try:
                import psutil

                proc = psutil.Process(self.process.pid)
                mem_info = proc.memory_info()
                usage["memory_mb"] = float(mem_info.rss) / (1024 * 1024)
                usage["cpu_percent"] = proc.cpu_percent(interval=0.1)
            except Exception:
                pass

        return usage

    def load_model(self, model_name: str) -> bool:
        """Explicitly load a model into memory (warm start).

        Sends a dummy request to trigger model loading, keeping it
        resident for subsequent requests.

        Args:
            model_name: Model to load.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            import requests

            # Send a minimal request to trigger model loading
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                    },
                },
                timeout=120,  # Cold start can take a while
            )
            return response.status_code in (200, 201)

        except Exception as e:
            logger.warning("Failed to pre-load model %s: %s", model_name, e)
            return False

    def unload_model(self, model_name: str) -> bool:
        """Explicitly unload a model from memory.

        Args:
            model_name: Model to unload.

        Returns:
            True if model unloaded successfully, False otherwise.
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/gc",
                json={"models": [model_name]},
                timeout=10,
            )
            return response.status_code == 200

        except Exception:
            return False

    def get_client(self) -> Any:  # noqa: ANN401
        """Get an API client for making requests to this server.

        Returns:
            OllamaClient configured to talk to this server.
            (MPLTX uses the same API as Ollama.)
        """
        from lib.ollama_client import OllamaClient

        return OllamaClient(
            base_url=self.base_url,
            timeout=120,  # Longer timeout for model loading
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Ollama server with auto-start (backward compatible)
# ---------------------------------------------------------------------------

class OllamaServer(BaseModelServer):
    """Ollama server with automatic start/stop.

    Extends the current OllamaClient with lifecycle management.
    If Ollama is not running, automatically starts it.

    This is the backward-compatible path — it keeps the existing
    OllamaClient API but adds process management.

    Attributes:
        base_url: Ollama server URL.
        timeout: Request timeout in seconds.
        model: Default model.
        auto_start: Whether to start Ollama if not running.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        model: Optional[str] = None,
        auto_start: bool = True,
    ):
        """Initialize Ollama server manager.

        Args:
            base_url: Ollama server URL.
            timeout: Request timeout in seconds.
            model: Default model.
            auto_start: Whether to start Ollama if not running.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.auto_start = auto_start
        self.client = None  # OllamaClient instance (created on demand)

    @property
    def backend(self) -> ServerBackend:
        """Return the backend enum value."""
        return ServerBackend.OLLAMA

    def start(self) -> bool:
        """Start the Ollama server if not already running.

        Tries to start Ollama via `ollama serve` if it's not already running.

        Returns:
            True if server is running, False otherwise.
        """
        if not self.auto_start:
            return self.health_check()

        if self.health_check():
            return True

        logger.info("Ollama not running, attempting to start...")

        try:
            # Try to start Ollama in the background
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            if self._wait_for_healthy(timeout=30):
                logger.info("Ollama started successfully")
                return True
            else:
                logger.error("Failed to start Ollama")
                return False

        except FileNotFoundError:
            logger.error(
                "Ollama not found. Install Ollama or start it manually, "
                "then set BOURBON_OLLAMA_URL to its address."
            )
            return False

    def stop(self) -> bool:
        """Stop the Ollama server.

        Returns:
            True if server stopped, False otherwise.
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/close",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_healthy(self, timeout: int = 30) -> bool:
        """Wait for the server to become healthy.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if server became healthy.
        """
        import requests

        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=5,
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)

        return False

    def get_state(self) -> ServerState:
        """Get current server state.

        Returns:
            ServerState with current status.
        """
        state = ServerState(
            backend=self.backend,
            is_running=self.health_check(),
            base_url=self.base_url,
        )

        if state.is_running:
            try:
                import requests

                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    state.models_loaded = [m["name"] for m in data.get("models", [])]
            except Exception:
                pass

        return state

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available.

        Args:
            model_name: Model name to check.

        Returns:
            True if model is available.
        """
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model_name in m.get("name", "") for m in models)
        except Exception:
            pass

        return False

    def pull_model(self, model_name: str, **kwargs: Any) -> bool:
        """Pull a model from the registry.

        Args:
            model_name: Model name to pull.
            **kwargs: Backend-specific options.

        Returns:
            True if model pulled successfully.
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300,
            )
            return response.status_code in (200, 201)
        except Exception as e:
            logger.exception("Failed to pull model %s: %s", model_name, e)
            return False

    def list_models(self) -> list[str]:
        """List all available models.

        Returns:
            List of model names.
        """
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            pass

        return []

    def get_resource_usage(self) -> dict[str, float]:
        """Get resource usage.

        Returns:
            Dictionary with resource metrics.
        """
        return {"memory_mb": 0.0, "gpu_percent": 0.0, "cpu_percent": 0.0}

    def load_model(self, model_name: str) -> bool:
        """Pre-load a model into memory.

        Args:
            model_name: Model to load.

        Returns:
            True if model loaded successfully.
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                },
                timeout=120,
            )
            return response.status_code in (200, 201)
        except Exception:
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.

        Args:
            model_name: Model to unload.

        Returns:
            True if model unloaded successfully.
        """
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/gc",
                json={"models": [model_name]},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_client(self) -> Any:  # noqa: ANN401
        """Get an API client.

        Returns:
            OllamaClient configured for this server.
        """
        from lib.ollama_client import OllamaClient

        return OllamaClient(
            base_url=self.base_url,
            timeout=self.timeout,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Model server manager (facade)
# ---------------------------------------------------------------------------

class ModelServerManager:
    """Manages model serving lifecycle for the pipeline.

    High-level facade that wraps a concrete backend (MPLTX or Ollama)
    and provides convenient methods for the pipeline stages:

    1. Ensure the server is running (start if needed)
    2. Ensure the required model is pulled and loaded
    3. Provide an API client for requests
    4. Track resource usage and model state
    5. Clean shutdown on exit

    Usage::

        # Create manager with MPLTX backend (preferred)
        manager = ModelServerManager(backend=ServerBackend.MPLTX)

        # Ensure all required models are available
        manager.ensure_models(["qwen3-vl:32b", "gpt-oss:120b", "qwen2-vl:7b"])

        # Get a client (server is guaranteed running)
        client = manager.get_client()
        result = client.image_to_text(image_path, prompt)

        # Check resource usage
        state = manager.get_state()
        print(f"Memory: {state.memory_usage_mb}MB")

        # Clean shutdown
        manager.close()

    Attributes:
        backend: The serving backend.
        server: The concrete server implementation.
    """

    def __init__(
        self,
        backend: ServerBackend = ServerBackend.MPLTX,
        config: Any | None = None,  # noqa: ANN401
        **kwargs: Any,
    ):
        """Initialize the model server manager.

        Args:
            backend: Serving backend (MPLTX preferred, Ollama fallback).
            config: Optional Config object for configuration.
            **kwargs: Backend-specific initialization parameters.
        """
        self._backend = backend
        self._server: BaseModelServer | None = None
        self._initialized = False

        if config is not None:
            self._init_from_config(config, **kwargs)
        else:
            self._init_default(**kwargs)

    def _init_default(self, **kwargs: Any) -> None:
        """Initialize with default settings."""
        if self._backend == ServerBackend.MPLTX:
            self._server = MpltxServer(**kwargs)
        elif self._backend == ServerBackend.OLLAMA:
            self._server = OllamaServer(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _init_from_config(self, config: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize from a Config object."""
        if self._backend == ServerBackend.MPLTX:
            # Check for MPLTX-specific config
            home_dir = getattr(config, "mpltx_home", None)
            port = getattr(config, "mpltx_port", 11435)
            model = getattr(config, "model", None)
            gpu_layers = getattr(config, "mpltx_gpu_layers", -1)
            context_size = getattr(config, "mpltx_context_size", 4096)

            self._server = MpltxServer(
                home_dir=home_dir,
                port=port,
                model=model,
                gpu_layers=gpu_layers,
                context_size=context_size,
                **kwargs,
            )
        elif self._backend == ServerBackend.OLLAMA:
            base_url = getattr(config, "ollama_url", "http://localhost:11434")
            model = getattr(config, "model", None)
            timeout = getattr(config, "timeout", 30)

            self._server = OllamaServer(
                base_url=base_url,
                timeout=timeout,
                model=model,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def server(self) -> BaseModelServer:
        """Get the underlying server implementation.

        Returns:
            The concrete server (MpltxServer or OllamaServer).
        """
        if self._server is None:
            raise RuntimeError("ModelServerManager not initialized")
        return self._server

    @property
    def state(self) -> ServerState:
        """Get current server state.

        Returns:
            ServerState with current status.
        """
        return self.server.get_state()

    def ensure_running(self) -> bool:
        """Ensure the server is running.

        Starts the server if not already running.

        Returns:
            True if server is running, False otherwise.
        """
        if not self.server.is_running:
            return self.server.start()
        return self.server.health_check()

    def ensure_model(self, model_name: str) -> bool:
        """Ensure a model is available and loaded.

        Checks if the model is available, pulls it if not,
        and loads it into memory.

        Args:
            model_name: Model name to ensure.

        Returns:
            True if model is available and loaded, False otherwise.
        """
        if not self.ensure_running():
            return False

        if not self.server.is_model_available(model_name):
            logger.info("Model '%s' not found, pulling...", model_name)
            if not self.server.pull_model(model_name):
                logger.error("Failed to pull model '%s'", model_name)
                return False
            logger.info("Model '%s' pulled successfully", model_name)

        # Warm-load the model
        if not self.server.load_model(model_name):
            logger.warning("Failed to pre-load model '%s', will load on first request", model_name)

        return True

    def ensure_models(self, model_names: list[str]) -> bool:
        """Ensure multiple models are available and loaded.

        Args:
            model_names: List of model names to ensure.

        Returns:
            True if all models are available, False otherwise.
        """
        all_ok = True
        for name in model_names:
            if not self.ensure_model(name):
                all_ok = False
                logger.warning("Model '%s' is not available", name)

        return all_ok

    def get_client(self) -> Any:  # noqa: ANN401
        """Get an API client for making requests.

        Ensures the server is running before returning the client.

        Returns:
            API client instance.

        Raises:
            RuntimeError: If server is not running.
        """
        if not self.ensure_running():
            raise RuntimeError(
                "Model server is not running. "
                "Start the server with manager.ensure_running() or "
                "ensure a model is loaded with manager.ensure_model()."
            )
        return self.server.get_client()

    def get_state(self) -> ServerState:
        """Get current server state.

        Returns:
            ServerState with current status.
        """
        return self.state

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage.

        Returns:
            Dictionary with resource metrics.
        """
        return self.server.get_resource_usage()

    def list_models(self) -> list[str]:
        """List all available models.

        Returns:
            List of model names.
        """
        return self.server.list_models()

    def close(self) -> None:
        """Clean shutdown: stop the server and release resources.

        Calls the backend's stop() method if auto-start was used.
        """
        if self._server is not None:
            try:
                self.server.stop()
            except Exception as e:
                logger.warning("Error during server shutdown: %s", e)
        self._initialized = False

    def __enter__(self) -> "ModelServerManager":
        """Context manager entry: ensure server is running."""
        self.ensure_running()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Context manager exit: clean shutdown."""
        self.close()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_server_manager(
    backend: ServerBackend = ServerBackend.MPLTX,
    **kwargs: Any,
) -> ModelServerManager:
    """Create a ModelServerManager with default settings.

    Args:
        backend: Serving backend.
        **kwargs: Backend-specific parameters.

    Returns:
        Configured ModelServerManager.
    """
    return ModelServerManager(backend=backend, **kwargs)


def create_server_manager_from_config(
    config: Any,  # noqa: ANN401
    backend: ServerBackend = ServerBackend.MPLTX,
    **kwargs: Any,
) -> ModelServerManager:
    """Create a ModelServerManager from a Config object.

    Args:
        config: Config object.
        backend: Serving backend.
        **kwargs: Backend-specific parameters.

    Returns:
        Configured ModelServerManager.
    """
    return ModelServerManager(backend=backend, config=config, **kwargs)