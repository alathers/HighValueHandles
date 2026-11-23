"""Ollama REST API client wrapper.

Provides a Pythonic interface to Ollama's local LLM server, including the
multi-image vision call used for label extraction (``chat_with_images``).
"""

import base64
import io
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image as PILImage
from PIL import ImageOps

from lib.exceptions import (
    OllamaConnectionError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionTimeoutError,
)

try:  # Optional HEIC/HEIF support (iPhone photos); the scanner accepts .heic.
    from pillow_heif import register_heif_opener  # type: ignore[import-not-found]

    register_heif_opener()
except ImportError:  # pragma: no cover - depends on the environment
    pass

logger = logging.getLogger(__name__)

#: Read timeout (seconds) for vision/generation calls: covers a cold model load.
DEFAULT_GENERATE_TIMEOUT = 600.0
#: Connect timeout (seconds) for vision/generation calls.
DEFAULT_CONNECT_TIMEOUT = 10.0
#: Default long edge (pixels) images are downscaled to before upload.
DEFAULT_MAX_IMAGE_EDGE = 1280
#: Ollama error texts meaning the model itself cannot be used on this server.
_MODEL_UNUSABLE_MARKERS = ("not found", "no longer compatible", "does not support")


@dataclass
class VisionChatResult:
    """Result of one ``chat_with_images`` call.

    Attributes:
        content: The assistant message text (the model's answer).
        thinking: Reasoning text, if the model produced any.
        model: Model that answered.
        done_reason: Why generation stopped (``'stop'``, ``'length'``, ...).
        prompt_eval_count: Prompt tokens processed (images included).
        eval_count: Output tokens generated.
        load_duration_s: Time spent loading the model, in seconds.
        total_duration_s: Total server-side time, in seconds.
    """

    content: str
    thinking: str = ""
    model: str = ""
    done_reason: Optional[str] = None
    prompt_eval_count: int = 0
    eval_count: int = 0
    load_duration_s: float = 0.0
    total_duration_s: float = 0.0


def prepare_image_b64(path: Path, max_edge: int = DEFAULT_MAX_IMAGE_EDGE, quality: int = 85) -> str:
    """Load, orient, downscale and JPEG-encode an image as base64.

    EXIF orientation is applied, the image is converted to RGB and its long
    edge is limited to ``max_edge`` pixels (aspect ratio kept). Full-size phone
    photos cost tens of thousands of prompt tokens without improving the
    reading, so callers should keep the default.

    Args:
        path: Image file path.
        max_edge: Maximum long edge in pixels; 0 or less keeps the original size.
        quality: JPEG quality (1-95).

    Returns:
        Base64-encoded JPEG bytes as an ASCII string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        with PILImage.open(path) as img:
            image = ImageOps.exif_transpose(img)
            image = image.convert("RGB")
            if max_edge and max_edge > 0:
                image.thumbnail((max_edge, max_edge), PILImage.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
    except (OSError, ValueError, SyntaxError) as e:
        # PIL raises UnidentifiedImageError (an OSError) for non-images and
        # OSError/SyntaxError/ValueError for truncated or corrupt files.
        raise ValueError(f"Invalid image file {path}: {e}") from e
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class ChatCompletionResponse:
    """Response from Ollama chat completion."""

    message: str
    model: str
    done: bool = True
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


@dataclass
class EmbeddingResponse:
    """Response from Ollama embedding."""

    embedding: List[float]
    model: str


class OllamaClient:
    """Wrapper for Ollama REST API.

    Provides methods for chat completion, image-to-text with vision models,
    and embeddings. Designed for local LLM hosting via Ollama.

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434)
        timeout: Request timeout in seconds for metadata endpoints (default: 30)
        model: Default model to use for requests
        generate_timeout: Read timeout in seconds for vision calls (default: 600),
            long enough for a cold model load
        connect_timeout: Connect timeout in seconds for vision calls (default: 10)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        model: Optional[str] = None,
        generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL.
            timeout: Request timeout in seconds (metadata endpoints).
            model: Default model to use.
            generate_timeout: Read timeout in seconds for ``chat_with_images``.
            connect_timeout: Connect timeout in seconds for ``chat_with_images``.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.generate_timeout = generate_timeout
        self.connect_timeout = connect_timeout
        self.session = requests.Session()
        self._capabilities: Dict[str, set[str]] = {}

    @staticmethod
    def _error_text(response: Any) -> str:  # noqa: ANN401
        """Ollama's error message from a response body (JSON ``error`` or raw text).

        Args:
            response: HTTP response.

        Returns:
            The error text, at most 500 characters ('' if none).
        """
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])[:500]
        text = getattr(response, "text", "")
        return text[:500] if isinstance(text, str) else ""

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint path (without base URL).
            method: HTTP method (GET, POST, etc.).
            data: Form data.
            files: File uploads.
            json_data: JSON data.
            timeout: Request timeout (default: ``self.timeout``).

        Returns:
            JSON response.

        Raises:
            requests.HTTPError: For HTTP status >= 400; the message keeps
                Ollama's own error text (e.g. "model 'x' not found").
            requests.RequestException: If the request fails otherwise.
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug("Ollama API request: %s %s", method, endpoint)

        response = self.session.request(
            method,
            url,
            data=data,
            files=files,
            json=json_data,
            timeout=self.timeout if timeout is None else timeout,
        )
        status = response.status_code
        if isinstance(status, int) and status >= 400:
            raise requests.HTTPError(f"HTTP {status}: {self._error_text(response)}", response=response)
        response.raise_for_status()

        logger.debug("Ollama API response: %s %s - Status: %s", method, endpoint, response.status_code)
        return response.json()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[ChatCompletionResponse]:
        """Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model name (uses instance default if None).
            stream: Whether to stream responses.
            format: Response format ('json' for JSON output).
            options: Generation options (temperature, top_k, etc.).

        Returns:
            List of ChatCompletionResponse objects (multiple if streaming).

        Raises:
            requests.RequestException: If request fails.
            ValueError: If response format invalid.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if format:
            payload["format"] = format
        if options:
            payload["options"] = options

        response = self._make_request("/api/chat", json_data=payload)

        if stream:
            responses = []
            for line in response.splitlines():
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("done", True) or chunk.get("message"):
                        responses.append(
                            ChatCompletionResponse(
                                message=chunk.get("message", {}).get("content", ""),
                                model=chunk.get("model", model),
                                done=chunk.get("done", True),
                                prompt_eval_count=chunk.get("prompt_eval_count", 0),
                                prompt_eval_duration=chunk.get("prompt_eval_duration", 0),
                                eval_count=chunk.get("eval_count", 0),
                                eval_duration=chunk.get("eval_duration", 0),
                            )
                        )
            return responses
        else:
            return [
                ChatCompletionResponse(
                    message=response.get("message", {}).get("content", ""),
                    model=response.get("model", model),
                    done=response.get("done", True),
                    prompt_eval_count=response.get("prompt_eval_count", 0),
                    prompt_eval_duration=response.get("prompt_eval_duration", 0),
                    eval_count=response.get("eval_count", 0),
                    eval_duration=response.get("eval_duration", 0),
                )
            ]

    def image_to_text(
        self,
        image_path: Path,
        prompt: str,
        model: Optional[str] = None,
        detail: str = "high",
        format: Optional[str] = None,
    ) -> str:
        """Extract text from image using vision model (legacy single-image call).

        Kept for compatibility; the bottle analyzer uses ``chat_with_images``,
        which sends several downscaled images, a JSON schema and ``think``.

        Args:
            image_path: Path to image file.
            prompt: Text prompt for the model.
            model: Vision model name (uses instance default if None).
            detail: Image detail level ('low', 'medium', 'high').
            format: Response format ('json' for structured output; default none).

        Returns:
            Extracted text as string.

        Raises:
            requests.RequestException: If request fails.
            ValueError: If image file invalid or response format invalid.
            FileNotFoundError: If image file not found.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and validate image
        try:
            with PILImage.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        # Prepare request
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [self._encode_image(image_path)],
            "stream": False,
        }

        if format:
            payload["format"] = format

        response = self._make_request("/api/generate", json_data=payload)
        return response.get("response", "")

    def get_capabilities(self, model: str) -> set[str]:
        """Capabilities Ollama reports for an installed model (cached per model).

        Args:
            model: Installed model name.

        Returns:
            Lower-case capability names, e.g. ``{'completion', 'vision',
            'thinking'}``. Empty if this Ollama version does not report them.

        Raises:
            VisionModelUnavailableError: If the model is not installed (HTTP 404).
            requests.RequestException: If the request fails otherwise.
        """
        if model in self._capabilities:
            return self._capabilities[model]
        try:
            body = self._make_request("/api/show", json_data={"model": model})
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                raise VisionModelUnavailableError(
                    f"Model '{model}' is not installed on {self.base_url}"
                ) from e
            raise
        caps = body.get("capabilities") if isinstance(body, dict) else None
        result = {str(c).lower() for c in caps} if isinstance(caps, list) else set()
        self._capabilities[model] = result
        return result

    def supports_vision(self, model: str) -> bool:
        """Whether the model accepts images.

        Args:
            model: Installed model name.

        Returns:
            True if Ollama reports the ``vision`` capability.
        """
        return "vision" in self.get_capabilities(model)

    def supports_thinking(self, model: str) -> bool:
        """Whether the model accepts the ``think`` switch.

        Args:
            model: Installed model name.

        Returns:
            True if Ollama reports the ``thinking`` capability.
        """
        return "thinking" in self.get_capabilities(model)

    def chat_with_images(
        self,
        prompt: str,
        image_paths: Sequence[Path],
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        think: Optional[bool] = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[str, int]] = "30m",
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
        read_timeout: Optional[float] = None,
    ) -> VisionChatResult:
        """Ask a vision model about one or more images via ``POST /api/chat``.

        All images go into one user message, each orientation-corrected,
        downscaled to ``max_image_edge`` and JPEG-encoded. ``think`` is only sent
        when the model reports the ``thinking`` capability: thinking models
        otherwise spend the whole output budget reasoning and return no content.

        Args:
            prompt: User message text.
            image_paths: Images to attach (at least one).
            model: Model name (uses instance default if None).
            system: Optional system message.
            format: ``'json'`` or a JSON-schema dict constraining the output.
            think: Thinking switch; None leaves the model default.
            options: Generation options (temperature, num_ctx, num_predict, ...).
            keep_alive: How long Ollama keeps the model loaded afterwards.
            max_image_edge: Long-edge limit in pixels (0 keeps full size).
            read_timeout: Read timeout in seconds (default: ``generate_timeout``).

        Returns:
            VisionChatResult with the answer and token/timing statistics.

        Raises:
            ValueError: If no model is set, no image is given or an image is invalid.
            FileNotFoundError: If an image file does not exist.
            VisionModelUnavailableError: If the model is missing, incompatible
                with this Ollama, or cannot take images.
            OllamaConnectionError: If Ollama cannot be reached.
            VisionTimeoutError: If no response arrives within the read timeout.
            VisionAnalysisError: For any other HTTP or protocol error.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("At least one image is required")

        caps: Optional[set[str]] = None
        try:
            caps = self.get_capabilities(model)
        except VisionModelUnavailableError:
            raise
        except requests.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        except Exception as e:  # capabilities are advisory; the chat call decides
            logger.debug("Could not read capabilities of %s: %s", model, e)
        if caps and "vision" not in caps:
            raise VisionModelUnavailableError(
                f"Model '{model}' does not support image input (capabilities: {sorted(caps)})"
            )

        images = [prepare_image_b64(p, max_image_edge) for p in paths]
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images})

        payload: Dict[str, Any] = {"model": model, "stream": False, "messages": messages}
        if think is not None and caps and "thinking" in caps:
            payload["think"] = bool(think)
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = dict(options)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        read = read_timeout if read_timeout is not None else self.generate_timeout
        logger.debug("Ollama API request: POST /api/chat (%s, %d image(s))", model, len(images))
        try:
            response = self.session.request(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(self.connect_timeout, read),
            )
        except requests.ConnectTimeout as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        except requests.Timeout as e:
            raise VisionTimeoutError(
                f"Vision model '{model}' did not respond within {read:.0f}s "
                "(model load + generation); increase --request-timeout"
            ) from e
        except requests.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        except requests.RequestException as e:
            raise VisionAnalysisError(f"Ollama /api/chat request failed for {model}: {e}") from e

        status = response.status_code if isinstance(response.status_code, int) else 0
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        if status >= 400 or error:
            err_text = str(error)[:500] if error else self._error_text(response)
            if status == 404 or any(m in err_text.lower() for m in _MODEL_UNUSABLE_MARKERS):
                raise VisionModelUnavailableError(
                    f"Vision model '{model}' is not usable on {self.base_url}: {err_text}"
                )
            raise VisionAnalysisError(f"Ollama /api/chat failed (HTTP {status}) for {model}: {err_text}")
        if not isinstance(body, dict):
            raise VisionAnalysisError(f"Ollama /api/chat returned a non-JSON body for {model}")

        message = body.get("message") or {}
        return VisionChatResult(
            content=str(message.get("content") or ""),
            thinking=str(message.get("thinking") or ""),
            model=str(body.get("model") or model),
            done_reason=body.get("done_reason"),
            prompt_eval_count=int(body.get("prompt_eval_count") or 0),
            eval_count=int(body.get("eval_count") or 0),
            load_duration_s=(body.get("load_duration") or 0) / 1e9,
            total_duration_s=(body.get("total_duration") or 0) / 1e9,
        )

    def embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate text embedding.

        Args:
            text: Input text.
            model: Embedding model name (uses instance default if None).

        Returns:
            EmbeddingResponse with embedding vector and model name.

        Raises:
            requests.RequestException: If request fails.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")

        payload = {
            "model": model,
            "prompt": text,
        }

        response = self._make_request("/api/embeddings", json_data=payload)
        return EmbeddingResponse(
            embedding=response.get("embedding", []),
            model=response.get("model", model),
        )

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64 string.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded image string.

        Raises:
            ValueError: If image encoding fails.
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models.

        Returns:
            List of model info dicts with 'name', 'modified_at', 'size', etc.

        Raises:
            requests.RequestException: If request fails.
        """
        response = self._make_request("/api/tags", method="GET")
        return response.get("models", [])

    def check_model(self, model: Optional[str] = None) -> bool:
        """Check if model is available locally.

        Args:
            model: Model name to check (uses instance default if None).

        Returns:
            True if model available, False otherwise.
        """
        model = model or self.model
        if not model:
            return False

        try:
            models = self.list_models()
            model_names = [m.get("name", "") for m in models]
            return model in model_names
        except Exception:
            return False

    def is_model_available(self, model: Optional[str] = None) -> bool:
        """Check if model is available locally (alias for check_model).

        Args:
            model: Model name to check (uses instance default if None).

        Returns:
            True if model available, False otherwise.
        """
        return self.check_model(model)

    def health_check(self) -> bool:
        """Check if Ollama server is running and accessible.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for quick usage

def get_ollama_client(
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
    model: Optional[str] = None,
    generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> OllamaClient:
    """Create Ollama client instance.

    Args:
        base_url: Ollama server URL.
        timeout: Request timeout in seconds (metadata endpoints).
        model: Default model to use.
        generate_timeout: Read timeout in seconds for vision calls.
        connect_timeout: Connect timeout in seconds for vision calls.

    Returns:
        Configured OllamaClient instance.
    """
    return OllamaClient(
        base_url=base_url,
        timeout=timeout,
        model=model,
        generate_timeout=generate_timeout,
        connect_timeout=connect_timeout,
    )
