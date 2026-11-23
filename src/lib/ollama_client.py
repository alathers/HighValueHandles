"""Ollama REST API client wrapper.

Provides a Pythonic interface to Ollama's local LLM server: the multi-image
vision call used for label extraction (``chat_with_images``) and the
text-only call used by the fact-check stage (``chat_text``). Both share one
request path (``_post_chat``) and differ only in the capability they require
and the exception family they raise (``Vision*`` or ``Text*``).

The client is safe to share between threads: each thread gets its own
``requests.Session``.
"""

import base64
import io
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from PIL import Image as PILImage
from PIL import ImageOps
from requests.adapters import HTTPAdapter

from lib.exceptions import (
    AIModelError,
    OllamaConnectionError,
    TextAnalysisError,
    TextModelUnavailableError,
    TextTimeoutError,
    TextTransientError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionTimeoutError,
    VisionTransientError,
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
#: HTTP statuses after which repeating the same request may succeed.
_TRANSIENT_STATUSES = frozenset({408, 429})
#: Think levels Ollama accepts as strings (gpt-oss accepts only these).
THINK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
#: Connection pool size per session (matches the maximum ``--concurrency``).
_POOL_MAXSIZE = 10

#: Type of the ``think`` argument: a switch, a level, or None (model default).
ThinkSetting = Union[bool, Literal["low", "medium", "high"], None]


@dataclass
class ChatResult:
    """Result of one ``chat_with_images`` or ``chat_text`` call.

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


def is_gpt_oss_model(model: str) -> bool:
    """Whether ``model`` belongs to the gpt-oss family (accepts only think levels).

    The registry host and namespace are ignored, so ``gpt-oss:120b`` and
    ``registry.ollama.ai/library/gpt-oss:20b`` both match.
    """
    base = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return base.startswith("gpt-oss")


def resolve_think(think: ThinkSetting, model: str, caps: Optional[set[str]]) -> Union[bool, str, None]:
    """The ``think`` value to send for ``model``, or None to omit the field.

    - None, or a model that does not report the ``thinking`` capability
      (unknown capabilities included): omitted. Thinking models otherwise
      spend the whole output budget reasoning.
    - A level string (``'low'|'medium'|'high'``): sent as-is.
    - A bool for a gpt-oss model, which ignores bools: False maps to
      ``'low'`` (it cannot turn thinking off) and True to ``'medium'``.
    - Any other bool: sent unchanged.

    Args:
        think: Requested setting.
        model: Model name.
        caps: Capabilities the server reported (None or empty if unknown).

    Returns:
        The payload value, or None to leave ``think`` out.

    Raises:
        ValueError: If ``think`` is a string that is not a known level.
    """
    if isinstance(think, str):
        if think not in THINK_LEVELS:
            raise ValueError(f"think must be a bool or one of {THINK_LEVELS}, got {think!r}")
    elif think is not None and not isinstance(think, bool):
        raise ValueError(f"think must be a bool or one of {THINK_LEVELS}, got {think!r}")
    if think is None or not caps or "thinking" not in caps:
        return None
    if isinstance(think, str):
        return think
    if is_gpt_oss_model(model):
        return "medium" if think else "low"
    return bool(think)


@dataclass(frozen=True)
class _ErrorSet:
    """Exception classes one kind of chat call raises (vision or text)."""

    kind: str
    unavailable: type[AIModelError]
    analysis: type[AIModelError]
    timeout: type[AIModelError]
    transient: type[AIModelError]
    capability: str
    timeout_hint: str


_VISION_ERRORS = _ErrorSet(
    kind="Vision",
    unavailable=VisionModelUnavailableError,
    analysis=VisionAnalysisError,
    timeout=VisionTimeoutError,
    transient=VisionTransientError,
    capability="vision",
    timeout_hint="increase --request-timeout",
)
_TEXT_ERRORS = _ErrorSet(
    kind="Text",
    unavailable=TextModelUnavailableError,
    analysis=TextAnalysisError,
    timeout=TextTimeoutError,
    transient=TextTransientError,
    capability="completion",
    timeout_hint="increase --text-timeout",
)


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


class OllamaClient:
    """Wrapper for the parts of the Ollama REST API the pipeline uses.

    ``get_capabilities`` (``POST /api/show``), the multi-image vision call
    ``chat_with_images`` and the text-only call ``chat_text`` (both
    ``POST /api/chat``). Model management (pulls, listing, health, loading
    and unloading) lives in :mod:`lib.model_server` /
    :mod:`lib.model_availability`.

    Each thread uses its own ``requests.Session`` (see :attr:`session`), so
    one client can serve parallel OCR/vision workers; :meth:`close` closes
    every session the client created.

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434)
        timeout: Request timeout in seconds for metadata endpoints (default: 30)
        model: Default model to use for requests
        generate_timeout: Read timeout in seconds for chat calls (default: 600),
            long enough for a cold model load
        connect_timeout: Connect timeout in seconds for chat calls (default: 10)
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
            generate_timeout: Read timeout in seconds for chat calls.
            connect_timeout: Connect timeout in seconds for chat calls.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.generate_timeout = generate_timeout
        self.connect_timeout = connect_timeout
        self._local = threading.local()
        self._sessions: List[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self._capabilities: Dict[str, set[str]] = {}
        self._capabilities_lock = threading.Lock()
        _ = self.session  # the constructing thread's session exists from the start

    # -- sessions ----------------------------------------------------------

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_maxsize=_POOL_MAXSIZE)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _track(self, session: Any) -> None:  # noqa: ANN401
        with self._sessions_lock:
            if not any(s is session for s in self._sessions):
                self._sessions.append(session)

    @property
    def session(self) -> requests.Session:
        """The calling thread's HTTP session (created on first use)."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._new_session()
            self._local.session = session
            self._track(session)
        return session

    @session.setter
    def session(self, value: Any) -> None:  # noqa: ANN401
        """Replace the calling thread's session (tests inject fakes this way)."""
        self._local.session = value
        if value is not None:
            self._track(value)

    # -- metadata ----------------------------------------------------------

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
        with self._capabilities_lock:
            cached = self._capabilities.get(model)
        if cached is not None:
            return cached
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
        with self._capabilities_lock:
            self._capabilities[model] = result
        return result

    def _required_capabilities(self, model: str, errors: _ErrorSet) -> Optional[set[str]]:
        """Read ``model``'s capabilities and require ``errors.capability``.

        A failed capability read is advisory (the chat call decides), except
        that a missing model or an unreachable server raises right away.

        Returns:
            The capabilities, or None if they could not be read.

        Raises:
            errors.unavailable: If the model is missing or lacks the capability.
            OllamaConnectionError: If Ollama cannot be reached.
        """
        caps: Optional[set[str]] = None
        try:
            caps = self.get_capabilities(model)
        except VisionModelUnavailableError as e:
            if errors.unavailable is VisionModelUnavailableError:
                raise
            raise errors.unavailable(str(e)) from e
        except requests.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        except Exception as e:  # capabilities are advisory; the chat call decides
            logger.debug("Could not read capabilities of %s: %s", model, e)
        if caps and errors.capability not in caps:
            what = "image input" if errors.capability == "vision" else f"'{errors.capability}'"
            raise errors.unavailable(
                f"Model '{model}' does not support {what} (capabilities: {sorted(caps)})"
            )
        return caps

    # -- chat --------------------------------------------------------------

    @staticmethod
    def _build_payload(
        model: str,
        messages: List[Dict[str, Any]],
        caps: Optional[set[str]],
        *,
        format: Optional[Union[str, Dict[str, Any]]],
        think: ThinkSetting,
        options: Optional[Dict[str, Any]],
        keep_alive: Optional[Union[str, int]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "stream": False, "messages": messages}
        think_value = resolve_think(think, model, caps)
        if think_value is not None:
            payload["think"] = think_value
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = dict(options)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return payload

    def _post_chat(
        self, payload: Dict[str, Any], model: str, read: float, *, errors: _ErrorSet
    ) -> ChatResult:
        """Send one ``POST /api/chat`` and map every failure to ``errors``.

        - connect failure or connect timeout: :class:`OllamaConnectionError`;
        - read timeout: ``errors.timeout``;
        - HTTP 404 or an "unusable model" message: ``errors.unavailable``;
        - HTTP 5xx, 408, 429 or another request failure: ``errors.transient``;
        - any other HTTP error, an ``error`` in the body or a non-JSON body:
          ``errors.analysis``.

        Args:
            payload: Request body.
            model: Model name (for messages).
            read: Read timeout in seconds.
            errors: Exception family to raise.

        Returns:
            The parsed answer and its statistics.
        """
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
            raise errors.timeout(
                f"{errors.kind} model '{model}' did not respond within {read:.0f}s "
                f"(model load + generation); {errors.timeout_hint}"
            ) from e
        except requests.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e
        except requests.RequestException as e:
            raise errors.transient(f"Ollama /api/chat request failed for {model}: {e}") from e

        status = response.status_code if isinstance(response.status_code, int) else 0
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        if status >= 400 or error:
            err_text = str(error)[:500] if error else self._error_text(response)
            if status == 404 or any(m in err_text.lower() for m in _MODEL_UNUSABLE_MARKERS):
                raise errors.unavailable(
                    f"{errors.kind} model '{model}' is not usable on {self.base_url}: {err_text}"
                )
            message = f"Ollama /api/chat failed (HTTP {status}) for {model}: {err_text}"
            if status >= 500 or status in _TRANSIENT_STATUSES:
                raise errors.transient(message)
            raise errors.analysis(message)
        if not isinstance(body, dict):
            raise errors.analysis(f"Ollama /api/chat returned a non-JSON body for {model}")

        message_body = body.get("message") or {}
        if not isinstance(message_body, dict):
            raise errors.analysis(
                f"Ollama /api/chat returned a malformed 'message' for {model}: "
                f"{type(message_body).__name__}"
            )
        try:
            return ChatResult(
                content=str(message_body.get("content") or ""),
                thinking=str(message_body.get("thinking") or ""),
                model=str(body.get("model") or model),
                done_reason=body.get("done_reason"),
                prompt_eval_count=int(body.get("prompt_eval_count") or 0),
                eval_count=int(body.get("eval_count") or 0),
                load_duration_s=float(body.get("load_duration") or 0) / 1e9,
                total_duration_s=float(body.get("total_duration") or 0) / 1e9,
            )
        except (TypeError, ValueError) as e:
            raise errors.analysis(
                f"Ollama /api/chat returned malformed statistics for {model}: {e}"
            ) from e

    def chat_with_images(
        self,
        prompt: str,
        image_paths: Sequence[Path],
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        think: ThinkSetting = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[str, int]] = "30m",
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
        read_timeout: Optional[float] = None,
    ) -> ChatResult:
        """Ask a vision model about one or more images via ``POST /api/chat``.

        All images go into one user message, each orientation-corrected,
        downscaled to ``max_image_edge`` and JPEG-encoded. ``think`` is
        handled by :func:`resolve_think`: it is only sent when the model
        reports the ``thinking`` capability.

        Args:
            prompt: User message text.
            image_paths: Images to attach (at least one).
            model: Model name (uses instance default if None).
            system: Optional system message.
            format: ``'json'`` or a JSON-schema dict constraining the output.
            think: Thinking switch or level; None leaves the model default.
            options: Generation options (temperature, num_ctx, num_predict, ...).
            keep_alive: How long Ollama keeps the model loaded afterwards.
            max_image_edge: Long-edge limit in pixels (0 keeps full size).
            read_timeout: Read timeout in seconds (default: ``generate_timeout``).

        Returns:
            ChatResult with the answer and token/timing statistics.

        Raises:
            ValueError: If no model is set, no image is given, an image is
                invalid or ``think`` is not a known level.
            FileNotFoundError: If an image file does not exist.
            VisionModelUnavailableError: If the model is missing, incompatible
                with this Ollama, or cannot take images.
            OllamaConnectionError: If Ollama cannot be reached.
            VisionTimeoutError: If no response arrives within the read timeout.
            VisionTransientError: For HTTP 5xx/408/429 and other request failures.
            VisionAnalysisError: For any other HTTP or protocol error.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("At least one image is required")

        caps = self._required_capabilities(model, _VISION_ERRORS)
        images = [prepare_image_b64(p, max_image_edge) for p in paths]
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images})
        payload = self._build_payload(
            model, messages, caps, format=format, think=think, options=options, keep_alive=keep_alive
        )

        read = read_timeout if read_timeout is not None else self.generate_timeout
        logger.debug("Ollama API request: POST /api/chat (%s, %d image(s))", model, len(images))
        return self._post_chat(payload, model, read, errors=_VISION_ERRORS)

    def chat_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        think: ThinkSetting = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[str, int]] = "5m",
        read_timeout: Optional[float] = None,
    ) -> ChatResult:
        """Ask a text model a question via ``POST /api/chat`` (no images).

        Used by the fact-check stage (e.g. gpt-oss with a JSON schema in
        ``format`` and a think level). The model must report the
        ``completion`` capability; a failed capability read is advisory.

        Args:
            prompt: User message text.
            model: Model name (uses instance default if None).
            system: Optional system message.
            format: ``'json'`` or a JSON-schema dict constraining the output.
            think: Thinking switch or level (``'low'|'medium'|'high'``); see
                :func:`resolve_think`. None leaves the model default.
            options: Generation options (temperature, num_ctx, num_predict, ...).
            keep_alive: How long Ollama keeps the model loaded afterwards.
            read_timeout: Read timeout in seconds (default: ``generate_timeout``).

        Returns:
            ChatResult with the answer, the separate reasoning text and
            token/timing statistics.

        Raises:
            ValueError: If no model is set or ``think`` is not a known level.
            TextModelUnavailableError: If the model is missing, incompatible
                with this Ollama, or lacks the ``completion`` capability.
            OllamaConnectionError: If Ollama cannot be reached.
            TextTimeoutError: If no response arrives within the read timeout.
            TextTransientError: For HTTP 5xx/408/429 and other request failures.
            TextAnalysisError: For any other HTTP error or a non-JSON body.
        """
        model = model or self.model
        if not model:
            raise ValueError("Model name required (not set in instance or call)")

        caps = self._required_capabilities(model, _TEXT_ERRORS)
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = self._build_payload(
            model, messages, caps, format=format, think=think, options=options, keep_alive=keep_alive
        )

        read = read_timeout if read_timeout is not None else self.generate_timeout
        logger.debug("Ollama API request: POST /api/chat (%s, text only)", model)
        return self._post_chat(payload, model, read, errors=_TEXT_ERRORS)

    def close(self):
        """Close every HTTP session this client created (all threads)."""
        with self._sessions_lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            try:
                session.close()
            except Exception as e:  # closing must never raise
                logger.debug("Error closing HTTP session: %s", e)
        self._local = threading.local()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
