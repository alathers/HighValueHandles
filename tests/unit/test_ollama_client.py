"""Unit tests for OllamaClient class.

Tests REST API communication with mocked HTTP requests.
"""

import base64
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image as PILImage

from lib.exceptions import (
    FATAL_MODEL_ERRORS,
    OllamaConnectionError,
    TextAnalysisError,
    TextModelUnavailableError,
    TextResponseError,
    TextTimeoutError,
    TextTransientError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionTimeoutError,
    VisionTransientError,
)
from lib.ollama_client import (
    ChatResult,
    OllamaClient,
    is_gpt_oss_model,
    prepare_image_b64,
    resolve_think,
)


class TestOllamaClientInitialization:
    """Test OllamaClient initialization."""

    def test_client_default_values(self):
        """Test client initializes with default values."""
        client = OllamaClient()

        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30
        assert client.model is None
        assert client.session is not None
        assert client.generate_timeout == 600.0
        assert client.connect_timeout == 10.0

    def test_client_custom_values(self):
        """Test client initializes with custom values."""
        client = OllamaClient(
            base_url="http://custom-host:9999",
            timeout=60,
            model="qwen2-vl:7b",
        )

        assert client.base_url == "http://custom-host:9999"
        assert client.timeout == 60
        assert client.model == "qwen2-vl:7b"

    def test_base_url_trailing_slash_removed(self):
        """Test trailing slash is removed from base_url."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_base_url_no_trailing_slash_kept(self):
        """Test base_url without trailing slash is kept."""
        client = OllamaClient(base_url="http://localhost:11434")
        assert client.base_url == "http://localhost:11434"


class TestOllamaClientPullModel:
    """Test model pull/download functionality."""

    def test_client_has_no_pull_model(self):
        """Pulls only go through ModelServerManager.ensure_model(allow_pull=True)."""
        assert not hasattr(OllamaClient(), "pull_model")


class TestOllamaClientSessionManagement:
    """Test HTTP session management."""

    @patch("requests.Session.close")
    def test_close_session(self, mock_close):
        """Test closing HTTP session."""
        client = OllamaClient()

        client.close()

        # Session should be closed
        mock_close.assert_called_once()

    def test_context_manager(self):
        """Test using client as context manager."""
        with patch.object(OllamaClient, "close") as mock_close:
            with OllamaClient() as client:
                assert client is not None

            mock_close.assert_called_once()


class TestOllamaClientErrors:
    """Error paths of the metadata request (``get_capabilities`` -> ``_make_request``)."""

    @patch("requests.Session.request")
    def test_http_500_keeps_ollama_message(self, mock_request):
        response = Mock()
        response.status_code = 500
        response.json.return_value = {"error": "out of memory"}
        mock_request.return_value = response

        with pytest.raises(requests.HTTPError, match="HTTP 500: out of memory"):
            OllamaClient().get_capabilities("test-model")

    @patch("requests.Session.request")
    def test_request_timeout_propagates(self, mock_request):
        mock_request.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            OllamaClient().get_capabilities("test-model")


# ---------------------------------------------------------------------------
# Vision: prepare_image_b64, capabilities, chat_with_images
# ---------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"name": {"type": ["string", "null"]}}, "required": ["name"]}


def _jpeg(path: Path, size=(64, 48), exif_orientation=None) -> Path:
    image = PILImage.new("RGB", size, color="white")
    if exif_orientation is not None:
        exif = PILImage.Exif()
        exif[0x0112] = exif_orientation
        image.save(path, format="JPEG", exif=exif.tobytes())
    else:
        image.save(path, format="JPEG")
    return path


def _decode(b64: str) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(b64)))


def _http(status=200, body=None, text=None):
    response = Mock()
    response.status_code = status
    if body is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = body
    response.text = text if text is not None else json.dumps(body or {})
    return response


def _router(show=None, chat=None):
    """Fake Session.request answering /api/show and /api/chat."""
    show = show if show is not None else _http(200, {"capabilities": ["completion", "vision", "thinking"]})

    def request(method, url, *args, **kwargs):
        if url.endswith("/api/show"):
            if isinstance(show, BaseException):
                raise show
            return show
        if url.endswith("/api/chat"):
            if isinstance(chat, BaseException):
                raise chat
            return chat
        raise AssertionError(f"unexpected request {method} {url}")

    return request


CHAT_OK = {
    "model": "qwen3.6:35b-a3b",
    "message": {"role": "assistant", "content": '{"name": "Blood Oath"}', "thinking": ""},
    "done_reason": "stop",
    "prompt_eval_count": 1654,
    "eval_count": 113,
    "load_duration": 2_500_000_000,
    "total_duration": 3_000_000_000,
}


class TestPrepareImageB64:
    def test_downscales_long_edge(self, tmp_path):
        path = _jpeg(tmp_path / "big.jpg", size=(4000, 3000))
        image = _decode(prepare_image_b64(path))
        assert max(image.size) == 1280
        assert image.size == (1280, 960)
        assert image.format == "JPEG"

    def test_small_image_not_upscaled(self, tmp_path):
        image = _decode(prepare_image_b64(_jpeg(tmp_path / "s.jpg", size=(64, 48))))
        assert image.size == (64, 48)

    def test_zero_max_edge_keeps_size(self, tmp_path):
        image = _decode(prepare_image_b64(_jpeg(tmp_path / "b.jpg", size=(2000, 1000)), max_edge=0))
        assert image.size == (2000, 1000)

    def test_exif_orientation_applied(self, tmp_path):
        # Orientation 6 = rotate 90 degrees clockwise when displayed.
        path = _jpeg(tmp_path / "rot.jpg", size=(200, 100), exif_orientation=6)
        assert _decode(prepare_image_b64(path)).size == (100, 200)

    def test_converts_to_rgb(self, tmp_path):
        path = tmp_path / "alpha.png"
        PILImage.new("RGBA", (10, 10), (255, 0, 0, 128)).save(path)
        assert _decode(prepare_image_b64(path)).mode == "RGB"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            prepare_image_b64(tmp_path / "nope.jpg")

    def test_invalid_file(self, tmp_path):
        path = tmp_path / "bad.jpg"
        path.write_text("not an image")
        with pytest.raises(ValueError, match="Invalid image file"):
            prepare_image_b64(path)


class TestOllamaClientCapabilities:
    @patch("requests.Session.request")
    def test_capabilities_parsed_and_cached(self, mock_request):
        mock_request.side_effect = _router()
        client = OllamaClient()

        assert client.get_capabilities("m") == {"completion", "vision", "thinking"}
        assert client.get_capabilities("m") == {"completion", "vision", "thinking"}  # cached
        assert mock_request.call_count == 1
        assert mock_request.call_args.kwargs["json"] == {"model": "m"}
        assert mock_request.call_args.kwargs["timeout"] == 30

    @patch("requests.Session.request")
    def test_missing_model_is_unavailable(self, mock_request):
        mock_request.side_effect = _router(show=_http(404, {"error": "model 'qwen3-vl:32b' not found"}))
        client = OllamaClient()

        with pytest.raises(VisionModelUnavailableError, match="not installed"):
            client.get_capabilities("qwen3-vl:32b")

    @patch("requests.Session.request")
    def test_unreported_capabilities_are_empty(self, mock_request):
        mock_request.side_effect = _router(show=_http(200, {"details": {}}))
        assert OllamaClient().get_capabilities("old") == set()

    @patch("requests.Session.request")
    def test_http_error_keeps_ollama_message(self, mock_request):
        mock_request.side_effect = _router(show=_http(500, {"error": "disk on fire"}))
        with pytest.raises(requests.HTTPError, match="HTTP 500: disk on fire"):
            OllamaClient().get_capabilities("m")


class TestOllamaClientChatWithImages:
    def _images(self, tmp_path, n=2):
        return [_jpeg(tmp_path / f"img{i}.jpg", size=(3000, 2000)) for i in range(n)]

    @patch("requests.Session.request")
    def test_request_payload(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(200, CHAT_OK))
        client = OllamaClient()
        options = {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}

        result = client.chat_with_images(
            "Read the label",
            self._images(tmp_path),
            model="qwen3.6:35b-a3b",
            system="You are an expert",
            format=SCHEMA,
            think=False,
            options=options,
        )

        url = mock_request.call_args.args[1]
        kwargs = mock_request.call_args.kwargs
        payload = kwargs["json"]
        assert url == "http://localhost:11434/api/chat"
        assert kwargs["timeout"] == (10.0, 600.0)
        assert payload["model"] == "qwen3.6:35b-a3b"
        assert payload["stream"] is False
        assert payload["format"] == SCHEMA
        assert payload["options"] == options
        assert payload["keep_alive"] == "30m"
        assert payload["think"] is False
        assert payload["messages"][0] == {"role": "system", "content": "You are an expert"}
        user = payload["messages"][-1]
        assert user["role"] == "user"
        assert user["content"] == "Read the label"
        assert len(user["images"]) == 2
        assert max(_decode(user["images"][0]).size) == 1280

        assert result == ChatResult(
            content='{"name": "Blood Oath"}',
            thinking="",
            model="qwen3.6:35b-a3b",
            done_reason="stop",
            prompt_eval_count=1654,
            eval_count=113,
            load_duration_s=2.5,
            total_duration_s=3.0,
        )

    @patch("requests.Session.request")
    def test_think_omitted_without_thinking_capability(self, mock_request, tmp_path):
        mock_request.side_effect = _router(
            show=_http(200, {"capabilities": ["completion", "vision"]}), chat=_http(200, CHAT_OK)
        )
        OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1), think=False)
        assert "think" not in mock_request.call_args.kwargs["json"]

    @patch("requests.Session.request")
    def test_custom_timeouts(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(200, CHAT_OK))
        client = OllamaClient(model="m", generate_timeout=120, connect_timeout=3)
        client.chat_with_images("p", self._images(tmp_path, 1))
        assert mock_request.call_args.kwargs["timeout"] == (3, 120)
        client.chat_with_images("p", self._images(tmp_path, 1), read_timeout=5)
        assert mock_request.call_args.kwargs["timeout"] == (3, 5)

    @patch("requests.Session.request")
    def test_model_without_vision_rejected_before_upload(self, mock_request, tmp_path):
        mock_request.side_effect = _router(show=_http(200, {"capabilities": ["completion"]}))
        with pytest.raises(VisionModelUnavailableError, match="does not support image input"):
            OllamaClient().chat_with_images("p", self._images(tmp_path, 1), model="gpt-oss:120b")
        assert all(not c.args[1].endswith("/api/chat") for c in mock_request.call_args_list)

    @patch("requests.Session.request")
    def test_capability_lookup_failure_is_not_fatal(self, mock_request, tmp_path):
        mock_request.side_effect = _router(show=_http(500, {"error": "busy"}), chat=_http(200, CHAT_OK))
        result = OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))
        assert result.content == '{"name": "Blood Oath"}'
        assert "think" not in mock_request.call_args.kwargs["json"]

    def test_requires_model_and_images(self, tmp_path):
        client = OllamaClient()
        with pytest.raises(ValueError, match="Model name required"):
            client.chat_with_images("p", self._images(tmp_path, 1))
        with pytest.raises(ValueError, match="At least one image"):
            client.chat_with_images("p", [], model="m")

    @pytest.mark.parametrize(
        "status,body",
        [
            (404, {"error": "model 'x' not found"}),
            (500, {"error": "'llama3.2-vision' is no longer compatible with your version of Ollama"}),
            (400, {"error": "model does not support images"}),
        ],
    )
    @patch("requests.Session.request")
    def test_unusable_model_errors(self, mock_request, status, body, tmp_path):
        mock_request.side_effect = _router(chat=_http(status, body))
        with pytest.raises(VisionModelUnavailableError, match="is not usable"):
            OllamaClient(model="x").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_other_http_error_keeps_message(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(500, {"error": "llama runner process has terminated"}))
        with pytest.raises(VisionAnalysisError, match=r"HTTP 500.*llama runner process has terminated"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_non_json_error_body(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(502, None, text="Bad Gateway"))
        with pytest.raises(VisionAnalysisError, match="HTTP 502.*Bad Gateway"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_error_in_200_body(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(200, {"error": "something odd"}))
        with pytest.raises(VisionAnalysisError, match="something odd"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_read_timeout(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=requests.ReadTimeout("read timed out"))
        with pytest.raises(VisionTimeoutError, match="did not respond within 600s"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_connect_timeout_is_connection_error(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=requests.ConnectTimeout("connect timed out"))
        with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_connection_error(self, mock_request, tmp_path):
        mock_request.side_effect = _router(show=requests.ConnectionError("refused"))
        with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama at http://localhost:11434"):
            OllamaClient(model="m").chat_with_images("p", self._images(tmp_path, 1))

    @patch("requests.Session.request")
    def test_missing_image_file(self, mock_request, tmp_path):
        mock_request.side_effect = _router(chat=_http(200, CHAT_OK))
        with pytest.raises(FileNotFoundError):
            OllamaClient(model="m").chat_with_images("p", [tmp_path / "missing.jpg"])


class TestOllamaClientMakeRequestErrors:
    @patch("requests.Session.request")
    def test_404_is_model_unavailable(self, mock_request):
        mock_request.return_value = _http(404, {"error": "model 'x' not found"})
        with pytest.raises(VisionModelUnavailableError, match="not installed") as exc_info:
            OllamaClient().get_capabilities("x")
        assert isinstance(exc_info.value.__cause__, requests.HTTPError)
        assert exc_info.value.__cause__.response.status_code == 404

    @patch("requests.Session.request")
    def test_500_is_http_error_with_message(self, mock_request):
        mock_request.return_value = _http(500, {"error": "model 'x' failed to load"})
        with pytest.raises(requests.HTTPError, match="HTTP 500: model 'x' failed to load"):
            OllamaClient().get_capabilities("x")


def test_client_vision_timeouts():
    client = OllamaClient(generate_timeout=42.0, connect_timeout=2.0)
    assert client.generate_timeout == 42.0
    assert client.connect_timeout == 2.0


# ---------------------------------------------------------------------------
# Text: chat_text, think handling, error mapping, sessions
# ---------------------------------------------------------------------------

GPT_OSS_CAPS = _http(200, {"capabilities": ["completion", "tools", "thinking"]})
TEXT_OK = {
    "model": "gpt-oss:120b",
    "message": {"role": "assistant", "content": '{"claims": []}', "thinking": "Checking the page..."},
    "done_reason": "stop",
    "prompt_eval_count": 4000,
    "eval_count": 389,
    "load_duration": 24_000_000_000,
    "total_duration": 27_000_000_000,
}


def _chat_payload(mock_request) -> dict:
    chat_calls = [c for c in mock_request.call_args_list if c.args[1].endswith("/api/chat")]
    assert chat_calls, "no /api/chat request was sent"
    return chat_calls[-1].kwargs["json"]


class TestChatText:
    @patch("requests.Session.request")
    def test_payload_has_no_images_and_passes_options(self, mock_request):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=_http(200, TEXT_OK))
        options = {"temperature": 0, "num_ctx": 16384, "num_predict": 6144}

        result = OllamaClient().chat_text(
            "Extract the claims",
            model="gpt-oss:120b",
            system="Quote verbatim",
            format=SCHEMA,
            think="high",
            options=options,
            keep_alive="10m",
            read_timeout=900,
        )

        payload = _chat_payload(mock_request)
        assert payload["model"] == "gpt-oss:120b"
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "Quote verbatim"},
            {"role": "user", "content": "Extract the claims"},
        ]
        assert all("images" not in m for m in payload["messages"])
        assert payload["format"] == SCHEMA
        assert payload["options"] == options
        assert payload["keep_alive"] == "10m"
        assert payload["think"] == "high"
        assert mock_request.call_args.kwargs["timeout"] == (10.0, 900)
        assert result == ChatResult(
            content='{"claims": []}',
            thinking="Checking the page...",
            model="gpt-oss:120b",
            done_reason="stop",
            prompt_eval_count=4000,
            eval_count=389,
            load_duration_s=24.0,
            total_duration_s=27.0,
        )

    @patch("requests.Session.request")
    def test_defaults(self, mock_request):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=_http(200, TEXT_OK))
        OllamaClient(model="gpt-oss:120b").chat_text("p")
        payload = _chat_payload(mock_request)
        assert payload["keep_alive"] == "5m"
        assert "think" not in payload  # None leaves the model default
        assert "format" not in payload
        assert "options" not in payload
        assert payload["messages"] == [{"role": "user", "content": "p"}]
        assert mock_request.call_args.kwargs["timeout"] == (10.0, 600.0)

    @patch("requests.Session.request")
    def test_keep_alive_none_omitted(self, mock_request):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=_http(200, TEXT_OK))
        OllamaClient(model="gpt-oss:120b").chat_text("p", keep_alive=None)
        assert "keep_alive" not in _chat_payload(mock_request)

    @patch("requests.Session.request")
    def test_think_level_needs_thinking_capability(self, mock_request):
        mock_request.side_effect = _router(
            show=_http(200, {"capabilities": ["completion"]}), chat=_http(200, TEXT_OK)
        )
        OllamaClient(model="llama3.1").chat_text("p", think="high")
        assert "think" not in _chat_payload(mock_request)

    @pytest.mark.parametrize("think,expected", [(False, "low"), (True, "medium"), ("high", "high")])
    @patch("requests.Session.request")
    def test_gpt_oss_bool_maps_to_level(self, mock_request, think, expected):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=_http(200, TEXT_OK))
        OllamaClient(model="gpt-oss:20b").chat_text("p", think=think)
        assert _chat_payload(mock_request)["think"] == expected

    @patch("requests.Session.request")
    def test_other_thinking_model_gets_bool(self, mock_request):
        mock_request.side_effect = _router(
            show=_http(200, {"capabilities": ["completion", "thinking"]}), chat=_http(200, TEXT_OK)
        )
        OllamaClient(model="qwen3:32b").chat_text("p", think=False)
        assert _chat_payload(mock_request)["think"] is False

    def test_invalid_think_level_rejected(self):
        with pytest.raises(ValueError, match="think must be"):
            resolve_think("max", "gpt-oss:120b", {"thinking"})

    @patch("requests.Session.request")
    def test_model_without_completion_rejected(self, mock_request):
        mock_request.side_effect = _router(show=_http(200, {"capabilities": ["embedding"]}))
        with pytest.raises(TextModelUnavailableError, match="'completion'"):
            OllamaClient().chat_text("p", model="nomic-embed-text")
        assert all(not c.args[1].endswith("/api/chat") for c in mock_request.call_args_list)

    @patch("requests.Session.request")
    def test_missing_model_is_text_unavailable(self, mock_request):
        mock_request.side_effect = _router(show=_http(404, {"error": "model 'gpt-oss:120b' not found"}))
        with pytest.raises(TextModelUnavailableError, match="not installed"):
            OllamaClient().chat_text("p", model="gpt-oss:120b")

    @patch("requests.Session.request")
    def test_capability_read_failure_is_advisory(self, mock_request):
        mock_request.side_effect = _router(show=_http(500, {"error": "busy"}), chat=_http(200, TEXT_OK))
        result = OllamaClient(model="gpt-oss:120b").chat_text("p", think="high")
        assert result.content == '{"claims": []}'
        assert "think" not in _chat_payload(mock_request)  # capability unknown

    def test_requires_model(self):
        with pytest.raises(ValueError, match="Model name required"):
            OllamaClient().chat_text("p")

    @patch("requests.Session.request")
    def test_connection_error(self, mock_request):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=requests.ConnectionError("refused"))
        with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama"):
            OllamaClient(model="gpt-oss:120b").chat_text("p")

    @patch("requests.Session.request")
    def test_non_json_200_body(self, mock_request):
        mock_request.side_effect = _router(show=GPT_OSS_CAPS, chat=_http(200, None, text="<html>"))
        with pytest.raises(TextAnalysisError, match="non-JSON"):
            OllamaClient(model="gpt-oss:120b").chat_text("p")


class TestChatErrorMapping:
    """HTTP status / exception -> error class, for vision and text alike."""

    @staticmethod
    def _call(kind, tmp_path):
        client = OllamaClient(model="m")
        if kind == "vision":
            return client.chat_with_images("p", [_jpeg(tmp_path / "x.jpg")])
        return client.chat_text("p")

    ERRORS = {
        "vision": {
            "transient": VisionTransientError,
            "analysis": VisionAnalysisError,
            "unavailable": VisionModelUnavailableError,
            "timeout": VisionTimeoutError,
        },
        "text": {
            "transient": TextTransientError,
            "analysis": TextAnalysisError,
            "unavailable": TextModelUnavailableError,
            "timeout": TextTimeoutError,
        },
    }

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @pytest.mark.parametrize("status", [500, 502, 503, 429, 408])
    @patch("requests.Session.request")
    def test_transient_statuses(self, mock_request, status, kind, tmp_path):
        mock_request.side_effect = _router(chat=_http(status, {"error": "busy"}))
        with pytest.raises(self.ERRORS[kind]["transient"], match=f"HTTP {status}"):
            self._call(kind, tmp_path)

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @patch("requests.Session.request")
    def test_generic_request_exception_is_transient(self, mock_request, kind, tmp_path):
        mock_request.side_effect = _router(chat=requests.exceptions.ChunkedEncodingError("cut"))
        with pytest.raises(self.ERRORS[kind]["transient"], match="request failed"):
            self._call(kind, tmp_path)

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @patch("requests.Session.request")
    def test_400_is_analysis_error_not_transient(self, mock_request, kind, tmp_path):
        mock_request.side_effect = _router(chat=_http(400, {"error": "invalid format schema"}))
        with pytest.raises(self.ERRORS[kind]["analysis"], match="HTTP 400") as exc_info:
            self._call(kind, tmp_path)
        assert not isinstance(exc_info.value, self.ERRORS[kind]["transient"])

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @patch("requests.Session.request")
    def test_404_is_unavailable(self, mock_request, kind, tmp_path):
        mock_request.side_effect = _router(chat=_http(404, {"error": "model 'm' not found"}))
        with pytest.raises(self.ERRORS[kind]["unavailable"], match="is not usable"):
            self._call(kind, tmp_path)

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @patch("requests.Session.request")
    def test_read_timeout(self, mock_request, kind, tmp_path):
        mock_request.side_effect = _router(chat=requests.ReadTimeout("slow"))
        with pytest.raises(self.ERRORS[kind]["timeout"], match="did not respond within 600s"):
            self._call(kind, tmp_path)

    @pytest.mark.parametrize("kind", ["vision", "text"])
    @pytest.mark.parametrize(
        "body",
        [
            {"model": "m", "message": "not a dict"},
            {"model": "m", "message": {"content": "x"}, "eval_count": "many"},
        ],
    )
    @patch("requests.Session.request")
    def test_malformed_200_body_is_analysis_error(self, mock_request, body, kind, tmp_path):
        mock_request.side_effect = _router(chat=_http(200, body))
        with pytest.raises(self.ERRORS[kind]["analysis"], match="malformed"):
            self._call(kind, tmp_path)

    def test_exception_hierarchy(self):
        assert issubclass(VisionTransientError, VisionAnalysisError)
        for cls in (TextResponseError, TextTimeoutError, TextTransientError):
            assert issubclass(cls, TextAnalysisError)
        for cls in (TextAnalysisError, TextModelUnavailableError, TextResponseError,
                    TextTimeoutError, TextTransientError):
            assert cls not in FATAL_MODEL_ERRORS
            assert not issubclass(cls, FATAL_MODEL_ERRORS)


class TestThinkResolution:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-oss:120b", True),
            ("gpt-oss:20b", True),
            ("GPT-OSS", True),
            ("registry.ollama.ai/library/gpt-oss:120b", True),
            ("qwen3.6:35b-a3b", False),
            ("my-gpt-oss", False),
        ],
    )
    def test_gpt_oss_family(self, model, expected):
        assert is_gpt_oss_model(model) is expected

    def test_none_or_no_thinking_capability_omits(self):
        assert resolve_think(None, "gpt-oss:120b", {"thinking"}) is None
        assert resolve_think("high", "gpt-oss:120b", {"completion"}) is None
        assert resolve_think(True, "qwen3:32b", None) is None
        assert resolve_think(True, "qwen3:32b", set()) is None

    def test_levels_and_bools(self):
        caps = {"completion", "thinking"}
        assert resolve_think("low", "qwen3:32b", caps) == "low"
        assert resolve_think(True, "qwen3:32b", caps) is True
        assert resolve_think(False, "gpt-oss:120b", caps) == "low"
        assert resolve_think(True, "gpt-oss:120b", caps) == "medium"


class TestSessionsPerThread:
    def test_threads_get_distinct_sessions_and_close_closes_all(self):
        import threading

        client = OllamaClient()
        main = client.session
        assert client.session is main  # stable within a thread
        seen = {}

        def grab(key):
            seen[key] = client.session

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sessions = [main, seen[0], seen[1]]
        assert len({id(s) for s in sessions}) == 3
        with patch("requests.Session.close", autospec=True) as close:
            client.close()
        closed = {id(call.args[0]) for call in close.call_args_list}
        assert closed == {id(s) for s in sessions}

    def test_sessions_mount_pool_of_ten(self):
        adapter = OllamaClient().session.get_adapter("http://localhost:11434")
        assert adapter._pool_maxsize == 10

    def test_injected_session_is_used_and_closed(self):
        client = OllamaClient()
        fake = Mock()
        fake.request.return_value = _http(200, {"capabilities": ["vision"]})
        client.session = fake
        assert client.get_capabilities("m") == {"vision"}
        client.close()
        fake.close.assert_called_once()
