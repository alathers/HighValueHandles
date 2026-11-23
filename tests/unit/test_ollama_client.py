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
    OllamaConnectionError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionTimeoutError,
)
from lib.ollama_client import (
    ChatCompletionResponse,
    EmbeddingResponse,
    OllamaClient,
    VisionChatResult,
    get_ollama_client,
    prepare_image_b64,
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


class TestOllamaClientChatCompletion:
    """Test chat completion functionality."""

    @patch("requests.Session.request")
    def test_chat_completion_basic(self, mock_request):
        """Test basic chat completion request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "test-model",
            "message": {"content": "Test response"},
            "done": True,
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="test-model")
        responses = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert len(responses) == 1
        assert responses[0].message == "Test response"
        assert responses[0].model == "test-model"

    @patch("requests.Session.request")
    def test_chat_completion_with_instance_model(self, mock_request):
        """Test chat completion uses instance model if not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "instance-model",
            "message": {"content": "Response"},
            "done": True,
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="instance-model")
        responses = client.chat_completion(messages=[{"role": "user", "content": "Hi"}])

        assert responses[0].model == "instance-model"

    @patch("requests.Session.request")
    def test_chat_completion_with_explicit_model(self, mock_request):
        """Test explicit model overrides instance model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "explicit-model",
            "message": {"content": "Response"},
            "done": True,
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="instance-model")
        responses = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}], model="explicit-model"
        )

        assert responses[0].model == "explicit-model"

    @patch("requests.Session.request")
    def test_chat_completion_no_model_raises(self, mock_request):
        """Test chat completion raises ValueError without model."""
        client = OllamaClient()  # No instance model

        with pytest.raises(ValueError, match="Model name required"):
            client.chat_completion(messages=[{"role": "user", "content": "Hello"}])

    @patch("requests.Session.request")
    def test_chat_completion_with_options(self, mock_request):
        """Test chat completion with generation options."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "test-model",
            "message": {"content": "Response"},
            "done": True,
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="test-model")
        responses = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            options={"temperature": 0.7, "top_k": 40},
        )

        assert len(responses) == 1

        # Verify options were sent
        call_kwargs = mock_request.call_args.kwargs
        json_data = call_kwargs["json"]
        assert json_data["options"]["temperature"] == 0.7
        assert json_data["options"]["top_k"] == 40

    @patch("requests.Session.request")
    def test_chat_completion_with_format(self, mock_request):
        """Test chat completion with format option."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "test-model",
            "message": {"content": '{"key": "value"}'},
            "done": True,
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="test-model")
        responses = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}], format="json"
        )

        assert len(responses) == 1

        # Verify format was sent
        call_kwargs = mock_request.call_args.kwargs
        json_data = call_kwargs["json"]
        assert json_data["format"] == "json"


class TestOllamaClientImageToText:
    """Test image-to-text functionality."""

    @patch("requests.Session.request")
    @patch("pathlib.Path.exists", return_value=True)
    def test_image_to_text_basic(self, mock_exists, mock_request):
        """Test basic image-to-text request."""
        with patch("PIL.Image.Image.verify"):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "qwen2-vl:7b",
                "response": "Extracted text from image",
            }
            mock_request.return_value = mock_response

            # Create test image file
            with patch("builtins.open", create=True) as mock_open:
                mock_file = Mock()
                mock_file.read.return_value = (
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00"
                )  # JPEG header
                mock_open.return_value.__enter__ = Mock(return_value=mock_file)
                mock_open.return_value.__exit__ = Mock(return_value=False)

                client = OllamaClient(model="qwen2-vl:7b")
                result = client.image_to_text(
                    Path("/tmp/test.jpg"), prompt="Extract text"
                )

                assert result == "Extracted text from image"

    @patch("requests.Session.request")
    def test_image_to_text_image_not_found_raises(self, mock_request):
        """Test image_to_text raises FileNotFoundError when image doesn't exist."""
        client = OllamaClient(model="test-model")

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            client.image_to_text(Path("/nonexistent/image.jpg"), prompt="Test")

    @patch("requests.Session.request")
    @patch("pathlib.Path.exists", return_value=True)
    def test_image_to_text_invalid_image_raises(self, mock_exists, mock_request):
        """Test image_to_text raises ValueError for invalid image."""
        with patch("PIL.Image.open") as mock_img_open:
            mock_img = Mock()
            mock_img.verify.side_effect = Exception("Invalid JPEG")
            mock_img_open.return_value = mock_img

            client = OllamaClient(model="test-model")

            with pytest.raises(ValueError, match="Invalid image file"):
                client.image_to_text(Path("/tmp/invalid.jpg"), prompt="Test")

    @patch("requests.Session.request")
    def test_image_to_text_no_model_raises(self, mock_request):
        """Test image_to_text raises ValueError without model."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Model name required"):
            client.image_to_text(Path("/tmp/test.jpg"), prompt="Test")


class TestOllamaClientEmbedding:
    """Test embedding generation."""

    @patch("requests.Session.request")
    def test_embedding_basic(self, mock_request):
        """Test basic embedding generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "nomic-embed-text",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        mock_request.return_value = mock_response

        client = OllamaClient(model="nomic-embed-text")
        response = client.embedding(text="Test text")

        assert isinstance(response, EmbeddingResponse)
        assert response.model == "nomic-embed-text"
        assert response.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @patch("requests.Session.request")
    def test_embedding_no_model_raises(self, mock_request):
        """Test embedding raises ValueError without model."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Model name required"):
            client.embedding(text="Test text")


class TestOllamaClientListModels:
    """Test list models functionality."""

    @patch("requests.Session.request")
    def test_list_models_basic(self, mock_request):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2-vl:7b", "modified_at": "2026-11-13T10:00:00Z", "size": 4096},
                {"name": "llava:latest", "modified_at": "2026-11-12T10:00:00Z", "size": 2048},
            ]
        }
        mock_request.return_value = mock_response

        client = OllamaClient()
        models = client.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "qwen2-vl:7b"
        assert models[1]["name"] == "llava:latest"

    @patch("requests.Session.request")
    def test_list_models_empty(self, mock_request):
        """Test listing models when none available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_request.return_value = mock_response

        client = OllamaClient()
        models = client.list_models()

        assert models == []


class TestOllamaClientPullModel:
    """Test model pull/download functionality."""

    def test_client_has_no_pull_model(self):
        """Pulls only go through ModelServerManager.ensure_model(allow_pull=True)."""
        assert not hasattr(OllamaClient(), "pull_model")


class TestOllamaClientCheckModel:
    """Test check model functionality."""

    @patch("requests.Session.request")
    def test_check_model_found(self, mock_request):
        """Test check_model returns True when model exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2-vl:7b", "size": 4096},
                {"name": "llava:latest", "size": 2048},
            ]
        }
        mock_request.return_value = mock_response

        client = OllamaClient()
        exists = client.check_model("qwen2-vl:7b")

        assert exists is True

    @patch("requests.Session.request")
    def test_check_model_not_found(self, mock_request):
        """Test check_model returns False when model doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "qwen2-vl:7b", "size": 4096}]
        }
        mock_request.return_value = mock_response

        client = OllamaClient()
        exists = client.check_model("nonexistent-model")

        assert exists is False

    @patch("requests.Session.request")
    def test_check_model_no_instance_model(self, mock_request):
        """Test check_model returns False when no model specified."""
        client = OllamaClient()  # No instance model
        exists = client.check_model()

        assert exists is False

    @patch("requests.Session.request")
    def test_check_model_error_returns_false(self, mock_request):
        """Test check_model returns False on API error."""
        mock_request.side_effect = Exception("API error")

        client = OllamaClient()
        exists = client.check_model("test-model")

        assert exists is False


class TestOllamaClientHealthCheck:
    """Test health check functionality."""

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test health_check returns True when server is healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OllamaClient()
        is_healthy = client.health_check()

        assert is_healthy is True

    @patch("requests.Session.get")
    def test_health_check_failure(self, mock_get):
        """Test health_check returns False when server is unhealthy."""
        mock_get.side_effect = Exception("Connection refused")

        client = OllamaClient()
        is_healthy = client.health_check()

        assert is_healthy is False

    @patch("requests.Session.get")
    def test_health_check_timeout(self, mock_get):
        """Test health_check returns False on timeout."""
        mock_get.side_effect = Exception("Request timed out")

        client = OllamaClient()
        is_healthy = client.health_check()

        assert is_healthy is False


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
    """Test error handling."""

    @patch("requests.Session.request")
    def test_request_http_error_raises(self, mock_request):
        """Test HTTP errors raise RequestException."""
        from requests.exceptions import HTTPError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("Not Found")
        mock_request.return_value = mock_response

        client = OllamaClient(model="test-model")

        with pytest.raises(HTTPError):
            client.chat_completion(messages=[{"role": "user", "content": "Hi"}])

    @patch("requests.Session.request")
    def test_request_timeout(self, mock_request):
        """Test request timeout raises exception."""
        from requests.exceptions import Timeout

        mock_request.side_effect = Timeout("Request timed out")

        client = OllamaClient(model="test-model")

        with pytest.raises(Timeout):
            client.list_models()


class TestOllamaClientConvenienceFunction:
    """Test get_ollama_client convenience function."""

    def test_get_ollama_client_basic(self):
        """Test get_ollama_client creates configured client."""
        client = get_ollama_client()

        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30

    def test_get_ollama_client_with_params(self):
        """Test get_ollama_client with custom parameters."""
        client = get_ollama_client(
            base_url="http://custom:9999", timeout=60, model="test-model"
        )

        assert client.base_url == "http://custom:9999"
        assert client.timeout == 60
        assert client.model == "test-model"


class TestChatCompletionResponse:
    """Test ChatCompletionResponse dataclass."""

    def test_response_dataclass_fields(self):
        """Test ChatCompletionResponse has correct fields."""
        response = ChatCompletionResponse(
            message="Test message",
            model="test-model",
            done=True,
            prompt_eval_count=10,
            prompt_eval_duration=100,
            eval_count=5,
            eval_duration=50,
        )

        assert response.message == "Test message"
        assert response.model == "test-model"
        assert response.done is True
        assert response.prompt_eval_count == 10
        assert response.prompt_eval_duration == 100
        assert response.eval_count == 5
        assert response.eval_duration == 50

    def test_response_defaults(self):
        """Test ChatCompletionResponse default values."""
        response = ChatCompletionResponse(
            message="Test", model="test-model"
        )

        assert response.done is True
        assert response.prompt_eval_count == 0
        assert response.prompt_eval_duration == 0
        assert response.eval_count == 0
        assert response.eval_duration == 0


class TestEmbeddingResponse:
    """Test EmbeddingResponse dataclass."""

    def test_embedding_dataclass_fields(self):
        """Test EmbeddingResponse has correct fields."""
        response = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3], model="nomic-embed-text"
        )

        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.model == "nomic-embed-text"


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
        assert client.supports_vision("m") is True
        assert client.supports_thinking("m") is True
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

        assert result == VisionChatResult(
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
    def test_http_error_message_preserved(self, mock_request):
        mock_request.return_value = _http(404, {"error": "model 'x' not found"})
        with pytest.raises(requests.HTTPError, match="HTTP 404: model 'x' not found") as exc_info:
            OllamaClient().list_models()
        assert exc_info.value.response.status_code == 404


def test_get_ollama_client_passes_vision_timeouts():
    client = get_ollama_client(generate_timeout=42.0, connect_timeout=2.0)
    assert client.generate_timeout == 42.0
    assert client.connect_timeout == 2.0
