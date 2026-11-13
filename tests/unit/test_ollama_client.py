"""Unit tests for OllamaClient class.

Tests REST API communication with mocked HTTP requests.
"""

import base64
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lib.ollama_client import (
    ChatCompletionResponse,
    EmbeddingResponse,
    OllamaClient,
    get_ollama_client,
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

    @patch("requests.Session.request")
    def test_pull_model_basic(self, mock_request):
        """Test pulling a model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_request.return_value = mock_response

        client = OllamaClient()
        result = client.pull_model("qwen2-vl:7b")

        assert result is True

    @patch("requests.Session.request")
    def test_pull_model_with_insecure(self, mock_request):
        """Test pulling model with insecure flag."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_request.return_value = mock_response

        client = OllamaClient()
        result = client.pull_model("test-model", insecure=True)

        assert result is True

        # Verify insecure flag was sent
        call_kwargs = mock_request.call_args.kwargs
        json_data = call_kwargs["json"]
        assert json_data["insecure"] is True


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
