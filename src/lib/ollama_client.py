"""Ollama REST API client wrapper.

Provides a Pythonic interface to Ollama's local LLM server.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


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
        timeout: Request timeout in seconds (default: 30)
        model: Default model to use for requests
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30, model: Optional[str] = None):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL.
            timeout: Request timeout in seconds.
            model: Default model to use.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint path (without base URL).
            method: HTTP method (GET, POST, etc.).
            data: Form data.
            files: File uploads.
            json_data: JSON data.

        Returns:
            JSON response.

        Raises:
            requests.RequestException: If request fails.
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug("Ollama API request: %s %s", method, endpoint)

        response = self.session.request(
            method, url, data=data, files=files, json=json_data, timeout=self.timeout
        )
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
        format: Optional[str] = "json",
    ) -> str:
        """Extract text from image using vision model.

        Args:
            image_path: Path to image file.
            prompt: Text prompt for the model.
            model: Vision model name (uses instance default if None).
            detail: Image detail level ('low', 'medium', 'high').
            format: Response format ('json' for structured output).

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
        import base64

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

    def pull_model(self, model: str, insecure: bool = False) -> bool:
        """Pull/download model from Ollama registry.

        Args:
            model: Model name to pull.
            insecure: Allow insecure registry connections.

        Returns:
            True if successful.

        Raises:
            requests.RequestException: If request fails.
        """
        payload = {
            "name": model,
            "insecure": insecure,
            "stream": False,
        }

        self._make_request("/api/pull", json_data=payload)
        return True

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
) -> OllamaClient:
    """Create Ollama client instance.

    Args:
        base_url: Ollama server URL.
        timeout: Request timeout in seconds.
        model: Default model to use.

    Returns:
        Configured OllamaClient instance.
    """
    return OllamaClient(base_url=base_url, timeout=timeout, model=model)
