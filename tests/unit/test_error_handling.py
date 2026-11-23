"""Unit tests for error classification in lib.error_handling."""

import pytest
import requests

from lib.error_handling import ErrorType, RetryManager, create_error_report
from lib.exceptions import (
    AIModelError,
    BourbonException,
    ConfigurationError,
    DuplicateImageError,
    FileSystemError,
    ImageProcessingError,
    ModelServerError,
    OCRError,
    OllamaConnectionError,
    RetryExhaustedError,
    ValidationError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionResponseError,
    VisionTimeoutError,
)


@pytest.mark.parametrize(
    "error,expected",
    [
        (OCRError("x"), ErrorType.OCR_FAILURE),
        (AIModelError("x"), ErrorType.AI_MODEL_FAILURE),
        (OllamaConnectionError("x"), ErrorType.AI_MODEL_FAILURE),
        # Subclasses are classified by their parent, not reported as UNKNOWN.
        (ModelServerError("x"), ErrorType.AI_MODEL_FAILURE),
        (VisionAnalysisError("x"), ErrorType.AI_MODEL_FAILURE),
        (VisionResponseError("x"), ErrorType.AI_MODEL_FAILURE),
        (VisionTimeoutError("x"), ErrorType.AI_MODEL_FAILURE),
        (VisionModelUnavailableError("x"), ErrorType.AI_MODEL_FAILURE),
        (ImageProcessingError("x"), ErrorType.IMAGE_PROCESSING),
        (ConfigurationError("x"), ErrorType.CONFIGURATION_ERROR),
        (FileSystemError("x"), ErrorType.FILESYSTEM_ERROR),
        (DuplicateImageError("a.jpg", "b.jpg"), ErrorType.FILESYSTEM_ERROR),
        (ValidationError("x"), ErrorType.VALIDATION_ERROR),
        (RetryExhaustedError("x", 2, ValueError("y")), ErrorType.UNKNOWN),
        (BourbonException("x"), ErrorType.UNKNOWN),
        (requests.ConnectionError("x"), ErrorType.NETWORK_ERROR),
        (ValueError("x"), ErrorType.VALIDATION_ERROR),
        (FileNotFoundError("x"), ErrorType.FILESYSTEM_ERROR),
        (RuntimeError("x"), ErrorType.UNKNOWN),
    ],
)
def test_classification_matches_in_both_paths(error, expected):
    assert RetryManager()._classify_error(error) is expected
    assert create_error_report(error).error_type is expected


def test_vision_error_hints_are_specific():
    assert "--request-timeout" in create_error_report(VisionTimeoutError("slow")).hint
    assert "ollama show" in create_error_report(VisionModelUnavailableError("missing")).hint
    assert "Ollama server is running" in create_error_report(ModelServerError("down")).hint


def test_explicit_hint_wins():
    assert create_error_report(VisionTimeoutError("slow"), hint="custom").hint == "custom"


def test_retry_filter_uses_subclass_classification():
    manager = RetryManager(retry_on=[ErrorType.AI_MODEL_FAILURE])
    assert manager.should_retry(VisionResponseError("empty"))
    assert not manager.should_retry(OCRError("x"))
