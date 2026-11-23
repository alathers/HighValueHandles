"""Error handling types and utilities for bourbon image processor.

Defines error types and structured error reports. Retries live in :mod:`lib.retry`.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Tuple

from lib.exceptions import (
    AIModelError,
    ConfigurationError,
    FileSystemError,
    ImageProcessingError,
    OCRError,
    ValidationError,
    VisionModelUnavailableError,
    VisionResponseError,
    VisionTimeoutError,
)

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Categorized error types for structured error handling."""

    IMAGE_PROCESSING = auto()
    """PIL/OpenCV image manipulation failures."""

    OCR_FAILURE = auto()
    """Tesseract OCR extraction failures."""

    AI_MODEL_FAILURE = auto()
    """Ollama/LLM model request failures."""

    CONFIGURATION_ERROR = auto()
    """Configuration validation or loading failures."""

    FILESYSTEM_ERROR = auto()
    """File read/write or path resolution failures."""

    NETWORK_ERROR = auto()
    """Ollama API connection failures."""

    VALIDATION_ERROR = auto()
    """Data validation or schema validation failures."""

    UNKNOWN = auto()
    """Unclassified or unexpected errors."""


@dataclass
class ErrorReport:
    """Structured error report for tracking failed operations.

    Attributes:
        error_type: Categorized error type.
        file_path: Source file path if applicable.
        message: Human-readable error message.
        details: Additional error details or traceback.
        timestamp: When the error occurred (ISO format).
        retry_count: Number of retry attempts made.
        hint: Suggested fix or next step.
    """

    error_type: ErrorType
    file_path: Optional[Path]
    message: str
    details: str
    timestamp: Optional[str] = None
    retry_count: int = 0
    hint: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of error report.
        """
        return {
            "error_type": self.error_type.name,
            "file_path": str(self.file_path) if self.file_path else None,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "hint": self.hint,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        result = f"[{self.error_type.name}] {self.message}"
        if self.file_path:
            result += f"\n  File: {self.file_path}"
        if self.hint:
            result += f"\n  Hint: {self.hint}"
        return result


def _classify_bourbon_exception(error: Exception) -> Optional[Tuple[ErrorType, str]]:
    """Classify this package's own exceptions by type, subclasses included.

    Uses ``isinstance`` so subclasses (for example the ``Vision*`` errors and
    ``ModelServerError`` under ``AIModelError``) get their parent's category
    instead of UNKNOWN.

    Args:
        error: Exception to classify.

    Returns:
        ``(error_type, default_hint)``, or None if ``error`` is not one of the
        package's exception types.
    """
    if isinstance(error, OCRError):
        return ErrorType.OCR_FAILURE, "Ensure Tesseract OCR is installed and in PATH. Check image clarity."
    if isinstance(error, VisionModelUnavailableError):
        return (
            ErrorType.AI_MODEL_FAILURE,
            (
                "Check the model is installed and supports images ('ollama show <model>'), "
                "or choose another vision model with -m."
            ),
        )
    if isinstance(error, VisionTimeoutError):
        return (
            ErrorType.AI_MODEL_FAILURE,
            "The vision model was too slow; raise --request-timeout or use a smaller model.",
        )
    if isinstance(error, VisionResponseError):
        return (
            ErrorType.AI_MODEL_FAILURE,
            "The vision model returned no usable data; check the photos show a readable label.",
        )
    if isinstance(error, AIModelError):
        return ErrorType.AI_MODEL_FAILURE, "Ensure Ollama server is running and the model is available."
    if isinstance(error, ImageProcessingError):
        return ErrorType.IMAGE_PROCESSING, "Check if the image file is corrupted or in an unsupported format."
    if isinstance(error, ConfigurationError):
        return ErrorType.CONFIGURATION_ERROR, "Check configuration values and ensure config file is valid."
    if isinstance(error, FileSystemError):
        return ErrorType.FILESYSTEM_ERROR, "Verify file path exists and you have read/write permissions."
    if isinstance(error, ValidationError):
        return ErrorType.VALIDATION_ERROR, "Check the input data format or model response schema."
    return None


def create_error_report(
    error: Exception,
    file_path: Optional[Path] = None,
    context: Optional[str] = None,
    hint: Optional[str] = None,
) -> ErrorReport:
    """Create ErrorReport from exception.

    Args:
        error: Exception that occurred.
        file_path: Source file path if applicable.
        context: Additional context for error message.
        hint: Suggested fix or next step.

    Returns:
        ErrorReport instance.
    """
    from datetime import UTC, datetime

    # Classify error type: the package's own exceptions first (subclasses included)
    error_name = type(error).__name__
    classified = _classify_bourbon_exception(error)

    if classified is not None:
        error_type = classified[0]
        hint = hint or classified[1]
    # Check for standard Python exceptions
    elif error_name in ("UnidentifiedImageError", "PIL.UnidentifiedImageError", "cv2.error"):
        error_type = ErrorType.IMAGE_PROCESSING
        hint = hint or "Check if the image file is corrupted or in an unsupported format."
    elif "tesseract" in error_name.lower() or "pytesseract" in error_name.lower():
        error_type = ErrorType.OCR_FAILURE
        hint = hint or "Ensure Tesseract OCR is installed and in PATH. Check image clarity."
    elif "requests" in error.__class__.__module__ or error_name == "ConnectionError":
        error_type = ErrorType.NETWORK_ERROR
        hint = hint or "Ensure Ollama server is running on the configured URL."
    elif error_name in ("ValueError", "KeyError", "json.JSONDecodeError"):
        error_type = ErrorType.VALIDATION_ERROR
        hint = hint or "Check the input data format or model response schema."
    elif error_name in ("FileNotFoundError", "PermissionError"):
        error_type = ErrorType.FILESYSTEM_ERROR
        hint = hint or "Verify file path exists and you have read/write permissions."
    else:
        error_type = ErrorType.UNKNOWN
        hint = hint or "Check logs for more details."

    # Build error message
    message = f"{type(error).__name__}: {str(error)}"
    if context:
        message = f"{context}: {message}"

    # Build details
    details = str(error)
    if hasattr(error, "__traceback__"):
        import traceback

        details += "\n\n" + "".join(traceback.format_exception(type(error), error, error.__traceback__))

    return ErrorReport(
        error_type=error_type,
        file_path=file_path,
        message=message,
        details=details,
        timestamp=datetime.now(UTC).isoformat(),
        hint=hint,
    )
