"""Error handling types and utilities for bourbon image processor.

Defines error types, retry logic, error reporting, and automatic retry management.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

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

    @classmethod
    def from_dict(cls, data: Dict) -> "ErrorReport":
        """Create ErrorReport from dictionary.

        Args:
            data: Dictionary representation of error report.

        Returns:
            ErrorReport instance.
        """
        return cls(
            error_type=ErrorType[data["error_type"]],
            file_path=Path(data["file_path"]) if data["file_path"] else None,
            message=data["message"],
            details=data["details"],
            timestamp=data.get("timestamp"),
            retry_count=data.get("retry_count", 0),
            hint=data.get("hint"),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        result = f"[{self.error_type.name}] {self.message}"
        if self.file_path:
            result += f"\n  File: {self.file_path}"
        if self.hint:
            result += f"\n  Hint: {self.hint}"
        return result


class RetryException(Exception):
    """Exception thrown when retry fails after all attempts.

    Wraps the original exception with retry context.

    Attributes:
        original_error: The underlying exception that caused the failure.
        retry_count: Number of retry attempts made.
    """

    def __init__(self, message: str, original_error: Exception, retry_count: int):
        """Initialize retry exception.

        Args:
            message: Exception message.
            original_error: Underlying exception.
            retry_count: Number of retries attempted.
        """
        super().__init__(message)
        self.original_error = original_error
        self.retry_count = retry_count


class RetryManager:
    """Manage automatic retry logic with exponential backoff.

    Implements configurable retry strategy: max attempts, backoff delays,
    and error type filtering.

    Attributes:
        max_attempts: Maximum number of attempts (default: 3).
        backoff_delays: List of backoff delays in seconds (default: [0, 2, 5]).
        retry_on: List of error types to retry on (empty = retry all).
        log_retries: Whether to log retry attempts (default: True).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_delays: Optional[List[int]] = None,
        retry_on: Optional[List[ErrorType]] = None,
        log_retries: bool = True,
    ):
        """Initialize retry manager.

        Args:
            max_attempts: Maximum number of attempts.
            backoff_delays: List of backoff delays in seconds.
            retry_on: Error types to retry on (empty = retry all).
            log_retries: Whether to log retry attempts.
        """
        self.max_attempts = max_attempts
        self.backoff_delays = backoff_delays or [0, 2, 5]
        self.retry_on = retry_on or []
        self.log_retries = log_retries

    def should_retry(self, error: Exception) -> bool:
        """Determine if error should trigger retry.

        Args:
            error: Exception that occurred.

        Returns:
            True if error should be retried, False otherwise.
        """
        if not self.retry_on:
            # No filter configured, retry all errors
            return True

        # Check if error matches any configured error type
        error_type = self._classify_error(error)
        return error_type in self.retry_on

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify exception into ErrorType category.

        Args:
            error: Exception to classify.

        Returns:
            Classified ErrorType.
        """
        # Check for custom bourbon processor exceptions first
        error_class_name = error.__class__.__name__

        if error_class_name == "OCRError":
            return ErrorType.OCR_FAILURE
        elif error_class_name in ("AIModelError", "OllamaConnectionError"):
            return ErrorType.AI_MODEL_FAILURE
        elif error_class_name == "ImageProcessingError":
            return ErrorType.IMAGE_PROCESSING
        elif error_class_name == "ConfigurationError":
            return ErrorType.CONFIGURATION_ERROR
        elif error_class_name in ("FileSystemError", "DuplicateImageError"):
            return ErrorType.FILESYSTEM_ERROR
        elif error_class_name == "ValidationError":
            return ErrorType.VALIDATION_ERROR
        # Check for common exception types
        error_name = type(error).__name__

        if error_name in (
            "UnidentifiedImageError",
            "PIL.UnidentifiedImageError",
            "cv2.error",
            "OSError",
        ):
            return ErrorType.IMAGE_PROCESSING
        elif "tesseract" in error_name.lower() or "pytesseract" in error_name.lower():
            return ErrorType.OCR_FAILURE
        elif "requests" in error.__class__.__module__ or error_name == "ConnectionError":
            return ErrorType.NETWORK_ERROR
        elif error_name in ("ValueError", "KeyError", "json.JSONDecodeError"):
            return ErrorType.VALIDATION_ERROR
        elif error_name in ("FileNotFoundError", "PermissionError"):
            return ErrorType.FILESYSTEM_ERROR
        else:
            return ErrorType.UNKNOWN

    def execute_with_retry(
        self,
        func,
        *args,
        error_context: Optional[str] = None,
        **kwargs,
    ):
        """Execute function with automatic retry on failure.

        Args:
            func: Function to execute.
            *args: Positional arguments to pass to func.
            error_context: Context string for error messages.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Function return value.

        Raises:
            RetryException: If all retry attempts fail.
        """
        last_error = None
        retry_count = 0

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                retry_count = attempt

                if attempt < self.max_attempts - 1 and self.should_retry(e):
                    # Retry this error
                    if self.log_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. Retrying..."
                        )

                    backoff = self.backoff_delays[min(attempt, len(self.backoff_delays) - 1)]
                    if backoff > 0:
                        time.sleep(backoff)
                else:
                    # Don't retry or max attempts reached
                    error_type = self._classify_error(e)
                    raise RetryException(
                        f"Failed after {retry_count + 1} attempt(s): {e} ({error_context})",
                        e,
                        retry_count,
                    ) from e

        # Should not reach here, but handle edge case
        raise RetryException(
            f"Failed after {self.max_attempts} attempts: {last_error} ({error_context})",
            last_error,
            self.max_attempts - 1,
        )


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
    from datetime import datetime

    # Classify error type
    error_name = type(error).__name__
    error_class_name = error.__class__.__name__

    # Check for custom bourbon processor exceptions first
    if error_class_name == "OCRError":
        error_type = ErrorType.OCR_FAILURE
        hint = hint or "Ensure Tesseract OCR is installed and in PATH. Check image clarity."
    elif error_class_name == "AIModelError" or error_class_name == "OllamaConnectionError":
        error_type = ErrorType.AI_MODEL_FAILURE
        hint = hint or "Ensure Ollama server is running and the model is available."
    elif error_class_name == "ImageProcessingError":
        error_type = ErrorType.IMAGE_PROCESSING
        hint = hint or "Check if the image file is corrupted or in an unsupported format."
    elif error_class_name == "ConfigurationError":
        error_type = ErrorType.CONFIGURATION_ERROR
        hint = hint or "Check configuration values and ensure config file is valid."
    elif error_class_name == "FileSystemError" or error_class_name == "DuplicateImageError":
        error_type = ErrorType.FILESYSTEM_ERROR
        hint = hint or "Verify file path exists and you have read/write permissions."
    elif error_class_name == "ValidationError":
        error_type = ErrorType.VALIDATION_ERROR
        hint = hint or "Check the input data format or model response schema."
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
        timestamp=datetime.utcnow().isoformat(),
        hint=hint,
    )


def save_error_reports(reports: List[ErrorReport], output_path: Path) -> None:
    """Save error reports to JSON file.

    Args:
        reports: List of error reports.
        output_path: Output file path (JSON format).
    """
    import json

    data = [report.to_dict() for report in reports]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(reports)} error report(s) to {output_path}")


def load_error_reports(input_path: Path) -> List[ErrorReport]:
    """Load error reports from JSON file.

    Args:
        input_path: Input file path (JSON format).

    Returns:
        List of error reports.

    Raises:
        FileNotFoundError: If file not found.
        json.JSONDecodeError: If file is invalid JSON.
    """
    import json

    with open(input_path) as f:
        data = json.load(f)

    return [ErrorReport.from_dict(item) for item in data]
