"""Custom exception types for bourbon image processor hierarchy."""


class BourbonException(Exception):
    """Base exception for all bourbon processor errors."""

    def __init__(self, message: str, details: str = ""):
        """Initialize bourbon exception.

        Args:
            message: Human-readable error message.
            details: Additional error details or context.
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return full error message with details."""
        if self.details:
            return f"{self.message}\n{self.details}"
        return self.message


class ImageProcessingError(BourbonException):
    """Error during image manipulation or processing."""

    pass


class OCRError(BourbonException):
    """Error during OCR text extraction."""

    pass


class AIModelError(BourbonException):
    """Error during AI/LLM model requests or responses."""

    pass


class OllamaConnectionError(AIModelError):
    """Error connecting to Ollama server."""

    pass


class ModelServerError(AIModelError, RuntimeError):
    """Model server could not be started, reached or verified.

    Also a ``RuntimeError`` so callers that predate this class (and catch
    ``RuntimeError``) keep working.
    """

    pass


class VisionAnalysisError(AIModelError):
    """Vision analysis of one bottle failed (HTTP error, bad or empty response, timeout).

    Per-bottle: the batch records the bottle as failed and moves on.
    """

    pass


class VisionResponseError(VisionAnalysisError):
    """The vision model answered, but the output was unusable.

    Raised for empty, length-truncated or unparseable content, or a reply
    with no identifying field (name, distillery and release all missing).
    """

    pass


class VisionTimeoutError(VisionAnalysisError):
    """The vision model did not respond within the read timeout."""

    pass


class VisionModelUnavailableError(AIModelError):
    """The vision model cannot be used at all on this server.

    Raised when the model is not installed, is incompatible with the running
    server version, or lacks the ``vision`` capability. Fatal for the run:
    every bottle would fail the same way.
    """

    pass


#: Errors after which no further bottle can succeed; batches re-raise these
#: and the CLI aborts the run instead of recording one failure per bottle.
FATAL_MODEL_ERRORS: tuple[type[AIModelError], ...] = (
    VisionModelUnavailableError,
    OllamaConnectionError,
    ModelServerError,
)


class ConfigurationError(BourbonException):
    """Error in configuration loading or validation."""

    pass


class FileSystemError(BourbonException):
    """Error during file read/write or path resolution."""

    pass


class ValidationError(BourbonException):
    """Error during data validation or schema validation."""

    pass


class DuplicateImageError(FileSystemError):
    """Error when duplicate image detected."""

    def __init__(self, image_path: str, existing_path: str):
        """Initialize duplicate image error.

        Args:
            image_path: Path of duplicate image.
            existing_path: Path of existing image.
        """
        message = f"Duplicate image detected: {image_path}"
        details = f"Existing image at: {existing_path}"
        super().__init__(message, details)
        self.image_path = image_path
        self.existing_path = existing_path


class RetryExhaustedError(BourbonException):
    """Error when all retry attempts are exhausted."""

    def __init__(self, message: str, retry_count: int, original_error: Exception):
        """Initialize retry exhausted error.

        Args:
            message: Error message.
            retry_count: Number of retry attempts.
            original_error: Original exception that caused retries.
        """
        details = f"Retry count: {retry_count}\nOriginal error: {type(original_error).__name__}: {str(original_error)}"
        super().__init__(message, details)
        self.retry_count = retry_count
        self.original_error = original_error
