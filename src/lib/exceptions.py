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


class VisionTransientError(VisionAnalysisError):
    """A vision call failed in a way that may succeed when repeated.

    Raised for HTTP 5xx, 408 and 429 answers and for request failures other
    than connect errors and timeouts. Retry policies may repeat the call.
    """

    pass


class VisionModelUnavailableError(AIModelError):
    """The vision model cannot be used at all on this server.

    Raised when the model is not installed, is incompatible with the running
    server version, or lacks the ``vision`` capability. Fatal for the run:
    every bottle would fail the same way.
    """

    pass


class TextAnalysisError(AIModelError):
    """A text-model (fact-check) call failed for one bottle.

    Raised for HTTP 4xx answers (other than 404) and non-JSON bodies. Never
    fatal for the run: the bottle's fact check is recorded as failed.
    """

    pass


class TextResponseError(TextAnalysisError):
    """The text model answered, but the output was unusable (empty, truncated, unparseable)."""

    pass


class TextTimeoutError(TextAnalysisError):
    """The text model did not respond within the read timeout."""

    pass


class TextTransientError(TextAnalysisError):
    """A text call failed in a way that may succeed when repeated (HTTP 5xx/408/429, request error)."""

    pass


class TextModelUnavailableError(AIModelError):
    """The text model cannot be used on this server.

    Raised when the model is not installed, is incompatible with the running
    server version, or lacks the ``completion`` capability. Not listed in
    :data:`FATAL_MODEL_ERRORS`: the fact-check stage reports it per bottle
    and the label-only results are still saved.
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
