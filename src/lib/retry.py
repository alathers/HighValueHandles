"""One retry policy for model requests and OCR.

:func:`call_with_retry` runs a callable until it succeeds, the error is not
worth repeating, or the policy's attempts are used up. It replaces the old
retry manager; every layer that retries (OCR per photo, the vision request,
later the web and fact-check stages) uses the same :class:`RetryPolicy`.

Accuracy notes:

- The original exception is re-raised on final failure, never wrapped, so
  callers and failure reports see the real error type. It carries
  ``exc.attempts`` (the number of attempts made) and a note.
- A read timeout is repeated at most ``max_timeout_retries`` times: a model
  that timed out twice will usually time out again, and each wait can be
  minutes long.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from lib.exceptions import (
    OllamaConnectionError,
    TextResponseError,
    TextTimeoutError,
    TextTransientError,
    VisionResponseError,
    VisionTimeoutError,
    VisionTransientError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Errors that may pass when the same request is made again.
TRANSIENT_MODEL_ERRORS: tuple[type[BaseException], ...] = (
    VisionResponseError,
    VisionTimeoutError,
    VisionTransientError,
    OllamaConnectionError,
    TextResponseError,
    TextTimeoutError,
    TextTransientError,
)

#: Errors counted against ``RetryPolicy.max_timeout_retries``.
TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (VisionTimeoutError, TextTimeoutError, TimeoutError)


@dataclass(frozen=True)
class RetryPolicy:
    """How often, and how patiently, a failed request is repeated.

    Attributes:
        max_attempts: Total attempts, at least 1 (1 = no retry).
        initial_delay: Seconds before the second attempt (before jitter).
        max_delay: Cap on the delay before any attempt (before jitter).
        max_timeout_retries: How many timeouts may be retried in one call.
    """

    max_attempts: int = 3
    initial_delay: float = 2.0
    max_delay: float = 30.0
    max_timeout_retries: int = 1

    #: One attempt, no retry (``--no-retry``).
    DISABLED: ClassVar[RetryPolicy]

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError(f"max_attempts must be an int >= 1, got {self.max_attempts!r}")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays must be >= 0")
        if self.max_timeout_retries < 0:
            raise ValueError("max_timeout_retries must be >= 0")

    @property
    def enabled(self) -> bool:
        """Whether more than one attempt is allowed."""
        return self.max_attempts > 1

    def delay_before(self, attempt: int, rand: float) -> float:
        """Backoff before ``attempt`` (2, 3, ...), with ``rand`` in [0, 1) as jitter.

        ``min(max_delay, initial_delay * 2**(attempt-2)) * (0.5 + 0.5*rand)``.
        """
        base = min(self.max_delay, self.initial_delay * 2 ** max(0, attempt - 2))
        return base * (0.5 + 0.5 * rand)


RetryPolicy.DISABLED = RetryPolicy(max_attempts=1)


def is_transient_model_error(exc: BaseException) -> bool:
    """True for model errors worth repeating (unusable output, timeouts, 5xx/408/429, refused connections).

    Never true for a model that cannot be used at all (``*ModelUnavailableError``),
    a server that could not be started (``ModelServerError``) or a plain
    ``VisionAnalysisError`` (an HTTP 4xx answer): repeating those cannot help.
    """
    return isinstance(exc, TRANSIENT_MODEL_ERRORS)


def call_with_retry(
    fn: Callable[[int], T],
    policy: RetryPolicy,
    *,
    describe: str,
    is_retryable: Callable[[BaseException], bool] = is_transient_model_error,
    sleep: Callable[[float], object] = time.sleep,
    rand: Callable[[], float] = random.random,
    cancel: threading.Event | None = None,
) -> tuple[T, int]:
    """Call ``fn(attempt)`` until it succeeds or the policy gives up.

    Args:
        fn: Called with the attempt number (1, 2, ...).
        policy: Attempts, backoff and the timeout-retry cap.
        describe: What is being attempted, for log lines and the error note.
        is_retryable: Whether an exception may be repeated.
        sleep: Sleep function (injected in tests).
        rand: Jitter source returning [0, 1) (injected in tests).
        cancel: When set, no further attempt is started.

    Returns:
        ``(value, attempts_used)``.

    Raises:
        Exception: The last attempt's original exception, with
            ``exc.attempts`` set and a note added.
    """
    attempt = 0
    timeouts = 0
    while True:
        attempt += 1
        try:
            return fn(attempt), attempt
        except Exception as exc:
            is_timeout = isinstance(exc, TIMEOUT_ERRORS)
            if is_timeout:
                timeouts += 1
            reason = None
            if not is_retryable(exc):
                reason = "not retryable"
            elif attempt >= policy.max_attempts:
                reason = "no attempts left"
            elif is_timeout and timeouts > policy.max_timeout_retries:
                reason = "timeout retry limit reached"
            elif cancel is not None and cancel.is_set():
                reason = "cancelled"
            if reason is None:
                delay = policy.delay_before(attempt + 1, rand())
                logger.warning(
                    "%s: attempt %d/%d failed: %s: %s; retrying in %.1fs",
                    describe, attempt, policy.max_attempts, type(exc).__name__, exc, delay,
                )
                if delay > 0:
                    sleep(delay)
                if cancel is None or not cancel.is_set():
                    continue
                reason = "cancelled"
            exc.attempts = attempt  # type: ignore[attr-defined]
            exc.add_note(f"{describe}: gave up after {attempt} attempt(s) ({reason})")
            raise
