"""Unit tests for lib.retry (fake sleep and jitter; nothing waits)."""

import threading

import pytest

from lib.exceptions import (
    ModelServerError,
    OCRError,
    OllamaConnectionError,
    TextModelUnavailableError,
    TextResponseError,
    TextTimeoutError,
    TextTransientError,
    VisionAnalysisError,
    VisionModelUnavailableError,
    VisionResponseError,
    VisionTimeoutError,
    VisionTransientError,
)
from lib.retry import RetryPolicy, call_with_retry, is_transient_model_error


class Script:
    """A callable that raises/returns the scripted outcomes in order and records attempt numbers."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.attempts: list[int] = []

    def __call__(self, attempt):
        self.attempts.append(attempt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _run(fn, policy, rand=0.0, **kwargs):
    sleeps: list[float] = []
    result = call_with_retry(fn, policy, describe="test", sleep=sleeps.append, rand=lambda: rand, **kwargs)
    return result, sleeps


class TestClassification:
    @pytest.mark.parametrize(
        "error",
        [
            VisionResponseError("x"), VisionTimeoutError("x"), VisionTransientError("x"),
            OllamaConnectionError("x"), TextResponseError("x"), TextTimeoutError("x"), TextTransientError("x"),
        ],
    )
    def test_transient(self, error):
        assert is_transient_model_error(error)

    @pytest.mark.parametrize(
        "error",
        [
            VisionModelUnavailableError("x"), TextModelUnavailableError("x"), ModelServerError("x"),
            VisionAnalysisError("HTTP 400"), OCRError("x"), ValueError("x"), RuntimeError("x"),
        ],
    )
    def test_not_transient(self, error):
        assert not is_transient_model_error(error)


class TestPolicy:
    def test_defaults_and_disabled(self):
        policy = RetryPolicy()
        assert (policy.max_attempts, policy.initial_delay, policy.max_delay, policy.max_timeout_retries) == (
            3, 2.0, 30.0, 1,
        )
        assert RetryPolicy.DISABLED.max_attempts == 1
        assert not RetryPolicy.DISABLED.enabled

    @pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
    def test_invalid_attempts_rejected(self, attempts):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=attempts)


class TestCallWithRetry:
    def test_success_first_time(self):
        (value, attempts), sleeps = _run(Script("ok"), RetryPolicy())
        assert (value, attempts) == ("ok", 1)
        assert sleeps == []

    def test_attempt_numbers_and_success_after_failures(self):
        fn = Script(VisionResponseError("a"), VisionTransientError("b"), "ok")
        (value, attempts), _ = _run(fn, RetryPolicy(max_attempts=3))
        assert (value, attempts) == ("ok", 3)
        assert fn.attempts == [1, 2, 3]

    def test_backoff_sequence_and_cap_without_jitter(self):
        """rand=1.0 gives the full delay: min(max, initial * 2**(k-2)) before attempt k."""
        sleeps = []
        fn = Script(*[VisionTransientError("x")] * 5)
        with pytest.raises(VisionTransientError):
            call_with_retry(
                fn, RetryPolicy(max_attempts=5, initial_delay=2.0, max_delay=5.0),
                describe="t", sleep=sleeps.append, rand=lambda: 1.0,
            )
        assert sleeps == [2.0, 4.0, 5.0, 5.0]
        assert fn.attempts == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(("rand", "expected"), [(0.0, 1.0), (0.5, 1.5), (0.999, 1.999)])
    def test_jitter_bounds(self, rand, expected):
        sleeps = []
        fn = Script(VisionTransientError("x"), "ok")
        call_with_retry(fn, RetryPolicy(initial_delay=2.0), describe="t", sleep=sleeps.append, rand=lambda: rand)
        assert sleeps == [pytest.approx(expected)]
        assert 1.0 <= sleeps[0] < 2.0  # between half and the full delay

    def test_non_retryable_raised_immediately(self):
        error = VisionModelUnavailableError("not installed")
        fn = Script(error, "never")
        with pytest.raises(VisionModelUnavailableError) as excinfo:
            _run(fn, RetryPolicy(max_attempts=5))
        assert excinfo.value is error
        assert fn.attempts == [1]
        assert excinfo.value.attempts == 1

    def test_timeouts_retried_once_only(self):
        fn = Script(VisionTimeoutError("slow"), VisionTimeoutError("slow"), "never")
        with pytest.raises(VisionTimeoutError) as excinfo:
            _run(fn, RetryPolicy(max_attempts=5))
        assert fn.attempts == [1, 2]
        assert excinfo.value.attempts == 2

    def test_timeout_then_other_errors_keep_retrying(self):
        fn = Script(VisionTimeoutError("slow"), VisionResponseError("empty"), "ok")
        (value, attempts), _ = _run(fn, RetryPolicy(max_attempts=3))
        assert (value, attempts) == ("ok", 3)

    def test_original_exception_reraised_with_attempts_and_note(self):
        last = VisionResponseError("third")
        fn = Script(VisionResponseError("first"), VisionResponseError("second"), last)
        with pytest.raises(VisionResponseError) as excinfo:
            _run(fn, RetryPolicy(max_attempts=3))
        assert excinfo.value is last
        assert excinfo.value.attempts == 3
        assert any("gave up after 3 attempt(s)" in note for note in excinfo.value.__notes__)

    def test_disabled_makes_one_attempt(self):
        fn = Script(VisionResponseError("x"), "never")
        with pytest.raises(VisionResponseError) as excinfo:
            _run(fn, RetryPolicy.DISABLED)
        assert fn.attempts == [1]
        assert excinfo.value.attempts == 1

    def test_connection_error_retried_then_propagates(self):
        error = OllamaConnectionError("refused")
        fn = Script(error, error, error)
        with pytest.raises(OllamaConnectionError) as excinfo:
            _run(fn, RetryPolicy(max_attempts=3))
        assert fn.attempts == [1, 2, 3]
        assert excinfo.value is error

    def test_cancel_stops_early(self):
        cancel = threading.Event()

        def fn(attempt):
            if attempt == 2:
                cancel.set()
            raise VisionTransientError(f"attempt {attempt}")

        sleeps = []
        with pytest.raises(VisionTransientError) as excinfo:
            call_with_retry(
                fn, RetryPolicy(max_attempts=5), describe="t", sleep=sleeps.append, rand=lambda: 0.0, cancel=cancel
            )
        assert excinfo.value.attempts == 2
        assert len(sleeps) == 1  # no wait after the cancel
        assert any("cancelled" in note for note in excinfo.value.__notes__)

    def test_cancel_during_backoff_stops_before_next_attempt(self):
        cancel = threading.Event()
        fn = Script(VisionTransientError("x"), "never")
        with pytest.raises(VisionTransientError):
            call_with_retry(
                fn, RetryPolicy(max_attempts=3), describe="t", sleep=lambda _s: cancel.set(), cancel=cancel
            )
        assert fn.attempts == [1]

    def test_custom_predicate(self):
        fn = Script(OCRError("tesseract"), "ok")
        (value, attempts), _ = _run(
            fn, RetryPolicy(max_attempts=2), is_retryable=lambda exc: isinstance(exc, OCRError)
        )
        assert (value, attempts) == ("ok", 2)

    def test_warning_logged_per_failed_attempt(self, caplog):
        fn = Script(VisionTransientError("503"), VisionTransientError("503"), "ok")
        with caplog.at_level("WARNING", logger="lib.retry"):
            _run(fn, RetryPolicy(max_attempts=3))
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 2
        assert "test: attempt 1/3 failed: VisionTransientError: 503; retrying in" in messages[0]
