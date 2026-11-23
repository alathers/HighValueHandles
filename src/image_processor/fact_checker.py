"""The gpt-oss fact check of one bottle: EXTRACT -> verify -> reconcile -> COMPOSE -> gate.

The text model only proposes; ``fact_verify`` checks every proposal against
the exact source text and ``fact_reconcile`` decides values, conflicts,
confidence and the publish gate. :meth:`FactChecker.check` never raises: any
failure is recorded in ``verification`` and the bottle keeps its label/OCR
facts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from image_processor.fact_prompt import (
    COMPOSE_SYSTEM,
    EXTRACT_SYSTEM,
    PROMPT_VERSION,
    TEXT_OPTIONS,
    EvidencePack,
    ExtractReply,
    build_compose_prompt,
    build_compose_schema,
    build_evidence_pack,
    build_extract_prompt,
    build_extract_schema,
    number_fields,
    parse_compose_reply,
    parse_extract_reply,
    parse_json_reply,
)
from image_processor.fact_reconcile import apply_gate, empty_verification, reconcile
from image_processor.fact_verify import (
    Accepted,
    AcceptedPrice,
    Rejected,
    as_page_match_map,
    label_release_tokens,
    page_matches,
    render_description,
    verify_claim,
    verify_copy,
    verify_price,
)
from lib.exceptions import (
    OllamaConnectionError,
    TextAnalysisError,
    TextModelUnavailableError,
    TextResponseError,
    TextTimeoutError,
    TextTransientError,
)
from models.facts import FACT_FIELDS, FactInput, FactSheet, FieldFact, PublishCopy, Verification

logger = logging.getLogger(__name__)

#: Retries never exceed this many calls per stage.
MAX_ATTEMPTS_CAP = 2
_THINK_ORDER = ("low", "medium", "high")

__all__ = ["FactChecker", "FactInput", "TextLLM"]


class TextLLM(Protocol):
    """What the fact check needs from a text model client (``OllamaClient``)."""

    def chat_text(
        self,
        prompt: str,
        model: str | None = None,
        *,
        system: str | None = None,
        format: Any | None = None,
        think: Any | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: Any | None = "5m",
        read_timeout: float | None = None,
    ) -> Any: ...


class _StageFailed(Exception):
    """A text call failed for this bottle (recorded, never raised from check())."""

    def __init__(self, reason: str, attempts: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts


class _Disabled(Exception):
    """The text model is unusable for the rest of the run."""


@dataclass
class _CallResult:
    value: Any
    attempts: int
    think: str | None
    degraded: bool


def _lower(think: str | None) -> str | None:
    if think in _THINK_ORDER and think != "low":
        return _THINK_ORDER[_THINK_ORDER.index(think) - 1]
    return None


class FactChecker:
    """Runs the fact check for one bottle at a time (serially).

    Args:
        llm: Text model client (``OllamaClient``); None means deterministic only.
        model: Text model name.
        think_extract: Think level of EXTRACT.
        think_compose: Think level of COMPOSE.
        num_ctx: Context window (kept identical on every call).
        timeout_s: Read timeout of one call.
        max_attempts: Calls per stage (capped at 2; 1 = no retry).
        keep_alive: How long Ollama keeps the model loaded between calls.
        usage_recorder: Called after each call with
            ``(model, duration_s, success, think)``.
    """

    def __init__(
        self,
        llm: TextLLM | None,
        *,
        model: str,
        think_extract: str = "high",
        think_compose: str = "medium",
        num_ctx: int = 32768,
        timeout_s: float = 900.0,
        max_attempts: int = 2,
        keep_alive: Any = "10m",
        usage_recorder: Callable[[str, float, bool, str | None], None] | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.think_extract = think_extract
        self.think_compose = think_compose
        self.num_ctx = int(num_ctx)
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS_CAP))
        self.keep_alive = keep_alive
        self.usage_recorder = usage_recorder
        self.disabled_reason: str | None = None
        self.options: dict[str, Any] = {**TEXT_OPTIONS, "num_ctx": self.num_ctx}

    # ------------------------------------------------------------------
    # Deterministic mode
    # ------------------------------------------------------------------

    @staticmethod
    def deterministic(inp: FactInput, reason: str, *, status: str = "skipped", model: str | None = None,
                      duration_s: float = 0.0, attempts: int = 0, rejected: list | None = None) -> FactSheet:
        """Label, OCR and user facts only; web pages are listed as sources but fill no field."""
        pack = build_evidence_pack(inp)
        verification = empty_verification(status, reason, model)
        verification.duration_s = round(duration_s, 3)
        verification.attempts = attempts
        verification.rejected = list(rejected or [])
        return reconcile(inp, pack, verification)

    # ------------------------------------------------------------------
    # Model calls
    # ------------------------------------------------------------------

    def _record(self, duration: float, success: bool, think: str | None) -> None:
        if self.usage_recorder is None:
            return
        try:
            self.usage_recorder(self.model, duration, success, think)
        except Exception as e:
            logger.debug("usage recorder failed: %s", e)

    def _call(self, stage: str, prompt: str, system: str, schema: dict[str, Any], think: str,
              validate: Callable[[dict[str, Any] | None], Any]) -> _CallResult:
        assert self.llm is not None
        attempts = 0
        level: str | None = think
        degraded = False
        last = "no reply"
        while attempts < self.max_attempts:
            attempts += 1
            start = time.monotonic()
            try:
                result = self.llm.chat_text(
                    prompt,
                    self.model,
                    system=system,
                    format=schema,
                    think=level,
                    options=dict(self.options),
                    keep_alive=self.keep_alive,
                    read_timeout=self.timeout_s,
                )
            except TextTimeoutError as e:
                self._record(time.monotonic() - start, False, level)
                raise _StageFailed(f"{stage} timed out after {self.timeout_s:.0f}s: {e}", attempts) from e
            except TextModelUnavailableError as e:
                self._record(time.monotonic() - start, False, level)
                self.disabled_reason = f"text model unavailable: {e}"
                raise _Disabled(self.disabled_reason) from e
            except OllamaConnectionError as e:
                self._record(time.monotonic() - start, False, level)
                raise _StageFailed("text model server unreachable", attempts) from e
            except (TextTransientError, TextResponseError) as e:
                self._record(time.monotonic() - start, False, level)
                last = f"{type(e).__name__}: {e}"
                logger.warning("%s attempt %d failed: %s", stage, attempts, last)
                continue
            except TextAnalysisError as e:
                self._record(time.monotonic() - start, False, level)
                raise _StageFailed(f"{stage} failed: {e}", attempts) from e
            duration = time.monotonic() - start
            content = getattr(result, "content", "") or ""
            if getattr(result, "done_reason", None) == "length":
                self._record(duration, False, level)
                last = f"{stage} output cut off (done_reason=length) at think={level}"
                lower = _lower(level)
                if lower is not None and not degraded:
                    level = lower
                    degraded = True
                logger.warning("%s; retrying at think=%s", last, level)
                continue
            if not content.strip():
                self._record(duration, False, level)
                last = f"{stage} reply was empty"
                logger.warning("%s (attempt %d)", last, attempts)
                continue
            parsed = validate(parse_json_reply(content))
            if parsed is None:
                self._record(duration, False, level)
                last = f"{stage} reply did not match the schema"
                logger.warning("%s (attempt %d)", last, attempts)
                continue
            self._record(duration, True, level)
            return _CallResult(parsed, attempts, level, degraded)
        raise _StageFailed(f"{last} ({attempts} attempt(s))", attempts)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self, inp: FactInput) -> FactSheet:
        """Fact-check one bottle. Never raises."""
        start = time.monotonic()
        try:
            if self.llm is None:
                return self.deterministic(inp, "text verification disabled (no text model)")
            if self.disabled_reason:
                return self.deterministic(inp, self.disabled_reason, model=self.model)
            return self._check(inp, start)
        except _Disabled as e:
            logger.warning("Fact check disabled for the rest of the run: %s", e)
            return self._safe_deterministic(inp, str(e), "skipped", start, 0)
        except Exception as e:
            logger.exception("Fact check failed unexpectedly for %s", getattr(inp, "bottle_id", "?"))
            return self._safe_deterministic(inp, f"unexpected error: {type(e).__name__}: {e}", "failed", start, 0)

    def _safe_deterministic(self, inp: FactInput, reason: str, status: str, start: float, attempts: int) -> FactSheet:
        try:
            return self.deterministic(
                inp, reason, status=status, model=self.model, duration_s=time.monotonic() - start, attempts=attempts
            )
        except Exception as e:
            logger.exception("Deterministic fact sheet failed for %s", getattr(inp, "bottle_id", "?"))
            sheet = FactSheet(
                fields={f: FieldFact.unknown("fact sheet could not be built") for f in FACT_FIELDS},
                sources=[],
                conflicts=[],
                prices=[],
                verification=Verification(
                    status="failed",
                    reason=f"{reason}; deterministic fallback failed: {type(e).__name__}: {e}",
                    text_model=self.model,
                    think=None,
                    prompt_version=PROMPT_VERSION,
                ),
                copy=None,
            )
            return apply_gate(sheet, list(getattr(inp, "group_warnings", []) or []))

    def _check(self, inp: FactInput, start: float) -> FactSheet:
        pack = build_evidence_pack(inp)
        rejected: list[dict[str, Any]] = [
            {"stage": "pack", "kind": "page", "source_id": d.get("source_id"), "reason": "dropped",
             "detail": d.get("reason")}
            for d in pack.dropped
        ]
        attempts = 0
        think_used: str | None = None
        degraded = False

        # EXTRACT
        if pack.quotable_ids:
            schema = build_extract_schema(pack)
            prompt = build_extract_prompt(pack, schema)
            try:
                res = self._call("EXTRACT", prompt, EXTRACT_SYSTEM, schema, self.think_extract, parse_extract_reply)
            except _StageFailed as e:
                return self.deterministic(
                    inp, e.reason, status="failed", model=self.model,
                    duration_s=time.monotonic() - start, attempts=e.attempts, rejected=rejected,
                )
            attempts += res.attempts
            think_used = res.think
            degraded = res.degraded
            reply: ExtractReply = res.value
        else:
            reply = ExtractReply()

        accepted, accepted_prices, rej = self._verify(reply, pack)
        rejected.extend(rej)

        verification = Verification(
            status="verified",
            reason=None if pack.quotable_ids else "no OCR or web text to check; label reading only",
            text_model=self.model,
            think=think_used,
            prompt_version=PROMPT_VERSION,
            duration_s=0.0,
            attempts=attempts,
            degraded_think=degraded,
            rejected=rejected,
        )
        sheet = reconcile(inp, pack, verification, accepted=accepted, accepted_prices=accepted_prices)

        # COMPOSE
        numbered = number_fields(sheet)
        name_ok = sheet.fields["name"].status in ("supported", "user")
        if name_ok and len(numbered) >= 2:
            schema = build_compose_schema(numbered)
            prompt = build_compose_prompt(numbered, schema)
            try:
                res = self._call("COMPOSE", prompt, COMPOSE_SYSTEM, schema, self.think_compose, parse_compose_reply)
                attempts += res.attempts
                check = verify_copy(res.value, numbered)
                if check.ok:
                    sheet.copy = PublishCopy(sentences=check.sentences, description_md=render_description(check.sentences))
                else:
                    sheet.copy = None
                    verification.rejected.extend(check.failures)
                    verification.status = "partial"
                    verification.reason = check.reason
            except _StageFailed as e:
                attempts += e.attempts
                sheet.copy = None
                verification.status = "partial"
                verification.reason = f"description not written: {e.reason}"
            except _Disabled as e:
                # EXTRACT results stay; later bottles go deterministic.
                logger.warning("Fact check disabled for the rest of the run: %s", e)
                attempts += 1
                sheet.copy = None
                verification.status = "partial"
                verification.reason = f"description not written: {e}"
        verification.attempts = attempts
        verification.duration_s = round(time.monotonic() - start, 3)
        return apply_gate(sheet, list(inp.group_warnings))

    @staticmethod
    def _verify(reply: ExtractReply, pack: EvidencePack) -> tuple[list[Accepted], list[AcceptedPrice], list[dict[str, Any]]]:
        rejected: list[dict[str, Any]] = [
            {"stage": "extract", "kind": m.get("kind"), "reason": "malformed", "detail": str(m.get("item"))[:300]}
            for m in reply.malformed
        ]
        pm = as_page_match_map(reply.page_match)
        for sid, match in pm.items():
            if pack.get(sid) is None or pack.get(sid).kind != "web":
                rejected.append(Rejected(match, "unknown_source", f"{sid} is not a web source").to_record())
            elif match.verdict == "exact":
                ok, why = page_matches(sid, pack, pm)
                if not ok:
                    rejected.append(Rejected(match, "page_not_matched", why).to_record())
        release = label_release_tokens(pack.label)
        accepted: list[Accepted] = []
        for claim in reply.claims:
            result = verify_claim(claim, pack, pm, release)
            if isinstance(result, Accepted):
                accepted.append(result)
            else:
                rejected.append(result.to_record())
        prices: list[AcceptedPrice] = []
        for price in reply.prices:
            presult = verify_price(price, pack, pm, release)
            if isinstance(presult, AcceptedPrice):
                prices.append(presult)
                if presult.downgraded:
                    rejected.append(
                        Rejected(price, "msrp_downgraded_to_retail", "no MSRP wording on a retailer page").to_record()
                    )
            else:
                rejected.append(presult.to_record())
        return accepted, prices, rejected
