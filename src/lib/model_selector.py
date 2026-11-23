"""Model selector for the bourbon image processor pipeline.

Maps each pipeline stage that calls an Ollama model to exactly one model and
records every call for the run's usage summary. There is no fallback: a
silent switch to another model would hide which model produced a fact, so a
stage either uses its configured model or fails.

Stages:

    Stage                 | Default model          | Role
    ----------------------|------------------------|------------------------------------
    VISION_ANALYSIS       | DEFAULT_VISION_MODEL   | reads label fields from the photos
    METADATA_GENERATION   | gpt-oss:120b           | text-only fact extraction from OCR
                          |                        | and web evidence, verified copy

:func:`active_stages` says which stages a run calls (the text stage only
when the fact check is enabled); pre-flight requires (and may pull) only
their models, each checked for the capability in :data:`STAGE_CAPABILITY`.

Steps that never call a model (OCR with Tesseract, capture-time grouping,
quantity from duplicate photos, error classification, preview rendering) are
not stages here.

Usage::

    from lib.model_selector import ModelSelector, ModelStage

    selector = ModelSelector.from_config(args)        # args.model -> vision
    model_name = selector.get_model(ModelStage.VISION_ANALYSIS)
    selector.record_usage(ModelStage.VISION_ANALYSIS, model_name, duration=12.3)
"""

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from lib.constants import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL
from lib.model_availability import normalize_model_name

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_VISION_MODEL",
    "STAGE_CAPABILITY",
    "ModelSelector",
    "ModelSpec",
    "ModelStage",
    "ModelUsage",
    "UsageLog",
    "active_stages",
]


# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------

class ModelStage(Enum):
    """Pipeline stages that call an Ollama model."""

    VISION_ANALYSIS = auto()
    """Vision model reads label fields from a bottle's photos."""

    METADATA_GENERATION = auto()
    """Text model: fact extraction from OCR/web evidence and verified description."""


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """The model assigned to one stage.

    Attributes:
        name: Ollama model tag (e.g. "qwen3.6:35b-a3b").
        stage: Pipeline stage this model serves.
        description: Human-readable description of what this model does.
    """

    name: str
    stage: ModelStage
    description: str = ""


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    """Tracks a single model invocation.

    Attributes:
        stage: Pipeline stage invoked.
        model: Model name used.
        success: Whether the call succeeded.
        duration_seconds: Time taken (0.0 if not recorded).
        input_tokens: Tokens consumed (0 if unknown).
        output_tokens: Tokens produced (0 if unknown).
        error: Error message if failed.
        think: Think level sent with the call (None if not sent).
    """

    stage: ModelStage
    model: str
    success: bool = True
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    think: Optional[str] = None


@dataclass
class UsageLog:
    """Accumulates usage records for reporting. Thread-safe.

    Parallel workers may record at the same time; appends and every summary
    run under one lock and summaries read a snapshot.

    Attributes:
        records: List of ModelUsage records.
    """

    records: list[ModelUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, stage: ModelStage, model: str, success: bool = True,
               duration: float = 0.0, input_tokens: int = 0,
               output_tokens: int = 0, error: Optional[str] = None,
               think: Optional[str] = None) -> None:
        """Record a single model invocation.

        Args:
            stage: Pipeline stage.
            model: Model name used.
            success: Whether the call succeeded.
            duration: Time taken in seconds.
            input_tokens: Input token count.
            output_tokens: Output token count.
            error: Error message if failed.
            think: Think level sent with the call, if any.
        """
        usage = ModelUsage(
            stage=stage,
            model=model,
            success=success,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            think=think,
        )
        with self._lock:
            self.records.append(usage)

    def snapshot(self) -> list[ModelUsage]:
        """A copy of the records, taken under the lock."""
        with self._lock:
            return list(self.records)

    @property
    def total_calls(self) -> int:
        """Total number of model invocations."""
        return len(self.snapshot())

    @property
    def successful_calls(self) -> int:
        """Number of successful model invocations."""
        return sum(1 for r in self.snapshot() if r.success)

    @property
    def total_duration(self) -> float:
        """Total time spent in model calls."""
        return sum(r.duration_seconds for r in self.snapshot())

    def summary_by_stage(self) -> dict[str, dict[str, int]]:
        """Get usage summary grouped by stage.

        Returns:
            Dictionary mapping stage names to {total, success}.
        """
        summary: dict[str, dict[str, int]] = {}
        for r in self.snapshot():
            stats = summary.setdefault(r.stage.name, {"total": 0, "success": 0})
            stats["total"] += 1
            if r.success:
                stats["success"] += 1
        return summary

    def summary_by_model(self) -> dict[str, dict[str, int]]:
        """Get usage summary grouped by model name.

        Returns:
            Dictionary mapping model names to {total, success}.
        """
        summary: dict[str, dict[str, int]] = {}
        for r in self.snapshot():
            stats = summary.setdefault(r.model, {"total": 0, "success": 0})
            stats["total"] += 1
            if r.success:
                stats["success"] += 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with summary statistics and per-record details.
        """
        records = self.snapshot()
        by_stage: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        for r in records:
            for key, summary in ((r.stage.name, by_stage), (r.model, by_model)):
                stats = summary.setdefault(key, {"total": 0, "success": 0})
                stats["total"] += 1
                if r.success:
                    stats["success"] += 1
        return {
            "total_calls": len(records),
            "successful_calls": sum(1 for r in records if r.success),
            "total_duration_seconds": round(sum(r.duration_seconds for r in records), 2),
            "by_stage": by_stage,
            "by_model": by_model,
            "records": [
                {
                    "stage": r.stage.name,
                    "model": r.model,
                    "success": r.success,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                    "think": r.think,
                }
                for r in records
            ],
        }


# ---------------------------------------------------------------------------
# Default model specifications
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[ModelStage, ModelSpec] = {
    ModelStage.VISION_ANALYSIS: ModelSpec(
        name=DEFAULT_VISION_MODEL,
        stage=ModelStage.VISION_ANALYSIS,
        description="Ollama vision model for label metadata extraction",
    ),
    ModelStage.METADATA_GENERATION: ModelSpec(
        name=DEFAULT_TEXT_MODEL,
        stage=ModelStage.METADATA_GENERATION,
        description=(
            "Text model: fact extraction from OCR/web evidence and verified description "
            "(text-only)"
        ),
    ),
}


#: Capability each stage's model must report (``/api/show`` capabilities).
STAGE_CAPABILITY: dict[ModelStage, str] = {
    ModelStage.VISION_ANALYSIS: "vision",
    ModelStage.METADATA_GENERATION: "completion",
}

#: When one model serves several stages, the capability that implies the
#: others is checked (a vision model also completes text).
_CAPABILITY_STRICTNESS: dict[str, int] = {"completion": 0, "vision": 1}


def active_stages(text_enabled: bool) -> frozenset[ModelStage]:
    """Stages whose models a run calls.

    Pre-flight requires (and may pull) only these stages' models.

    Args:
        text_enabled: Whether the text (fact-check) stage runs.

    Returns:
        ``{VISION_ANALYSIS}``, plus ``METADATA_GENERATION`` when
        ``text_enabled``.
    """
    if text_enabled:
        return frozenset({ModelStage.VISION_ANALYSIS, ModelStage.METADATA_GENERATION})
    return frozenset({ModelStage.VISION_ANALYSIS})


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

class ModelSelector:
    """Selects the model for each pipeline stage and tracks usage.

    Attributes:
        models: Mapping of stage to its ModelSpec.
        usage: UsageLog for tracking all model invocations (None if disabled).
    """

    def __init__(
        self,
        models: dict[ModelStage, ModelSpec] | None = None,
        log_usage: bool = True,
    ):
        """Initialize model selector.

        Args:
            models: Stage-to-model mapping. If None, uses the defaults.
            log_usage: Whether to track usage statistics.
        """
        if models is None:
            models = dict(_DEFAULT_MODELS)
        self.models: dict[ModelStage, ModelSpec] = models
        self.usage = UsageLog() if log_usage else None

    @classmethod
    def from_config(cls, config: Any) -> "ModelSelector":  # noqa: ANN401
        """Create a ModelSelector from a config/args object.

        ``config.model`` (the CLI ``--model``) becomes the vision-analysis
        model. A per-stage ``config.<stage>_model`` attribute (e.g.
        ``metadata_generation_model``) overrides that stage's model and wins
        over ``config.model``. Only non-empty strings count, so unset or
        mocked attributes leave the defaults in place.

        Args:
            config: Object with 'model' and optional per-stage overrides.

        Returns:
            Configured ModelSelector instance.
        """

        def _str_attr(name: str) -> Optional[str]:
            value = getattr(config, name, None)
            return value.strip() if isinstance(value, str) and value.strip() else None

        global_model = _str_attr("model")
        models: dict[ModelStage, ModelSpec] = {}
        for stage, spec in _DEFAULT_MODELS.items():
            override = _str_attr(f"{stage.name.lower()}_model")
            if override is None and stage is ModelStage.VISION_ANALYSIS:
                override = global_model
            if override and override != spec.name:
                # A replaced model must not keep the old model's wording.
                description = f"{override} ({stage.name.lower().replace('_', ' ')} override)"
                models[stage] = ModelSpec(name=override, stage=stage, description=description)
            else:
                models[stage] = spec
        return cls(models=models)

    def get_model(self, stage: ModelStage) -> str:
        """Get the model name for a stage.

        Args:
            stage: Pipeline stage.

        Returns:
            Model name string (e.g., "qwen3.6:35b-a3b").

        Raises:
            KeyError: If the stage has no registered model (never substitutes
                another stage's model).
        """
        spec = self.models.get(stage)
        if spec is None:
            raise KeyError(f"No model registered for stage {stage.name}")
        return spec.name

    def required_model_specs(self, stages: Iterable[ModelStage]) -> list[tuple[str, str]]:
        """Models (with the capability each must have) for the given stages.

        Names are deduplicated by :func:`normalize_model_name`, keeping the
        order of ``ModelStage`` and the first spelling. When one model
        serves several stages, the stricter capability is kept (``'vision'``
        over ``'completion'``).

        Args:
            stages: Stages to include, e.g. ``active_stages(text_enabled)``.

        Returns:
            ``(name, capability)`` pairs, e.g.
            ``[('qwen3.6:35b-a3b', 'vision'), ('gpt-oss:120b', 'completion')]``.
        """
        wanted = set(stages)
        specs: list[tuple[str, str]] = []
        index: dict[str, int] = {}
        for stage in ModelStage:
            if stage not in wanted:
                continue
            name = self.get_model(stage)
            capability = STAGE_CAPABILITY[stage]
            key = normalize_model_name(name)
            if key not in index:
                index[key] = len(specs)
                specs.append((name, capability))
                continue
            kept_name, kept_cap = specs[index[key]]
            if _CAPABILITY_STRICTNESS.get(capability, 0) > _CAPABILITY_STRICTNESS.get(kept_cap, 0):
                specs[index[key]] = (kept_name, capability)
        return specs

    def required_models(self, stages: Optional[Iterable[ModelStage]] = None) -> list[str]:
        """Names of the models that must be installed before the pipeline runs.

        Same order and deduplication as :meth:`required_model_specs`.

        Args:
            stages: Stages to include (default: ``active_stages(False)``,
                the vision stage only).

        Returns:
            Model names, e.g. ``['qwen3.6:35b-a3b']``.
        """
        if stages is None:
            stages = active_stages(False)
        return [name for name, _ in self.required_model_specs(stages)]

    def record_usage(
        self,
        stage: ModelStage,
        model: str,
        success: bool = True,
        duration: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: Optional[str] = None,
        think: Optional[str] = None,
    ) -> None:
        """Record a model invocation for usage tracking (thread-safe).

        Args:
            stage: Pipeline stage.
            model: Model name used.
            success: Whether the call succeeded.
            duration: Time taken in seconds.
            input_tokens: Input token count.
            output_tokens: Output token count.
            error: Error message if failed.
            think: Think level sent with the call, if any.
        """
        if self.usage is not None:
            self.usage.record(
                stage=stage,
                model=model,
                success=success,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error,
                think=think,
            )

    def get_usage_summary(self) -> dict[str, Any]:
        """Get a dictionary summary of all recorded usage.

        Returns:
            Dictionary with total_calls, successful_calls, total_duration,
            by_stage, by_model, and per-record details.
        """
        if self.usage is None:
            return {"error": "Usage logging not enabled"}
        return self.usage.to_dict()

    def print_stage_report(self, active: Optional[Iterable[ModelStage]] = None) -> None:
        """Log one line per stage: its model and whether this run calls it.

        Args:
            active: Stages this run calls (default: ``active_stages(False)``).
        """
        active_set = set(active_stages(False) if active is None else active)
        logger.info("=" * 70)
        logger.info("MODEL SELECTOR STAGE REPORT")
        logger.info("=" * 70)
        for stage in ModelStage:
            spec = self.models.get(stage)
            status = "active this run" if stage in active_set else "inactive"
            logger.info(
                "  %s: %s (%s)",
                stage.name,
                spec.name if spec is not None else "no model registered",
                status,
            )
        logger.info("=" * 70)
