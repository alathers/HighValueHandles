"""Model selector for the bourbon image processor pipeline.

Assigns the appropriate AI/ML model to each processing stage, manages
fallback chains, tracks usage, and provides runtime model resolution.

Pipeline stages and their recommended models (from research.md):

    Stage                          | Primary Model       | Fallback
    --------------------------------|---------------------|------------------
    1. OCR / Text Extraction        | Tesseract-OCR       | qwen2-vl:7b
    2. Vision Model Analysis        | qwen3-vl:32b        | qwen2-vl:7b
    3. Image Grouping (Similarity)  | ORB (OpenCV)        | filename heuristics
    4. Quantity Detection            | Hash-based          | Visual analysis
    5. Metadata Generation (Text)   | gpt-oss:120b        | qwen3-vl:32b
    6. Web Research (Vendor Cache)  | DuckDuckGo API      | Skip (non-blocking)
    7. Error Classification          | Rule-based          | N/A

Usage::

    from lib.model_selector import ModelSelector, ModelStage

    selector = ModelSelector.from_config(config)

    # Get the primary model for a stage
    model_name = selector.get_model(ModelStage.VISION_ANALYSIS)

    # Execute with automatic fallback
    result = selector.execute_with_fallback(
        stage=ModelStage.VISION_ANALYSIS,
        fn=vision_fn,
        image_path=image_path,
        prompt=prompt,
    )

"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------

class ModelStage(Enum):
    """Pipeline stages that require model selection.

    Each stage maps to a recommended primary model and one or more
    fallback models, configured via the ModelSelector.
    """

    OCR_EXTRACTION = auto()
    """Stage 1: Text extraction from images (Tesseract OCR)."""

    VISION_ANALYSIS = auto()
    """Stage 2: Vision model metadata extraction from bottle images."""

    IMAGE_GROUPING = auto()
    """Stage 3: Visual similarity for image grouping (ORB/OpenCV)."""

    QUANTITY_DETECTION = auto()
    """Stage 4: Quantity detection via hash-based or visual analysis."""

    METADATA_GENERATION = auto()
    """Stage 5: Text-to-text metadata refinement (GPT-OSS)."""

    WEB_RESEARCH = auto()
    """Stage 6: Web search for vendor images and reference links."""

    ERROR_CLASSIFICATION = auto()
    """Stage 7: Error type classification for retry logic."""

    PREVIEW_GENERATION = auto()
    """Stage 8: Markdown preview report generation (text model)."""


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a single model in a fallback chain.

    Attributes:
        name: Model identifier (e.g., "qwen3-vl:32b").
        stage: Pipeline stage this model serves.
        role: "primary" or "fallback".
        priority: Lower numbers = tried first within same role.
        max_tokens: Maximum output tokens (0 = unlimited).
        temperature: Generation temperature (0.0-2.0).
        description: Human-readable description of what this model does.
    """

    name: str
    stage: ModelStage
    role: str = "primary"
    priority: int = 0
    max_tokens: int = 0
    temperature: float = 0.1
    description: str = ""

    def __post_init__(self):
        """Validate model spec."""
        if self.role not in ("primary", "fallback"):
            raise ValueError(f"role must be 'primary' or 'fallback', got {self.role!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")


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
        fallback_used: Whether a fallback model was used instead of primary.
    """

    stage: ModelStage
    model: str
    success: bool = True
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    fallback_used: bool = False


@dataclass
class UsageLog:
    """Accumulates usage records for reporting.

    Attributes:
        records: List of ModelUsage records.
    """

    records: list[ModelUsage] = field(default_factory=list)

    def record(self, stage: ModelStage, model: str, success: bool = True,
               duration: float = 0.0, input_tokens: int = 0,
               output_tokens: int = 0, error: Optional[str] = None,
               fallback_used: bool = False) -> None:
        """Record a single model invocation.

        Args:
            stage: Pipeline stage.
            model: Model name used.
            success: Whether the call succeeded.
            duration: Time taken in seconds.
            input_tokens: Input token count.
            output_tokens: Output token count.
            error: Error message if failed.
            fallback_used: Whether a fallback model was used.
        """
        self.records.append(ModelUsage(
            stage=stage,
            model=model,
            success=success,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            fallback_used=fallback_used,
        ))

    @property
    def total_calls(self) -> int:
        """Total number of model invocations."""
        return len(self.records)

    @property
    def successful_calls(self) -> int:
        """Number of successful model invocations."""
        return sum(1 for r in self.records if r.success)

    @property
    def fallback_calls(self) -> int:
        """Number of times a fallback model was used."""
        return sum(1 for r in self.records if r.fallback_used)

    @property
    def total_duration(self) -> float:
        """Total time spent in model calls."""
        return sum(r.duration_seconds for r in self.records)

    def summary_by_stage(self) -> dict[str, dict[str, int]]:
        """Get usage summary grouped by stage.

        Returns:
            Dictionary mapping stage names to {total, success, fallback}.
        """
        summary: dict[str, dict[str, int]] = {}
        for r in self.records:
            stage_name = r.stage.name
            if stage_name not in summary:
                summary[stage_name] = {"total": 0, "success": 0, "fallback": 0}
            summary[stage_name]["total"] += 1
            if r.success:
                summary[stage_name]["success"] += 1
            if r.fallback_used:
                summary[stage_name]["fallback"] += 1
        return summary

    def summary_by_model(self) -> dict[str, dict[str, int]]:
        """Get usage summary grouped by model name.

        Returns:
            Dictionary mapping model names to {total, success, fallback}.
        """
        summary: dict[str, dict[str, int]] = {}
        for r in self.records:
            if r.model not in summary:
                summary[r.model] = {"total": 0, "success": 0, "fallback": 0}
            summary[r.model]["total"] += 1
            if r.success:
                summary[r.model]["success"] += 1
            if r.fallback_used:
                summary[r.model]["fallback"] += 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with summary statistics and per-record details.
        """
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "fallback_calls": self.fallback_calls,
            "total_duration_seconds": round(self.total_duration, 2),
            "by_stage": self.summary_by_stage(),
            "by_model": self.summary_by_model(),
            "records": [
                {
                    "stage": r.stage.name,
                    "model": r.model,
                    "success": r.success,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                    "fallback_used": r.fallback_used,
                }
                for r in self.records
            ],
        }


# ---------------------------------------------------------------------------
# Default model specifications (from research.md)
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[ModelStage, list[ModelSpec]] = {
    ModelStage.OCR_EXTRACTION: [
        ModelSpec(
            name="tesseract",
            stage=ModelStage.OCR_EXTRACTION,
            role="primary",
            priority=0,
            description="Tesseract OCR system engine for text extraction",
        ),
        ModelSpec(
            name="qwen2-vl:7b",
            stage=ModelStage.OCR_EXTRACTION,
            role="fallback",
            priority=1,
            description="Qwen2-VL 7B vision model as OCR fallback",
        ),
    ],
    ModelStage.VISION_ANALYSIS: [
        ModelSpec(
            name="qwen3-vl:32b",
            stage=ModelStage.VISION_ANALYSIS,
            role="primary",
            priority=0,
            description="Qwen3-VL 32B primary vision model for metadata extraction",
        ),
        ModelSpec(
            name="qwen2-vl:7b",
            stage=ModelStage.VISION_ANALYSIS,
            role="fallback",
            priority=1,
            description="Qwen2-VL 7B fallback for vision analysis",
        ),
    ],
    ModelStage.IMAGE_GROUPING: [
        ModelSpec(
            name="orb-opencv",
            stage=ModelStage.IMAGE_GROUPING,
            role="primary",
            priority=0,
            description="OpenCV ORB feature matching for image similarity",
        ),
        ModelSpec(
            name="filename-heuristic",
            stage=ModelStage.IMAGE_GROUPING,
            role="fallback",
            priority=1,
            description="Filename pattern matching as last-resort grouping",
        ),
    ],
    ModelStage.QUANTITY_DETECTION: [
        ModelSpec(
            name="md5-hash",
            stage=ModelStage.QUANTITY_DETECTION,
            role="primary",
            priority=0,
            description="MD5 hash-based duplicate detection for quantity",
        ),
        ModelSpec(
            name="visual-analysis",
            stage=ModelStage.QUANTITY_DETECTION,
            role="fallback",
            priority=1,
            description="Visual analysis heuristic for implicit multiple bottles",
        ),
    ],
    ModelStage.METADATA_GENERATION: [
        ModelSpec(
            name="gpt-oss:120b",
            stage=ModelStage.METADATA_GENERATION,
            role="primary",
            priority=0,
            description="GPT-OSS 120B for structured metadata generation",
        ),
        ModelSpec(
            name="qwen3-vl:32b",
            stage=ModelStage.METADATA_GENERATION,
            role="fallback",
            priority=1,
            description="Qwen3-VL 32B fallback for text-to-text metadata",
        ),
    ],
    ModelStage.WEB_RESEARCH: [
        ModelSpec(
            name="duckduckgo-api",
            stage=ModelStage.WEB_RESEARCH,
            role="primary",
            priority=0,
            description="DuckDuckGo web search for vendor image lookup",
        ),
    ],
    ModelStage.ERROR_CLASSIFICATION: [
        ModelSpec(
            name="rule-based",
            stage=ModelStage.ERROR_CLASSIFICATION,
            role="primary",
            priority=0,
            description="Rule-based error classification for retry logic",
        ),
    ],
    ModelStage.PREVIEW_GENERATION: [
        ModelSpec(
            name="template-markdown",
            stage=ModelStage.PREVIEW_GENERATION,
            role="primary",
            priority=0,
            description="Template-based markdown generation (no model needed)",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

class ModelSelector:
    """Selects and tracks models for each pipeline stage.

    Manages a registry of model specifications (primary + fallback chains),
    resolves models at runtime, and tracks usage for reporting.

    Attributes:
        models: Mapping of stage to list of ModelSpec (primary first).
        usage: UsageLog for tracking all model invocations.
        default_model: Fallback model name when stage-specific models unavailable.
    """

    def __init__(
        self,
        models: dict[ModelStage, list[ModelSpec]] | None = None,
        default_model: str = "qwen3-vl:32b",
        allow_fallback: bool = True,
        log_usage: bool = True,
    ):
        """Initialize model selector.

        Args:
            models: Custom model specifications. If None, uses defaults from research.md.
            default_model: Fallback model when stage-specific models unavailable.
            allow_fallback: Whether to attempt fallback models on failure.
            log_usage: Whether to track usage statistics.
        """
        if models is None:
            models = {stage: list(specs) for stage, specs in _DEFAULT_MODELS.items()}
        self.models: dict[ModelStage, list[ModelSpec]] = models
        self.default_model = default_model
        self.allow_fallback = allow_fallback
        self.usage = UsageLog() if log_usage else None

    @classmethod
    def from_config(cls, config: Any) -> "ModelSelector":  # noqa: ANN401
        """Create a ModelSelector from a Config object.

        Args:
            config: Config object with 'model' and optional per-stage overrides.

        Returns:
            Configured ModelSelector instance.
        """
        default_model = getattr(config, "model", "qwen3-vl:32b")

        # Check for per-stage model overrides in config (future extension)
        custom_models: dict[ModelStage, list[ModelSpec]] = {}
        for stage, specs in _DEFAULT_MODELS.items():
            custom_models[stage] = [
                ModelSpec(
                    name=getattr(config, f"{stage.name.lower()}_model", spec.name),
                    stage=stage,
                    role=spec.role,
                    priority=spec.priority,
                    max_tokens=spec.max_tokens,
                    temperature=spec.temperature,
                    description=spec.description,
                )
                for spec in specs
            ]

        return cls(
            models=custom_models,
            default_model=default_model,
        )

    def get_model(self, stage: ModelStage) -> str:
        """Get the primary model name for a stage.

        Args:
            stage: Pipeline stage.

        Returns:
            Model name string (e.g., "qwen3-vl:32b").

        Raises:
            KeyError: If stage has no registered models.
        """
        specs = self.models.get(stage)
        if not specs:
            return self.default_model
        return specs[0].name

    def get_fallback_chain(self, stage: ModelStage) -> list[str]:
        """Get all models for a stage in priority order (primary + fallbacks).

        Args:
            stage: Pipeline stage.

        Returns:
            List of model names, primary first.
        """
        specs = self.models.get(stage, [])
        return [spec.name for spec in specs]

    def get_stage_description(self, stage: ModelStage) -> str:
        """Get human-readable description of a stage's primary model.

        Args:
            stage: Pipeline stage.

        Returns:
            Description string.
        """
        specs = self.models.get(stage, [])
        if specs:
            return specs[0].description
        return f"Default model for {stage.name}"

    def record_usage(
        self,
        stage: ModelStage,
        model: str,
        success: bool = True,
        duration: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: Optional[str] = None,
        fallback_used: bool = False,
    ) -> None:
        """Record a model invocation for usage tracking.

        Args:
            stage: Pipeline stage.
            model: Model name used.
            success: Whether the call succeeded.
            duration: Time taken in seconds.
            input_tokens: Input token count.
            output_tokens: Output token count.
            error: Error message if failed.
            fallback_used: Whether a fallback model was used.
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
                fallback_used=fallback_used,
            )

    def get_usage_summary(self) -> dict[str, Any]:
        """Get a dictionary summary of all recorded usage.

        Returns:
            Dictionary with total_calls, successful_calls, fallback_calls,
            total_duration, by_stage, by_model, and per-record details.
        """
        if self.usage is None:
            return {"error": "Usage logging not enabled"}
        return self.usage.to_dict()

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is in the registered specifications.

        Args:
            model_name: Model name to check.

        Returns:
            True if the model is registered in any stage's chain.
        """
        for specs in self.models.values():
            if any(spec.name == model_name for spec in specs):
                return True
        return model_name == self.default_model

    def print_stage_report(self) -> None:
        """Print a human-readable report of all stage configurations.

        Logs the primary model, fallback chain, and description for each stage.
        """
        logger.info("=" * 70)
        logger.info("MODEL SELECTOR STAGE REPORT")
        logger.info("=" * 70)
        for stage, specs in self.models.items():
            primary = specs[0].name if specs else "N/A"
            fallbacks = ", ".join(spec.name for spec in specs[1:])
            logger.info(
                "  %s\n    Primary: %s\n    Fallbacks: %s\n    %s",
                stage.name,
                primary,
                fallbacks if fallbacks else "(none)",
                specs[0].description if specs else "No models registered",
            )
        logger.info("-" * 70)
        logger.info("Default fallback model: %s", self.default_model)
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_default_selector() -> ModelSelector:
    """Create a ModelSelector with default specifications from research.md.

    Returns:
        Configured ModelSelector with all stage defaults.
    """
    return ModelSelector()


def get_stage_models(stage: ModelStage) -> list[str]:
    """Get all model names for a stage from defaults.

    Args:
        stage: Pipeline stage.

    Returns:
        List of model names (primary + fallbacks).
    """
    specs = _DEFAULT_MODELS.get(stage, [])
    return [spec.name for spec in specs]