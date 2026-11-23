"""Unit tests for ModelSelector — ModelStage enum, ModelSpec, UsageLog, ModelSelector.

Tests cover:
  - 8 ModelStage enum values
  - ModelSpec validation (role, temperature, max_tokens)
  - Default specs match research.md decisions
  - ModelSelector.get_model(), get_fallback_chain(), get_stage_description()
  - UsageLog.record(), summary_by_stage(), summary_by_model(), to_dict()
  - ModelSelector.record_usage(), get_usage_summary(), is_model_available(), print_stage_report()
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from lib.model_selector import (
    ModelSelector,
    ModelStage,
    ModelSpec,
    ModelUsage,
    UsageLog,
    create_default_selector,
    get_stage_models,
    _DEFAULT_MODELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelStage enum
# ---------------------------------------------------------------------------


class TestModelStage:
    """Verify all 8 pipeline stages are defined."""

    EXPECTED_STAGES = {
        "OCR_EXTRACTION",
        "VISION_ANALYSIS",
        "IMAGE_GROUPING",
        "QUANTITY_DETECTION",
        "METADATA_GENERATION",
        "WEB_RESEARCH",
        "ERROR_CLASSIFICATION",
        "PREVIEW_GENERATION",
    }

    def test_all_stages_present(self):
        assert len(ModelStage) == 8

    def test_expected_stage_names(self):
        actual = {s.name for s in ModelStage}
        assert actual == self.EXPECTED_STAGES

    def test_no_duplicate_values(self):
        values = [s.value for s in ModelStage]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ModelSpec validation
# ---------------------------------------------------------------------------


class TestModelSpec:
    """Test ModelSpec dataclass validation."""

    def test_valid_primary(self):
        spec = ModelSpec(
            name="qwen3-vl:32b",
            stage=ModelStage.VISION_ANALYSIS,
            role="primary",
            priority=0,
            max_tokens=4096,
            temperature=0.1,
            description="Test model",
        )
        assert spec.name == "qwen3-vl:32b"
        assert spec.role == "primary"
        assert spec.priority == 0

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS, role="invalid")

    def test_temperature_bounds(self):
        with pytest.raises(ValueError, match="temperature"):
            ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS, temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS, temperature=2.1)

    def test_max_tokens_non_negative(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS, max_tokens=-1)

    def test_defaults(self):
        spec = ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS)
        assert spec.role == "primary"
        assert spec.priority == 0
        assert spec.max_tokens == 0
        assert spec.temperature == 0.1
        assert spec.description == ""


# ---------------------------------------------------------------------------
# Default specs match research.md
# ---------------------------------------------------------------------------


class TestDefaultSpecs:
    """Verify default model specs match research.md decisions."""

    def test_ocr_primary_is_tesseract(self):
        specs = _DEFAULT_MODELS[ModelStage.OCR_EXTRACTION]
        assert specs[0].name == "tesseract"
        assert specs[0].role == "primary"

    def test_ocr_fallback_is_qwen2_vl_7b(self):
        specs = _DEFAULT_MODELS[ModelStage.OCR_EXTRACTION]
        assert len(specs) == 2
        assert specs[1].name == "qwen2-vl:7b"

    def test_vision_primary_is_qwen3_vl_32b(self):
        specs = _DEFAULT_MODELS[ModelStage.VISION_ANALYSIS]
        assert specs[0].name == "qwen3-vl:32b"

    def test_vision_fallback_is_qwen2_vl_7b(self):
        specs = _DEFAULT_MODELS[ModelStage.VISION_ANALYSIS]
        assert specs[1].name == "qwen2-vl:7b"

    def test_metadata_primary_is_gpt_oss_120b(self):
        specs = _DEFAULT_MODELS[ModelStage.METADATA_GENERATION]
        assert specs[0].name == "gpt-oss:120b"

    def test_metadata_fallback_is_qwen3_vl_32b(self):
        specs = _DEFAULT_MODELS[ModelStage.METADATA_GENERATION]
        assert specs[1].name == "qwen3-vl:32b"

    def test_grouping_primary_is_orb(self):
        specs = _DEFAULT_MODELS[ModelStage.IMAGE_GROUPING]
        assert specs[0].name == "orb-opencv"

    def test_quantity_primary_is_md5_hash(self):
        specs = _DEFAULT_MODELS[ModelStage.QUANTITY_DETECTION]
        assert specs[0].name == "md5-hash"

    def test_web_research_primary(self):
        specs = _DEFAULT_MODELS[ModelStage.WEB_RESEARCH]
        assert specs[0].name == "duckduckgo-api"

    def test_error_classification_primary(self):
        specs = _DEFAULT_MODELS[ModelStage.ERROR_CLASSIFICATION]
        assert specs[0].name == "rule-based"

    def test_preview_generation_primary(self):
        specs = _DEFAULT_MODELS[ModelStage.PREVIEW_GENERATION]
        assert specs[0].name == "template-markdown"

    def test_all_stages_have_at_least_one_model(self):
        for stage, specs in _DEFAULT_MODELS.items():
            assert len(specs) >= 1, f"{stage.name} has no models"

    def test_all_stages_have_primary(self):
        for stage, specs in _DEFAULT_MODELS.items():
            assert specs[0].role == "primary", f"{stage.name} missing primary"


# ---------------------------------------------------------------------------
# UsageLog
# ---------------------------------------------------------------------------


class TestModelUsage:
    """Test ModelUsage dataclass."""

    def test_defaults(self):
        usage = ModelUsage(stage=ModelStage.VISION_ANALYSIS, model="qwen3-vl:32b")
        assert usage.success is True
        assert usage.duration_seconds == 0.0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.error is None
        assert usage.fallback_used is False


class TestUsageLog:
    """Test UsageLog class."""

    def test_empty_log(self):
        log = UsageLog()
        assert log.total_calls == 0
        assert log.successful_calls == 0
        assert log.fallback_calls == 0
        assert log.total_duration == 0.0

    def test_record_single(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b", success=True, duration=1.5)
        assert log.total_calls == 1
        assert log.successful_calls == 1
        assert log.total_duration == 1.5

    def test_record_fallback(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen2-vl:7b", success=True, fallback_used=True)
        assert log.fallback_calls == 1

    def test_record_failure(self):
        log = UsageLog()
        log.record(
            ModelStage.VISION_ANALYSIS,
            "qwen3-vl:32b",
            success=False,
            error="Connection refused",
        )
        assert log.successful_calls == 0

    def test_summary_by_stage(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.VISION_ANALYSIS, "qwen2-vl:7b", fallback_used=True)
        log.record(ModelStage.OCR_EXTRACTION, "tesseract")
        summary = log.summary_by_stage()
        assert summary["VISION_ANALYSIS"]["total"] == 2
        assert summary["VISION_ANALYSIS"]["fallback"] == 1
        assert summary["OCR_EXTRACTION"]["total"] == 1

    def test_summary_by_model(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.OCR_EXTRACTION, "tesseract")
        summary = log.summary_by_model()
        assert summary["qwen3-vl:32b"]["total"] == 2
        assert summary["tesseract"]["total"] == 1

    def test_to_dict(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b", duration=1.5)
        d = log.to_dict()
        assert d["total_calls"] == 1
        assert d["successful_calls"] == 1
        assert d["total_duration_seconds"] == 1.5
        assert "by_stage" in d
        assert "by_model" in d
        assert len(d["records"]) == 1
        # JSON-serializable (no custom objects)
        json.dumps(d)


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


class TestModelSelector:
    """Test ModelSelector class."""

    def test_default_selector(self):
        selector = ModelSelector()
        assert len(selector.models) == 8
        assert selector.default_model == "qwen3-vl:32b"
        assert selector.allow_fallback is True
        assert selector.usage is not None

    def test_get_model_vision(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "qwen3-vl:32b"

    def test_get_model_ocr(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.OCR_EXTRACTION) == "tesseract"

    def test_get_model_metadata(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.METADATA_GENERATION) == "gpt-oss:120b"

    def test_get_model_grouping(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.IMAGE_GROUPING) == "orb-opencv"

    def test_get_model_quantity(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.QUANTITY_DETECTION) == "md5-hash"

    def test_get_model_web_research(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.WEB_RESEARCH) == "duckduckgo-api"

    def test_get_model_error_classification(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.ERROR_CLASSIFICATION) == "rule-based"

    def test_get_model_preview(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.PREVIEW_GENERATION) == "template-markdown"

    def test_get_fallback_chain(self):
        selector = ModelSelector()
        chain = selector.get_fallback_chain(ModelStage.VISION_ANALYSIS)
        assert chain == ["qwen3-vl:32b", "qwen2-vl:7b"]

    def test_get_fallback_chain_single_model(self):
        selector = ModelSelector()
        chain = selector.get_fallback_chain(ModelStage.WEB_RESEARCH)
        assert chain == ["duckduckgo-api"]

    def test_get_stage_description(self):
        selector = ModelSelector()
        desc = selector.get_stage_description(ModelStage.VISION_ANALYSIS)
        assert "Qwen3-VL" in desc or "vision" in desc.lower()

    def test_custom_models(self):
        custom = {
            ModelStage.VISION_ANALYSIS: [
                ModelSpec(name="llama3.2", stage=ModelStage.VISION_ANALYSIS, role="primary"),
                ModelSpec(name="qwen2-vl:7b", stage=ModelStage.VISION_ANALYSIS, role="fallback"),
            ],
        }
        for stage, specs in _DEFAULT_MODELS.items():
            if stage not in custom:
                custom[stage] = specs
        selector = ModelSelector(models=custom)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"

    def test_from_config(self):
        config = MagicMock()
        config.model = "llama3.2"
        config.vision_analysis_model = "llama3.2"
        selector = ModelSelector.from_config(config)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"

    def test_from_config_no_overrides(self):
        config = MagicMock()
        config.model = "qwen3-vl:32b"
        # No per-stage override attributes
        selector = ModelSelector.from_config(config)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "qwen3-vl:32b"

    def test_record_usage(self):
        selector = ModelSelector()
        selector.record_usage(
            stage=ModelStage.VISION_ANALYSIS,
            model="qwen3-vl:32b",
            success=True,
            duration=2.5,
        )
        assert selector.usage.total_calls == 1

    def test_get_usage_summary(self):
        selector = ModelSelector()
        selector.record_usage(
            stage=ModelStage.VISION_ANALYSIS,
            model="qwen3-vl:32b",
            success=True,
            duration=2.5,
        )
        summary = selector.get_usage_summary()
        assert summary["total_calls"] == 1
        assert summary["successful_calls"] == 1

    def test_get_usage_summary_no_logging(self):
        selector = ModelSelector(log_usage=False)
        summary = selector.get_usage_summary()
        assert summary == {"error": "Usage logging not enabled"}

    def test_is_model_available(self):
        selector = ModelSelector()
        assert selector.is_model_available("qwen3-vl:32b") is True
        assert selector.is_model_available("tesseract") is True
        assert selector.is_model_available("nonexistent-model") is False

    def test_print_stage_report(self, caplog):
        selector = ModelSelector()
        with caplog.at_level(logging.INFO):
            selector.print_stage_report()
        assert "MODEL SELECTOR STAGE REPORT" in caplog.text
        assert "qwen3-vl:32b" in caplog.text
        assert "Tesseract" in caplog.text or "tesseract" in caplog.text


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Test create_default_selector() and get_stage_models()."""

    def test_create_default_selector(self):
        selector = create_default_selector()
        assert isinstance(selector, ModelSelector)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "qwen3-vl:32b"

    def test_get_stage_models(self):
        models = get_stage_models(ModelStage.VISION_ANALYSIS)
        assert models == ["qwen3-vl:32b", "qwen2-vl:7b"]

    def test_get_stage_models_single(self):
        models = get_stage_models(ModelStage.WEB_RESEARCH)
        assert models == ["duckduckgo-api"]