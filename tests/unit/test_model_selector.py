"""Unit tests for ModelSelector: ModelStage, ModelSpec, UsageLog, ModelSelector.

Tests cover:
  - The two model stages (vision analysis, metadata generation)
  - Default models; no fallback models anywhere
  - ModelSelector.get_model() (KeyError for an unregistered stage), from_config(),
    required_models(), required_model_specs(), active_stages(), STAGE_CAPABILITY
  - UsageLog.record(), summary_by_stage(), summary_by_model(), to_dict(),
    thread safety
  - ModelSelector.record_usage(), get_usage_summary(), print_stage_report()
"""

import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.constants import DEFAULT_VISION_MODEL
from lib.model_selector import (
    _DEFAULT_MODELS,
    DEFAULT_TEXT_MODEL,
    STAGE_CAPABILITY,
    ModelSelector,
    ModelSpec,
    ModelStage,
    ModelUsage,
    UsageLog,
    active_stages,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelStage enum
# ---------------------------------------------------------------------------


class TestModelStage:
    """Only stages that call a model exist."""

    def test_stage_names(self):
        assert {s.name for s in ModelStage} == {"VISION_ANALYSIS", "METADATA_GENERATION"}

    def test_no_duplicate_values(self):
        values = [s.value for s in ModelStage]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_fields(self):
        spec = ModelSpec(name="qwen3-vl:32b", stage=ModelStage.VISION_ANALYSIS, description="d")
        assert (spec.name, spec.stage, spec.description) == (
            "qwen3-vl:32b",
            ModelStage.VISION_ANALYSIS,
            "d",
        )

    def test_description_defaults_empty(self):
        assert ModelSpec(name="test", stage=ModelStage.VISION_ANALYSIS).description == ""


# ---------------------------------------------------------------------------
# Default specs
# ---------------------------------------------------------------------------


class TestDefaultSpecs:
    def test_one_model_per_stage(self):
        assert set(_DEFAULT_MODELS) == set(ModelStage)
        for stage, spec in _DEFAULT_MODELS.items():
            assert isinstance(spec, ModelSpec)
            assert spec.stage is stage

    def test_vision_default(self):
        spec = _DEFAULT_MODELS[ModelStage.VISION_ANALYSIS]
        assert spec.name == DEFAULT_VISION_MODEL
        assert "qwen3-vl" not in spec.description.lower()

    def test_metadata_generation_default_is_gpt_oss(self):
        spec = _DEFAULT_MODELS[ModelStage.METADATA_GENERATION]
        assert spec.name == DEFAULT_TEXT_MODEL == "gpt-oss:120b"
        assert "text-only" in spec.description


# ---------------------------------------------------------------------------
# UsageLog
# ---------------------------------------------------------------------------


class TestModelUsage:
    def test_defaults(self):
        usage = ModelUsage(stage=ModelStage.VISION_ANALYSIS, model="qwen3-vl:32b")
        assert usage.success is True
        assert usage.duration_seconds == 0.0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.error is None
        assert not hasattr(usage, "fallback_used")


class TestUsageLog:
    def test_empty_log(self):
        log = UsageLog()
        assert log.total_calls == 0
        assert log.successful_calls == 0
        assert log.total_duration == 0.0

    def test_record_single(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b", success=True, duration=1.5)
        assert log.total_calls == 1
        assert log.successful_calls == 1
        assert log.total_duration == 1.5

    def test_record_failure(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b", success=False, error="Connection refused")
        assert log.successful_calls == 0
        assert log.records[0].error == "Connection refused"

    def test_summary_by_stage(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b", success=False)
        log.record(ModelStage.METADATA_GENERATION, "gpt-oss:120b")
        summary = log.summary_by_stage()
        assert summary["VISION_ANALYSIS"] == {"total": 2, "success": 1}
        assert summary["METADATA_GENERATION"] == {"total": 1, "success": 1}

    def test_summary_by_model(self):
        log = UsageLog()
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        log.record(ModelStage.METADATA_GENERATION, "gpt-oss:120b")
        summary = log.summary_by_model()
        assert summary["qwen3-vl:32b"]["total"] == 2
        assert summary["gpt-oss:120b"]["total"] == 1

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
        assert "fallback_calls" not in d
        assert "fallback_used" not in d["records"][0]
        json.dumps(d)  # JSON-serializable (no custom objects)

    def test_records_think_level(self):
        log = UsageLog()
        log.record(ModelStage.METADATA_GENERATION, "gpt-oss:120b", think="high")
        log.record(ModelStage.VISION_ANALYSIS, "qwen3-vl:32b")
        records = log.to_dict()["records"]
        assert records[0]["think"] == "high"
        assert records[1]["think"] is None

    def test_selector_record_usage_passes_think(self):
        selector = ModelSelector()
        selector.record_usage(ModelStage.METADATA_GENERATION, "gpt-oss:120b", think="medium")
        assert selector.usage.records[0].think == "medium"

    def test_thread_safe_under_parallel_records(self):
        log = UsageLog()
        threads_n, per_thread = 8, 100
        start = threading.Barrier(threads_n)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                start.wait()
                for j in range(per_thread):
                    log.record(
                        ModelStage.VISION_ANALYSIS, f"m{i}", success=j % 2 == 0, duration=0.5
                    )
                    log.to_dict()  # summaries run concurrently with appends
            except BaseException as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert log.total_calls == threads_n * per_thread
        assert log.successful_calls == threads_n * per_thread // 2
        assert log.total_duration == pytest.approx(threads_n * per_thread * 0.5)
        assert log.summary_by_stage()["VISION_ANALYSIS"]["total"] == threads_n * per_thread
        assert all(v["total"] == per_thread for v in log.summary_by_model().values())


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


class TestModelSelector:
    def test_default_selector(self):
        selector = ModelSelector()
        assert set(selector.models) == set(ModelStage)
        assert selector.usage is not None

    def test_get_model_defaults(self):
        selector = ModelSelector()
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == DEFAULT_VISION_MODEL
        assert selector.get_model(ModelStage.METADATA_GENERATION) == "gpt-oss:120b"

    def test_get_model_unregistered_stage_raises_key_error(self):
        """Never substitutes another stage's model (a vision model for a text stage)."""
        selector = ModelSelector(
            models={ModelStage.VISION_ANALYSIS: _DEFAULT_MODELS[ModelStage.VISION_ANALYSIS]}
        )
        with pytest.raises(KeyError, match="METADATA_GENERATION"):
            selector.get_model(ModelStage.METADATA_GENERATION)

    def test_custom_models(self):
        custom = dict(_DEFAULT_MODELS)
        custom[ModelStage.VISION_ANALYSIS] = ModelSpec(name="llama3.2", stage=ModelStage.VISION_ANALYSIS)
        assert ModelSelector(models=custom).get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"

    def test_from_config_model_sets_vision_only(self):
        selector = ModelSelector.from_config(SimpleNamespace(model="llama3.2"))
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"
        assert selector.get_model(ModelStage.METADATA_GENERATION) == DEFAULT_TEXT_MODEL

    def test_from_config_per_stage_override(self):
        config = SimpleNamespace(
            model="qwen3-vl:32b",
            vision_analysis_model="llama3.2",
            metadata_generation_model="gpt-oss:20b",
        )
        selector = ModelSelector.from_config(config)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"
        assert selector.get_model(ModelStage.METADATA_GENERATION) == "gpt-oss:20b"
        assert "override" in selector.models[ModelStage.METADATA_GENERATION].description

    def test_from_config_ignores_mock_attributes(self):
        config = MagicMock()
        config.model = "qwen3-vl:32b"
        selector = ModelSelector.from_config(config)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "qwen3-vl:32b"
        assert selector.get_model(ModelStage.METADATA_GENERATION) == DEFAULT_TEXT_MODEL

    def test_active_stages(self):
        assert active_stages(False) == frozenset({ModelStage.VISION_ANALYSIS})
        assert active_stages(True) == frozenset(
            {ModelStage.VISION_ANALYSIS, ModelStage.METADATA_GENERATION}
        )

    def test_stage_capabilities(self):
        assert STAGE_CAPABILITY == {
            ModelStage.VISION_ANALYSIS: "vision",
            ModelStage.METADATA_GENERATION: "completion",
        }

    def test_required_models_only_vision(self):
        selector = ModelSelector.from_config(SimpleNamespace(model="gemma4:26b"))
        required = selector.required_models()
        assert required == ["gemma4:26b"]
        assert "gpt-oss:120b" not in required  # text stage inactive by default

    @pytest.mark.parametrize("text_enabled", [False, True])
    def test_gpt_oss_required_iff_text_stage_active(self, text_enabled):
        selector = ModelSelector.from_config(SimpleNamespace(model="gemma4:26b"))
        required = selector.required_models(active_stages(text_enabled))
        assert ("gpt-oss:120b" in required) is text_enabled

    def test_required_model_specs_capability_pairs(self):
        selector = ModelSelector.from_config(SimpleNamespace(model="gemma4:26b"))
        assert selector.required_model_specs(active_stages(False)) == [("gemma4:26b", "vision")]
        assert selector.required_model_specs(active_stages(True)) == [
            ("gemma4:26b", "vision"),
            ("gpt-oss:120b", "completion"),
        ]

    def test_required_model_specs_same_model_keeps_vision(self):
        config = SimpleNamespace(model="qwen3-vl:32b", metadata_generation_model="QWEN3-VL:32b")
        selector = ModelSelector.from_config(config)
        assert selector.required_model_specs(active_stages(True)) == [("qwen3-vl:32b", "vision")]

    def test_required_model_specs_dedupes_normalized_names(self):
        config = SimpleNamespace(
            model="library/text-and-vision", metadata_generation_model="text-and-vision:latest"
        )
        selector = ModelSelector.from_config(config)
        assert selector.required_model_specs(active_stages(True)) == [("library/text-and-vision", "vision")]

    def test_required_model_specs_empty(self):
        assert ModelSelector().required_model_specs([]) == []

    def test_required_models_custom_stages_dedupe_in_stage_order(self):
        selector = ModelSelector()
        stages = [ModelStage.METADATA_GENERATION, ModelStage.VISION_ANALYSIS, ModelStage.VISION_ANALYSIS]
        assert selector.required_models(stages) == [DEFAULT_VISION_MODEL, "gpt-oss:120b"]

    def test_bare_selector_requires_default_vision_model(self):
        assert ModelSelector().required_models() == [DEFAULT_VISION_MODEL]

    def test_required_models_empty(self):
        assert ModelSelector().required_models([]) == []

    def test_record_usage(self):
        selector = ModelSelector()
        selector.record_usage(stage=ModelStage.VISION_ANALYSIS, model="qwen3-vl:32b", success=True, duration=2.5)
        assert selector.usage.total_calls == 1

    def test_get_usage_summary(self):
        selector = ModelSelector()
        selector.record_usage(stage=ModelStage.VISION_ANALYSIS, model="qwen3-vl:32b", success=True, duration=2.5)
        summary = selector.get_usage_summary()
        assert summary["total_calls"] == 1
        assert summary["successful_calls"] == 1

    def test_get_usage_summary_no_logging(self):
        selector = ModelSelector(log_usage=False)
        assert selector.get_usage_summary() == {"error": "Usage logging not enabled"}

    def test_print_stage_report_one_line_per_stage(self, caplog):
        selector = ModelSelector.from_config(SimpleNamespace(model="gemma4:26b"))
        with caplog.at_level(logging.INFO, logger="lib.model_selector"):
            selector.print_stage_report()
        assert "MODEL SELECTOR STAGE REPORT" in caplog.text
        assert "VISION_ANALYSIS: gemma4:26b (active this run)" in caplog.text
        assert "METADATA_GENERATION: gpt-oss:120b (inactive)" in caplog.text
        assert "allback" not in caplog.text

    def test_print_stage_report_marks_active_stages(self, caplog):
        selector = ModelSelector.from_config(SimpleNamespace(model="gemma4:26b"))
        with caplog.at_level(logging.INFO, logger="lib.model_selector"):
            selector.print_stage_report(active_stages(True))
        assert "VISION_ANALYSIS: gemma4:26b (active this run)" in caplog.text
        assert "METADATA_GENERATION: gpt-oss:120b (active this run)" in caplog.text
        assert "inactive" not in caplog.text
