# Feature Specification: Model Selector

> **Superseded (2026-11): Ollama is the only model-serving backend.** The MTPLX/MPLTX
> backend described below (`MpltxServer`, `ServerBackend.MPLTX`, `mpltx_*` config
> fields, `--server-backend mpltx` as the default) was removed: MTPLX runs only on
> Apple Silicon and this tool must stay portable to Linux. `--server-backend`
> accepts only `ollama`; `mtplx`/`mpltx` values and `mpltx_*` settings are
> rejected with an explanation. Only the vision-analysis model is called (no
> fallback chain, no gpt-oss metadata stage). See README.md for current behaviour.

**Feature Branch**: `002-model-selector`

**Created**: 2026-11-21

**Status**: Complete

**Input**: Review of existing codebase and research.md decisions for per-stage model selection.

---

## Overview

The bourbon image processor currently uses a single model configuration (`--model` CLI flag) that applies to all AI/ML operations. This is suboptimal because different pipeline stages have fundamentally different requirements:

- **OCR extraction** needs fast, reliable text recognition (Tesseract excels here)
- **Vision analysis** needs strong multimodal understanding (Qwen3-VL 32B)
- **Metadata generation** needs strong text reasoning (GPT-OSS 120B)
- **Image grouping** needs fast feature matching (ORB/OpenCV, no LLM needed)
- **Quantity detection** needs hash comparison (no LLM needed)
- **Web research** needs search API integration (no LLM needed)
- **Error classification** needs pattern matching (rule-based, no LLM needed)

The model selector centralizes model assignment per stage, provides fallback chains, and tracks usage statistics.

---

## Clarifications

### Session 2026-11-21

- Q: Should all stages use AI models? → A: No. Several stages (image grouping, quantity detection, error classification, preview generation) work better with deterministic algorithms. The model selector should distinguish between AI models and deterministic tools.
- Q: How should fallback chains work? → A: Each stage has a primary model and zero or more fallbacks. When the primary fails, the selector automatically tries the next model in the chain. Usage tracking records whether a fallback was used.
- Q: Should the CLI still support a single `--model` flag? → A: Yes, for backward compatibility. The single `--model` flag sets the primary model for stages that use AI models. Per-stage model overrides are configured via environment variables (e.g., `BOURBON_VISION_ANALYSIS_MODEL`).
- Q: How should model availability be checked? → A: The selector checks model availability during pre-flight and logs which models are available/unavailable. Stages without their primary model attempt fallbacks automatically.
- Q: What usage tracking is needed? → A: Track per-invocation: stage, model used, success/failure, duration, tokens (if available), and whether a fallback was used. Provide summary reports grouped by stage and by model.

---

## User Scenarios & Testing

### User Story 1 - Configure Per-Stage Models (Priority: P1)

A user wants to use different models for different pipeline stages to optimize cost, speed, and accuracy. For example, using Tesseract for OCR (fast, local) but Qwen3-VL 32B for vision analysis (high accuracy).

**Why this priority**: This is the core value — different stages have different requirements. A single model cannot optimally serve all stages.

**Acceptance Scenarios**:

1. **Given** the system starts, **When** the user runs with default configuration, **Then** each stage uses its recommended primary model from research.md
2. **Given** the user sets `BOURBON_VISION_ANALYSIS_MODEL=qwen2-vl:7b`, **When** processing runs, **Then** vision analysis uses qwen2-vl:7b instead of qwen3-vl:32b
3. **Given** the user sets `--model llama3.2`, **When** processing runs, **Then** all AI stages use llama3.2 as their primary model (backward compatible with single --model flag)
4. **Given** the system starts, **When** the user runs with `--debug`, **Then** the model selector prints a stage report showing primary models, fallback chains, and descriptions

### User Story 2 - Automatic Fallback on Model Unavailable (Priority: P1)

A user has a local Ollama instance with qwen2-vl:7b but not qwen3-vl:32b. The system should automatically fall back to available models without failing.

**Why this priority**: Without fallback, processing fails when the preferred model is unavailable, violating the local-first design principle.

**Acceptance Scenarios**:

1. **Given** qwen3-vl:32b is not available in Ollama, **When** vision analysis runs, **Then** the system automatically falls back to qwen2-vl:7b
2. **Given** no AI model is available for a stage, **When** processing runs, **Then** the system falls back to the stage's non-AI fallback (e.g., Tesseract for OCR, ORB for grouping)
3. **Given** a fallback model is used, **When** usage is tracked, **Then** the record shows `fallback_used: true`
4. **Given** all models in a chain are unavailable, **When** processing runs, **Then** the system logs a warning and uses the most capable available fallback

### User Story 3 - Usage Tracking and Reporting (Priority: P2)

A user wants to understand model usage: which models are used most, how many fallbacks occur, total processing time per stage, and which stages are most expensive.

**Why this priority**: Usage visibility helps users optimize their setup and understand costs (in terms of compute time, not monetary cost, since everything is local).

**Acceptance Scenarios**:

1. **Given** processing completes, **When** the user checks usage, **Then** they see a summary with total calls, successful calls, fallback count, and total duration
2. **Given** processing completes, **When** the user checks usage, **Then** they see a breakdown by stage (OCR, Vision, Grouping, etc.)
3. **Given** processing completes, **When** the user checks usage, **Then** they see a breakdown by model name (how many times each model was called)
4. **Given** usage logging is enabled, **When** the system serializes usage, **Then** it produces a JSON-serializable dictionary with all records

---

## Functional Requirements

- **FR-001**: System MUST define a `ModelStage` enum covering all 8 pipeline stages: OCR_EXTRACTION, VISION_ANALYSIS, IMAGE_GROUPING, QUANTITY_DETECTION, METADATA_GENERATION, WEB_RESEARCH, ERROR_CLASSIFICATION, PREVIEW_GENERATION
- **FR-002**: System MUST define default model specifications for each stage based on research.md decisions
- **FR-003**: Each stage MUST have a primary model and zero or more fallback models
- **FR-004**: System MUST provide a `ModelSelector` class that manages model specifications, resolves models at runtime, and tracks usage
- **FR-005**: System MUST provide `get_model(stage)` returning the primary model name for a stage
- **FR-006**: System MUST provide `get_fallback_chain(stage)` returning all models for a stage in priority order
- **FR-007**: System MUST integrate with existing Config class via `ModelSelector.from_config(config)`
- **FR-008**: The single `--model` CLI flag MUST set the primary model for all AI stages (backward compatible)
- **FR-009**: Per-stage model overrides MUST be configurable via environment variables (`BOURBON_<STAGE>_MODEL`)
- **FR-010**: System MUST track usage per invocation: stage, model, success, duration, tokens, error, fallback_used
- **FR-012**: System MUST provide `get_usage_summary()` returning a dictionary with total_calls, successful_calls, fallback_calls, total_duration, by_stage, by_model, and per-record details
- **FR-013**: System MUST provide `print_stage_report()` that logs the primary model, fallback chain, and description for each stage
- **FR-014**: System MUST provide `record_usage()` method to log individual model invocations
- **FR-015**: System MUST provide `is_model_available(model_name)` to check if a model is in any registered chain
- **FR-016**: System MUST provide `create_default_selector()` convenience function
- **FR-017**: System MUST provide `get_stage_models(stage)` convenience function returning all model names for a stage
- **FR-018**: System MUST provide a `ModelServerManager` that manages the full lifecycle of model serving (start, stop, health-check, pull, load/unload, resource monitoring)
- **FR-019**: System MUST support two serving backends via `ServerBackend` enum: `MPLTX` (native MPLTX process management) and `OLLAMA` (Ollama with auto-start, backward compatible)
- **FR-020**: The `MpltxServer` backend MUST manage an `mpltx` process: start with configurable port, GPU layers, context size, keep-alive; stop gracefully (SIGTERM then SIGKILL); health-check via `/api/tags`; pull models via `/api/pull`; warm-load via `/api/generate`; cold-start detection
- **FR-021**: The `OllamaServer` backend MUST auto-start `ollama serve` if not already running, with configurable `--ollama-url` (backward compatible with existing `--ollama-url` CLI flag)
- **FR-022**: System MUST provide `ModelServerManager.ensure_running()`, `ensure_model(name)`, `ensure_models(names)`, `get_client()`, `get_state()`, `get_resource_usage()`, `list_models()`, and `close()` methods
- **FR-023**: System MUST provide `ServerState` dataclass tracking: backend, is_running, base_url, pid, models_loaded, memory_usage_mb, gpu_usage_percent, cold_start_seconds, uptime_seconds
- **FR-024**: System MUST integrate with existing Config class: Config MUST support `mpltx_home`, `mpltx_port`, `mpltx_gpu_layers`, `mpltx_context_size`, `server_backend` fields
- **FR-025**: System MUST integrate with existing CLI: pre-flight checks MUST use `ModelServerManager.ensure_models()` instead of bare `OllamaClient.health_check()`; `--model` flag MUST set primary model for all AI stages in the selector
- **FR-026**: System MUST integrate with BottleAnalyzer and BatchProcessor: both MUST accept a `ModelSelector` instance and use it for per-stage model resolution

### Key Entities

- **ModelStage**: Enum of all 8 pipeline stages that may use models
- **ModelSpec**: Specification for a single model (name, stage, role, priority, max_tokens, temperature, description)
- **ModelSelector**: Central class managing model registry, resolution, and usage tracking
- **UsageLog**: Accumulator for ModelUsage records with summary statistics
- **ModelUsage**: Single model invocation record (stage, model, success, duration, tokens, error, fallback_used)
- **ModelServerManager**: Facade managing model serving lifecycle (start, stop, pull, load, health-check, resource monitoring)
- **ServerBackend**: Enum of serving backends (MPLTX, OLLAMA)
- **BaseModelServer**: Abstract interface for model serving backends
- **MpltxServer**: Concrete MPLTX process management backend
- **OllamaServer**: Concrete Ollama auto-start backend (backward compatible)
- **ServerState**: Dataclass tracking server state (backend, is_running, pid, models_loaded, memory, GPU, uptime)

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: All 8 pipeline stages have a defined primary model and at least one fallback (deterministic or AI)
- **SC-002**: Default configuration uses the models specified in research.md for all stages
- **SC-003**: Backward compatibility: existing `--model` flag continues to work, setting primary model for all AI stages
- **SC-004**: Per-stage model overrides via environment variables take precedence over the single `--model` flag
- **SC-005**: Usage tracking captures 100% of model invocations with stage, model, success, and duration
- **SC-006**: Usage summary provides accurate counts grouped by stage and by model name
- **SC-007**: Model selector integrates with existing Config class without breaking changes
- **SC-008**: Model selector integrates with existing OllamaClient without breaking changes

---

## Assumptions

- Research.md decisions (Qwen3-VL 32B primary, GPT-OSS 120B text, Tesseract OCR) are the authoritative source for default models
- Ollama is the local model server; models must be pulled before use
- Non-AI stages (ORB, hash-based, rule-based) are always available as system dependencies
- The single `--model` CLI flag is the primary user-facing configuration; per-stage overrides are advanced
- Usage tracking is optional (can be disabled for performance-critical deployments)