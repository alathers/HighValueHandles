# Implementation Tasks: Model Selector

> **Superseded (2026-11-23).** This document describes the original design;
> [README.md](../../README.md) describes the code as it is now:
>
> - Ollama is the only model-serving backend. The MTPLX/MPLTX backend
>   (`MpltxServer`, `ServerBackend`, `mpltx_*` settings) and the
>   `--server-backend` flag were removed; MTPLX runs only on Apple Silicon and
>   this tool must stay portable to Linux.
> - ORB/visual-similarity grouping was removed. Photos are grouped by EXIF
>   capture time, and the vision model's same-bottle answer splits mixed groups.
>   There is no `IMAGE_GROUPING` model stage.
> - Only two stages call a model: `VISION_ANALYSIS` (default `qwen3.6:35b-a3b`)
>   and `METADATA_GENERATION`, the gpt-oss fact check (default `gpt-oss:120b`;
>   `--text-model`, `--no-verify`).
> - There are no fallback models or fallback chains; a missing model stops the
>   run at pre-flight.
> - `WEB_RESEARCH` was removed as a model stage: web search is not a model. Web
>   lookups are a separate, non-model step (`--web`, `--web-provider`).

**Feature**: Model Selector & Serving Manager
**Branch**: `002-model-selector`
**Generated**: 2026-11-21
**Status**: Ready for Implementation

---

## Phase 1: Core Library (Foundation)

**Goal**: Create `model_selector.py` and `model_server.py` with per-stage model selection, fallback chains, usage tracking, and model serving lifecycle management (MPLTX + Ollama).

### Independent Test Criteria
- `ModelSelector` returns correct primary model and fallback chains for all 8 stages
- Default specs match research.md decisions (Tesseract, Qwen3-VL 32B, GPT-OSS 120B, ORB, hash-based, rule-based, template)
- `ModelServerManager` starts/stops both MpltxServer and OllamaServer backends
- `ModelServerManager.ensure_model()` pulls and loads models
- Usage tracking records and summarizes all model invocations

### Implementation Tasks

- [X] T139 [P1] Create `src/lib/model_selector.py` with `ModelStage` enum (8 stages: OCR_EXTRACTION, VISION_ANALYSIS, IMAGE_GROUPING, QUANTITY_DETECTION, METADATA_GENERATION, WEB_RESEARCH, ERROR_CLASSIFICATION, PREVIEW_GENERATION)
- [X] T140 [P1] Create `src/lib/model_selector.py` `ModelSpec` dataclass (name, stage, role, priority, max_tokens, temperature, description) with validation
- [X] T141 [P1] Create `src/lib/model_selector.py` `ModelUsage` dataclass (stage, model, success, duration_seconds, input_tokens, output_tokens, error, fallback_used)
- [X] T142 [P1] Create `src/lib/model_selector.py` `UsageLog` class with `record()`, `total_calls`, `successful_calls`, `fallback_calls`, `total_duration`, `summary_by_stage()`, `summary_by_model()`, `to_dict()`
- [X] T143 [P1] Create `src/lib/model_selector.py` default model specifications from research.md: Tesseract for OCR, Qwen3-VL 32B for vision, GPT-OSS 120B for metadata, ORB for grouping, hash-based for quantity, DuckDuckGo for web research, rule-based for error, template for preview
- [X] T144 [P1] Create `src/lib/model_selector.py` `ModelSelector` class with `__init__(models, default_model, allow_fallback, log_usage)`, `from_config(config)`, `get_model(stage)`, `get_fallback_chain(stage)`, `get_stage_description(stage)`
- [X] T145 [P1] Implement `ModelSelector.record_usage()` and `ModelSelector.get_usage_summary()` for usage tracking and reporting
- [X] T146 [P1] Implement `ModelSelector.is_model_available()` and `ModelSelector.print_stage_report()` for model checking and stage reporting
- [X] T147 [P1] Add `create_default_selector()` and `get_stage_models(stage)` convenience functions to `src/lib/model_selector.py`
- [X] T148 [P1] Create `src/lib/model_server.py` with `ServerBackend` enum (MPLTX, OLLAMA) and `ServerState` dataclass (backend, is_running, base_url, pid, models_loaded, memory_usage_mb, gpu_usage_percent, cold_start_seconds, uptime_seconds)
- [X] T149 [P1] Create `src/lib/model_server.py` `BaseModelServer` abstract class with: `start()`, `stop()`, `health_check()`, `get_state()`, `is_model_available()`, `pull_model()`, `list_models()`, `get_resource_usage()`, `load_model()`, `unload_model()`, `get_client()`
- [X] T150 [P1] Create `src/lib/model_server.py` `MpltxServer` class: process management (start with configurable port/GPU/context/keep-alive, stop with SIGTERM/SIGKILL, health-check via `/api/tags`, pull via `/api/pull`, warm-load via `/api/generate`, resource monitoring via `/api/usage` or psutil)
- [X] T151 [P1] Create `src/lib/model_server.py` `OllamaServer` class: auto-start `ollama serve` if not running, health-check, pull, list, backward compatible with existing `--ollama-url`
- [X] T152 [P1] Create `src/lib/model_server.py` `ModelServerManager` facade: `ensure_running()`, `ensure_model(name)`, `ensure_models(names)`, `get_client()`, `get_state()`, `get_resource_usage()`, `list_models()`, `close()`, context manager (`__enter__`/`__exit__`), `from_config(config)`

---

## Phase 2: Configuration & Pipeline Integration

**Goal**: Update existing modules to use the new libraries without breaking backward compatibility.

### Independent Test Criteria
- Config class supports `server_backend`, `mpltx_home`, `mpltx_port`, `mpltx_gpu_layers`, `mpltx_context_size`
- CLI pre-flight uses `ModelServerManager.ensure_models()` instead of bare `OllamaClient.health_check()`
- Existing `--model` flag continues to work (sets primary for all AI stages)
- BottleAnalyzer accepts optional `model_selector` parameter
- BatchProcessor accepts optional `model_selector` parameter

### Implementation Tasks

- [X] T153 [P2] Update `src/lib/config.py` — Add fields: `server_backend` (str, default "mpltx"), `mpltx_home` (str, default "~/.mpltx"), `mpltx_port` (int, default 11435), `mpltx_gpu_layers` (int, default -1), `mpltx_context_size` (int, default 4096)
- [X] T154 [P2] Update `src/lib/config.py` — Add environment variable support: `BOURBON_SERVER_BACKEND`, `BOURBON_MPLTX_HOME`, `BOURBON_MPLTX_PORT`, `BOURBON_MPLTX_GPU_LAYERS`, `BOURBON_MPLTX_CONTEXT_SIZE`
- [X] T155 [P2] Update `src/cli/commands.py` — Add `--server-backend` CLI flag (choices: "mpltx", "ollama", default "mpltx")
- [X] T156 [P2] Update `src/cli/commands.py` — In `pre_flight_checks()`: replace bare `OllamaClient.health_check()` with `ModelServerManager.ensure_models()` using backend from config
- [X] T157 [P2] Update `src/cli/commands.py` — In `pre_flight_checks()`: use `ModelServerManager.ensure_model()` to check each required model (qwen3-vl:32b, qwen2-vl:7b, gpt-oss:120b)
- [X] T158 [P2] Update `src/cli/commands.py` — In `pre_flight_checks()`: print model selector stage report in debug mode (`--debug`)
- [X] T159 [P2] Update `src/cli/commands.py` — In `process_command()`: create `ModelServerManager` from config, ensure all required models, pass `ModelSelector` to `BottleAnalyzer`
- [X] T160 [P2] Update `src/image_processor/bottle_analyzer.py` — Add optional `model_selector: ModelSelector | None = None` parameter to `__init__`
- [X] T161 [P2] Update `src/image_processor/bottle_analyzer.py` — Use `model_selector.get_model(ModelStage.VISION_ANALYSIS)` instead of hardcoded `qwen2-vl:7b` in `process_single_bottle()` and `analyze_with_vision_model()`
- [X] T162 [P2] Update `src/image_processor/bottle_analyzer.py` — Record usage for each AI call: `model_selector.record_usage(stage, model, success, duration, input_tokens, output_tokens, error, fallback_used)`
- [X] T163 [P2] Update `src/image_processor/batch_processor.py` — Add optional `model_selector: ModelSelector | None = None` parameter to `__init__`
- [X] T164 [P2] Update `src/image_processor/batch_processor.py` — Pass `model_selector` to `BottleAnalyzer` when creating it
- [X] T165 [P2] Update `src/cli/commands.py` — In `retry_command()`: use `ModelServerManager` instead of bare `OllamaClient` for re-processing

---

## Phase 3: Tests

**Goal**: Unit tests for model_selector and model_server, contract tests for CLI integration.

### Independent Test Criteria
- All 8 stages have correct default primary model and fallback chain
- Usage tracking records and summarizes correctly
- MpltxServer start/stop/health-check/pull/load/unload work (mocked HTTP)
- OllamaServer start/stop/health-check work (mocked HTTP)
- ModelServerManager ensure_model/ensure_models/get_client work (mocked)
- CLI `--model` flag sets primary model for all AI stages
- Per-stage env vars override `--model` flag
- Pre-flight uses ModelServerManager

### Implementation Tasks

- [X] T166 [P3] Create `tests/unit/test_model_selector.py` — Test `ModelStage` enum values (8 stages), `ModelSpec` validation, default specs match research.md
- [X] T167 [P3] Test `ModelSelector.get_model()`, `get_fallback_chain()`, `get_stage_description()`, `from_config()` (with and without per-stage overrides)
- [X] T168 [P3] Test `UsageLog.record()`, `total_calls`, `successful_calls`, `fallback_calls`, `total_duration`, `summary_by_stage()`, `summary_by_model()`, `to_dict()` (JSON-serializable)
- [X] T169 [P3] Test `ModelSelector.record_usage()`, `get_usage_summary()`, `is_model_available()`, `print_stage_report()` (output validation)
- [X] T170 [P3] Create `tests/unit/test_model_server.py` — Test `ServerBackend` enum, `ServerState` dataclass
- [X] T171 [P3] Test `MpltxServer.start()` (success + failure), `stop()` (graceful + force), `health_check()`, `pull_model()`, `load_model()`, `unload_model()` (mocked HTTP responses)
- [X] T172 [P3] Test `OllamaServer.start()` (success + failure), `stop()`, `health_check()` (mocked HTTP), `ModelServerManager.ensure_model()`, `ensure_models()`, `get_client()`, context manager usage (mocked)
- [X] T173 [P3] Create `tests/contract/test_cli_model_integration.py` — Test `--model` flag sets primary model for all AI stages, `BOURBON_<STAGE>_MODEL` env vars override per-stage, pre-flight uses `ModelServerManager`, debug mode prints stage report

---

## Dependencies

```
Phase 1 (Core Library) ← Phase 2 (Integration) ← Phase 3 (Tests)
```

**Rationale**:
- Phase 1 creates the foundational libraries that Phase 2 integrates
- Phase 3 tests both the libraries and their integration

---

## Implementation Strategy

### MVP Scope
**Recommended**: Complete Phase 1 (Core Library) + Phase 2 (Integration)

This enables:
- Per-stage model selection with fallback chains
- Usage tracking and reporting
- MPLTX model serving lifecycle management
- Backward compatible CLI (`--model` flag still works)
- Automatic fallback when preferred model unavailable

**Estimated effort**: ~23 tasks (Phase 1 + 2), ~40% of total task count

### Incremental Delivery Sequence

1. **Iteration 1** (Core Library): Phase 1 (T139-T152)
   - Validates core hypothesis: can we assign the right model to each stage?
   - Provides immediate value for per-stage optimization

2. **Iteration 2** (Integration): Phase 2 (T153-T165)
   - Existing pipeline works with model selector integrated
   - Backward compatible: existing `--model` flag continues to work

3. **Iteration 3** (Tests): Phase 3 (T166-T173)
   - Validates correctness, backward compatibility, and full pipeline

### Risk Mitigation

- **Backward compatibility**: Existing `--model` flag and `--ollama-url` continue to work
- **MPLTX binary**: If not found, falls back to Ollama backend (configurable via `server_backend`)
- **Model availability**: Fallback chains handle missing models gracefully
- **No new dependencies**: Uses existing `requests` library
- **Process management**: SIGTERM/SIGKILL fallback ensures clean shutdown even if server is unresponsive

---

**Total Tasks**: 35 (14 existing + 21 new)
**Core Library Tasks (new)**: 14 (T139-T152) — COMPLETE
**Integration Tasks (new)**: 13 (T153-T165) — COMPLETE
**Test Tasks (new)**: 8 (T166-T173) — COMPLETE

**Format Validation**: All tasks follow checklist format (- [X] TaskID [P?] [Story?] Description with file path)
**Status**: ALL TASKS COMPLETE — Phase 1, 2, and 3 fully implemented