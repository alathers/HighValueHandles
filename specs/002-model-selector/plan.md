# Implementation Plan: Model Selector

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

**Branch**: `002-model-selector` | **Date**: 2026-11-21 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-model-selector/spec.md`

## Summary

Instrument per-stage model selection, fallback chains, usage tracking, and model serving lifecycle management (MPLTX + Ollama) for the bourbon image processor. The model selector assigns the right model to each pipeline stage, manages fallback chains, tracks usage, and the serving manager handles start/stop/pull/health-check.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies** (existing, no new):
- Requests: HTTP client for model serving APIs
- Pillow/PIL: Image processing
- OpenCV (cv2): Computer vision utilities
- NumPy: Numerical operations
- Pytesseract: OCR integration

**Storage**: File system (existing `processed/` directory)

**Testing**: pytest (existing in project)

**Target Platform**: Local CLI tool (macOS, Linux, Windows)

**Constraints**:
- Local-only processing (MPLTX or Ollama with local models)
- MPLTX backend is preferred; Ollama backend is backward compatible
- Existing `--model` CLI flag must continue to work (sets primary for all AI stages)
- Per-stage model overrides via environment variables (`BOURBON_<STAGE>_MODEL`)
- Existing `OllamaClient` API must not break — `ModelServerManager` wraps it

## Constitution Check

**I. Python + uv Environment Management**:
- ✓ No new dependencies required — uses existing `requests` library
- ✓ MPLTX binary expected to be in PATH or `MPLTX_HOME/bin/` (user responsibility)

**II. Test-First (NON-NEGOTIABLE)**:
- ✓ Unit tests for `ModelSelector`, `ModelServerManager`, `MpltxServer`, `OllamaServer`
- ✓ Contract tests for CLI integration
- ✓ Integration tests for full pipeline with model selector

**III. Documentation Standards**:
- ✓ Google-style docstrings for all public functions
- ✓ Inline comments explaining non-obvious logic (process management, cold-start)
- ✓ Spec.md, plan.md, tasks.md in `specs/002-model-selector/`

**IV. Git Workflow Governance**:
- ✓ Feature branch: `002-model-selector` (per existing pattern)

**V. Quality Gates (NON-NEGOTIABLE)**:
- ✓ Pre-commit hooks for linting and large file blocking
- ✓ All tests must pass

## Project Structure

### Documentation (this feature)

```text
specs/002-model-selector/
├── spec.md              # This file (feature specification)
├── plan.md              # This file (implementation plan)
└── tasks.md             # Phase 2 output (task breakdown)
```

### Source Code (repository root)

```text
src/
├── lib/                      # Shared utilities (new files)
│   ├── model_selector.py     # Per-stage model selection + usage tracking (NEW)
│   ├── model_server.py       # Model serving lifecycle management (NEW)
│   ├── config.py             # Updated: add MPLTX fields
│   ├── ollama_client.py      # Unchanged (backward compatible)
│   └── error_handling.py     # Unchanged
├── image_processor/          # Domain modules (updated)
│   ├── bottle_analyzer.py    # Updated: accept ModelSelector
│   ├── batch_processor.py    # Updated: accept ModelSelector
│   ├── text_extractor.py     # Unchanged
│   └── preview_generator.py  # Unchanged
├── cli/                      # CLI interface (updated)
│   └── commands.py           # Updated: use ModelServerManager in pre-flight
└── models/                   # Data models (unchanged)
    └── bottle.py

tests/
├── unit/
│   ├── test_model_selector.py    # Unit tests for ModelSelector (NEW)
│   └── test_model_server.py      # Unit tests for ModelServerManager (NEW)
└── contract/
    └── test_cli_model_integration.py  # CLI integration tests (NEW)
```

## Complexity Tracking

> No constitution violations — all gates pass.

## Implementation Phases

### Phase 1: Core Library (Foundation)

Create the two new library modules:

1. **`src/lib/model_selector.py`** — Per-stage model selection with fallback chains and usage tracking
   - 8 `ModelStage` enum values
   - Default model specs from `research.md` (Tesseract, Qwen3-VL 32B, GPT-OSS 120B, ORB, hash-based, rule-based, template)
   - `ModelSelector` with `get_model()`, `get_fallback_chain()`, `from_config()`, `record_usage()`, `get_usage_summary()`, `print_stage_report()`
   - `UsageLog` with per-invocation records and summary-by-stage / summary-by-model

2. **`src/lib/model_server.py`** — Model serving lifecycle management
   - `ServerBackend` enum (MPLTX, OLLAMA)
   - `BaseModelServer` abstract interface
   - `MpltxServer` — native MPLTX process management (start, stop, health-check, pull, load/unload, resource monitoring, cold-start)
   - `OllamaServer` — Ollama with auto-start (backward compatible)
   - `ModelServerManager` — facade with `ensure_running()`, `ensure_model()`, `ensure_models()`, `get_client()`, context manager support

### Phase 2: Configuration Integration

Update existing modules to use the new libraries:

1. **`src/lib/config.py`** — Add MPLTX-specific config fields:
   - `server_backend` (enum string: "mpltx" or "ollama")
   - `mpltx_home` (path to MPLTX data directory)
   - `mpltx_port` (HTTP API port, default 11435)
   - `mpltx_gpu_layers` (GPU offload, default -1 = all)
   - `mpltx_context_size` (context window, default 4096)

2. **`src/cli/commands.py`** — Update pre-flight checks:
   - Replace bare `OllamaClient.health_check()` with `ModelServerManager.ensure_models()`
   - Create `ModelServerManager` from config (backend from config, or default MPLTX)
   - Ensure all required models before processing starts
   - Print model selector stage report in debug mode

3. **`src/image_processor/bottle_analyzer.py`** — Update to accept `ModelSelector`:
   - Accept optional `model_selector` parameter in `__init__`
   - Use `model_selector.get_model(stage)` for per-stage model resolution
   - Record usage for each AI call

4. **`src/image_processor/batch_processor.py`** — Update to accept `ModelSelector`:
   - Accept optional `model_selector` parameter in `__init__`
   - Pass selector through to `BottleAnalyzer`

### Phase 3: Tests

1. **`tests/unit/test_model_selector.py`** — Unit tests for `ModelSelector`:
   - Default model specs from research.md
   - `get_model()`, `get_fallback_chain()`, `from_config()`
   - Usage tracking: `record_usage()`, `get_usage_summary()`, `summary_by_stage()`, `summary_by_model()`
   - `print_stage_report()` output validation

2. **`tests/unit/test_model_server.py`** — Unit tests for `ModelServerManager`:
   - `MpltxServer` start/stop, health-check, pull, load/unload (mocked HTTP)
   - `OllamaServer` start/stop, health-check (mocked HTTP)
   - `ModelServerManager.ensure_model()`, `ensure_models()`, `get_client()`
   - Context manager usage

3. **`tests/contract/test_cli_model_integration.py`** — CLI integration tests:
   - `--model` flag sets primary model for all AI stages
   - `BOURBON_<STAGE>_MODEL` env vars override per-stage
   - Pre-flight uses `ModelServerManager`
   - Debug mode prints stage report

## Implementation Strategy

**Iteration 1** (Core Library): Create `model_selector.py` and `model_server.py`
- Validates core hypothesis: can we assign the right model to each stage?
- Provides immediate value for per-stage optimization

**Iteration 2** (Integration): Update config, CLI, bottle_analyzer, batch_processor
- Existing pipeline works with model selector integrated
- Backward compatible: existing `--model` flag continues to work

**Iteration 3** (Tests): Unit + contract + integration tests
- Validates correctness, backward compatibility, and full pipeline

**Risk Mitigation**:
- **Backward compatibility**: Existing `--model` flag and `--ollama-url` continue to work
- **MPLTX binary**: If not found, falls back to Ollama backend (configurable)
- **Model availability**: Fallback chains handle missing models gracefully
- **No new dependencies**: Uses existing `requests` library

**Estimated effort**: ~30 tasks, 3 phases

**Status**: ALL PHASES COMPLETE — Core Library (Phase 1), Configuration Integration (Phase 2), and Tests (Phase 3) fully implemented.

---

**Total Tasks**: 30
**Core Library Tasks**: 13
**Integration Tasks**: 10
**Test Tasks**: 7
**All tasks marked complete in tasks.md**

**Format Validation**: All tasks follow checklist format (- [ ] TaskID [P?] [Story?] Description with file path)