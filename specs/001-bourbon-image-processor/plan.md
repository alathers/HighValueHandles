# Implementation Plan: Bourbon Image Processor

> **Superseded in part (2026-11-23).** This document describes the original
> design. The code has since changed; [README.md](../../README.md) describes what it
> does now. In particular:
>
> - Lookahead and ORB/visual-similarity grouping were replaced by EXIF
>   capture-time grouping plus the vision model's same-bottle split.
>   `--lookahead` was removed (it now fails with exit code 2).
> - Interactive merge and split (`[M]`/`[S]`) were removed. Review actions are
>   `[A]ccept`, `[R]eject`, `[E]dit`, `[C]onflicts`, `[N]ext` and `[Q]uit`.
> - `ServerBackend` and MTPLX were removed; Ollama is the only backend, and
>   `--server-backend` / `BOURBON_SERVER_BACKEND` no longer exist.
> - T100-T109 (vendor cache) are superseded by the web-lookup design: search
>   providers (DuckDuckGo by default, or Brave, Ollama web search, SearXNG),
>   verbatim quotes verified by code, a vendor image verified by the vision
>   model (`-v`), and no OpenCV image matching.
> - FR-018 ("retry up to 3 times") now means up to `retry.max_attempts`
>   (`--retry-attempts`, default 3) attempts per model request, not per bottle.
> - Configuration is the sectioned `bourbon.toml` (plus a user file, `BOURBON_*`
>   variables and flags), shown by `bourbon-process config show`.
> - MSRP is never taken from a price tag; the gpt-oss fact check and a publish
>   gate with human approval decide what is publishable.

**Branch**: `001-bourbon-image-processor` | **Date**: 2026-11-09 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-bourbon-image-processor/spec.md`

## Summary

Extract bourbon bottle metadata (name, distillery, MSRP, release information, tasting notes) from images using AI-powered OCR and visual analysis. Group related images, detect quantities, output structured markdown files, organize in processed sub-folders, support configurable batch processing with preview reports for incremental feedback, and optionally cache vendor reference images.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**:
- Requests: HTTP client for Ollama API
- Pillow/PIL: Image processing (orientation correction, resizing)
- OpenCV (cv2): Computer vision utilities for image analysis
- NumPy: Numerical operations
- Pytesseract: OCR integration for text extraction
- Tesseract-OCR: System OCR engine (requires external installation)

**Storage**: File system (processed/ sub-directories with bottle folders containing images, markdown, extracted text files)

**Testing**: pytest (existing in project)

**Target Platform**: Local CLI tool (macOS, Linux, Windows)

**Project Type**: CLI tool/AI-powered image processing

**Performance Goals**:
- 60 seconds per bottle set for extraction and markdown generation
- 95%+ accuracy on clearly labeled bottles (name, distillery, proof, age)
- 10 bottles processed in under 10 minutes

**Constraints**:
- Local-only processing (Ollama with local models)
- Optional vendor image caching must not block processing
- Configurable batch size (default 5, configurable, "all" for full-folder)
- Lookahead of 3 images to catch multi-angle photos
- Optional interactive CLI mode for preview review

**Scale/Scope**: Individual inventor collections (10s to 100s of bottles), not enterprise-scale

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution Version: 1.0.0 (Ratified 2026-11-09)

### Pre-Phase 0 Evaluation: PASS ✓
Constitution was placeholder template only; no active constraints to evaluate.

### Post-Phase 1 Evaluation: PASS ✓

**I. Python + uv Environment Management**:
- ✓ Python 3.11+ specified as implementation language
- ✓ dependencies defined for uv sync (Requests, Pillow, OpenCV, NumPy, Pytesseract, Tesseract-OCR system, pytest)
- ✓ Environment management aligns with uv workflow

**II. Test-First (NON-NEGOTIABLE)**:
- ✓ pytest identified as testing framework
- ✓ Test structure defined: contract/, integration/, unit/ tiers
- ✓ Testing targets aligned with success criteria (accuracy, performance)

**III. Documentation Standards**:
- ✓ Comprehensive design artifacts: research.md (16 technical decisions), data-model.md (full entity definitions), contracts/cli-interface.md (complete CLI specification), quickstart.md (8 validation scenarios)
- TODO: Create "ADHD version" distilled documentation during implementation phase
- ✓ Spec, plan, contracts suitable for deep understanding

**IV. Git Workflow Governance**:
- ✓ Branch naming convention: `001-bourbon-image-processor`
- ✓ Design artifacts tracked in branch
- TODO: Verify worktree-based development during implementation tasks
- ✓ No main commits implied by branch-based workflow

**V. Quality Gates (NON-NEGOTIABLE)**:
- TODO: Pre-commit hooks for linting and large file blocking to be configured in implementation
- TODO: Define file size thresholds in pre-commit configuration
- ✓ Quality gates listed as implementation requirement in constitution

**Tooling Requirements**:
- TODO: Cross-platform `install.sh` script to be created in implementation phase
- ✓ Tesseract-OCR noted as external system dependency requiring installation

**Development Workflow - Git Worktree Development**:
- TODO: Verify worktree creation during task execution
- Current plan merges into main - must follow worktree isolation policy per constitution

All constitutional gates PASS - ready for /speckit-tasks to break down implementation work.

## Project Structure

### Documentation (this feature)

```text
specs/001-bourbon-image-processor/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
src/
├── image_processor/          # Main processing module
│   ├── __init__.py
│   ├── bottle_analyzer.py    # Bottle detection and metadata extraction
│   ├── image_organizer.py   # Image grouping and quantity detection
│   ├── preview_generator.py # Markdown preview report generation
│   ├── vendor_cache.py      # Optional vendor image lookup and caching
│   ├── text_extractor.py     # OCR and text extraction with orientation correction
│   └── batch_processor.py    # Batch orchestration, lookahead, feedback loop
├── models/                   # Data models
│   ├── __init__.py
│   └── bottle.py             # Bottle, Image, ExtractedMetadata entities
├── cli/                      # Command-line interface
│   ├── __init__.py
│   └── commands.py           # CLI argument parsing and commands
└── lib/                      # Shared utilities
    ├── __init__.py
    ├── ollama_client.py      # Ollama API wrapper
    ├── config.py             # Configuration management
    └── error_handling.py     # Error reports, hints.md, retry logic

HighValueRoom/                # Existing implementation (source of dependencies)
└── (existing scraper.py and related code)

tests/
├── contract/                 # Contract tests for CLI interface and processing pipeline
├── integration/              # End-to-end tests with real images
└── unit/                     # Unit tests for individual modules

processed/                    # Output directory (created at runtime)
└── {bottle_name}_{timestamp}/  # Sub-folders per bottle set
    ├── {original_image_files}
    ├── bottle.md              # Structured metadata markdown
    ├── extracted_text.txt     # Raw extracted text
    ├── vendor_ref.jpg (optional)
    ├── error_report.json (if failed)
```

**Structure Decision**: Single CLI project option selected. Feature modules organized by domain concern (bottle analysis, image organizing, preview generation) with shared utilities for Ollama integration, configuration, and error handling. Testing includes contract tests for CLI interface, integration tests for end-to-end processing, and unit tests for individual modules. Existing HighValueRoom/ directory contains current implementation (scraper.py) which will be refactored into the new structure.

## Complexity Tracking

> No constitution violations to justify - all gates pass.
