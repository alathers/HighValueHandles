# Research: Bourbon Image Processor

**Feature**: Bourbon Image Processor
**Date**: 2026-11-09
**Status**: Complete

## Overview

This document consolidates research findings and technical decisions for implementing an AI-powered bourbon bottle image processor that extracts metadata, organizes output files, supports configurable batch processing, and optionally caches vendor reference images.

---

## Decision 1: Programming Language and Framework

**Decision**: Python 3.11+

**Rationale**:
- Strong ecosystem for AI/ML and computer vision (PyTorch, OpenCV, Pillow)
- Excellent OCR integration via pytesseract
- Existing codebase (scraper.py) is Python
- High reliability and maintainability
- Good async support for concurrent API calls to Ollama

**Alternatives Considered**:
- Node.js/TypeScript: Good for asynchronous workflows, weaker OCR and computer vision libraries
- Go: Strong performance and concurrency, immature AI/ML ecosystem
- Rust: High performance, but steep learning curve and limited ML ecosystem

**Impact**: Defines dependency management, deployment model, team expertise requirements

---

## Decision 2: AI Model Integration Strategy

**Decision**: Local-only processing using Ollama for model management

**Rationale**:
- No recurring cloud API costs
- Data privacy (images never leave local machine)
- Existing Ollama setup in project
- Python requests library provides simple HTTP client for Ollama REST API
- Models remain available offline after initial download
- Control over model versions and updates

**Alternatives Considered**:
- Cloud APIs (OpenAI GPT-4V, Anthropic Claude 3.5 Sonnet Vision): Higher quality but requires subscriptions, internet connectivity, and sends images externally
- Direct HuggingFace Transformers integration: More complex deployment, requires GPU for acceptable performance
- Azure Computer Vision / AWS Rekognition: Cloud-dependent, vendor lock-in

**Models Selected**:
- **Qwen3-VL 32B**: Primary vision model for image analysis and OCR-like extraction
- **GPT-OSS 120B**: Text model for metadata generation from extracted text
- **Tesseract-OCR**: System OCR engine for backup text extraction and handling edge cases

**Model Selection Rationale**:
- Qwen3-VL 32B provides strong multimodal understanding and can read text directly from images
- GPT-OSS 120B handles text-to-text transformations (raw analysis → structured metadata)
- Tesseract provides fallback for cases where vision model struggles with OCR
- All models are available locally via Ollama

**Fallback Strategy**:
- If Tesseract OCR is installed, use it as primary text extraction before vision model analysis
- Use vision model to validate and augment OCR results
- Combine OCR + vision model outputs for comprehensive extraction

**Impact**: Defines runtime dependencies (Ollama service), model download requirements, processing performance

---

## Decision 3: Image Processing and Orientation Detection

**Decision**: OpenCV for orientation detection, Pillow/PIL for image manipulation

**Rationale**:
- OpenCV provides robust algorithms for text orientation detection (EAST, FastText)
- Pillow/PIL offers simple image rotation, resizing, and format conversion
- OpenCV + Pillow established pattern in Python computer vision projects
- Both libraries integrate well with Tesseract OCR (which requires properly oriented images)

**Orientation Detection Strategy**:
1. Detect text regions in image using Edge Detection
2. Compute angle of text lines
3. Rotate image to correct orientation (0°, 90°, 180°, 270°)
4. Validate rotation by attempting OCR and checking text quality (character recognition rate)
5. Fall back to default if no clear orientation detected

**Alternatives Considered**:
- Tesseract only orientation detection: Less accurate for complex layouts
- Deep learning models (tensorflow, pytorch): Overkill for this use case, higher complexity
- No orientation correction: Would violate FR-016

**Impact**: Defines image processing pipeline, affects OCR accuracy, adds processing time

---

## Decision 4: Image Grouping and Quantity Detection

**Decision**: Multi-stage approach combining visual similarity, text matching, and batch number extraction

**Strategy**:

1. **Initial Grouping by Visual Similarity**:
   - Extract visual features using OpenCV (SIFT/ORB feature extraction)
   - Compute similarity scores between consecutive images
   - Group images above threshold into tentative bottle sets

2. **Text Verification**:
   - Extract text from all images in tentative groups
   - Compare bottle names, distilleries, proofs
   - Extract batch numbers, barrel numbers for distinguishing similar bottles
   - Group confirmed by text + visual similarity

3. **Quantity Detection**:
   - Detect exact duplicate images (hash comparison) → count as quantity
   - Detect multiple identical bottles in single frame (object detection) → infer quantity
   - Always track quantity field in metadata (default 1)

4. **Sequential Photo Handling**:
   - Similar consecutive photos assumed DISTINCT (conservative)
   - Differentiating features (batch numbers, proof variations) create separate bottles
   - User can manually merge if needed

**Edge Case Handling**:
- Multiple MSRPs on images: Extract all, mark as ambiguous
- Batch number text: Critical for distinguishing similar bottles
- Multi-angle photos of same bottle: Lookahead of 3 images to catch related photos before batch cutoff

**Alternatives Considered**:
- Clustering based on embeddings: More complex, may over-split bottles with different photos
- Manual user grouping: Too labor-intensive for 10-100+ bottle workflows
- Simple timestamp grouping: Unreliable (user拍照顺序 may vary)

**Impact**: Defines algorithm complexity for bottle organization, affects accuracy of quantity tracking

---

## Decision 5: Batch Processing Architecture

**Decision**: Incremental per-batch processing with lookahead and configurable sizing

**Design**:

1. **Default Batch Size**: 5 bottles
2. **Lookahead Window**: 3 images beyond batch to catch multi-angle photos
3. **Batch Size Configurability**: CLI argument `--batch-size N` or `--all` for full-folder processing
4. **Preview Report Generation**: After each batch (except when `--all` flag used)
5. **User Feedback Loop**:
   - Preview report generated in markdown format
   - User can provide feedback (correct groupings, fix extractions)
   - System re-processes only flagged items
   - Apply learnings to subsequent batches

**Processing Flow**:
```
1. Load images from folder
2. Group images into bottle sets (image_organizer.py)
3. Extract metadata per bottle (bottle_analyzer.py)
4. Generate preview report (markdown + optional CLI interactive mode)
5. Wait for user feedback (optional)
6. Write bottle files to processed/ sub-folders
7. Clear metadata from memory
8. Continue to next batch
```

**Alternatives Considered**:
- Process all images at once: No opportunity for user feedback, cannot learn and improve mid-run
- Real-time streaming: Too fragmented for coherent feedback
- Fixed batch size only: Less flexibility for varying collection sizes

**Impact**: Defines user experience, workflow efficiency, ability to course-correct

---

## Decision 6: Preview Report Format and Interaction

**Decision**: Hybrid approach: Primary markdown report with optional interactive CLI prompts

**Markdown Preview Report**:
- Structured tables showing bottle groups, images, extracted metadata
- Clear column headers: Bottle Name, Quantity, Images, Metadata Summary
- Tables allow easy diffing between report versions
- Version control friendly for tracking changes over time
- Non-blocking:-processes continues with or without user review

**Optional CLI Interactive Mode**:
- Enable with `--interactive` flag
- Prompt after each bottle grouping decision
- Accept/Reject/Modify options per grouping
- Faster workflow for users comfortable with CLI
- Gracefully degrades to markdown-only if user declines prompts

**Report Structure**:
```markdown
# Batch {N} Preview Report

## Grouping Summary
| Bottle Name | Quantity | Images | Confidence |
|-------------|----------|--------|------------|
| Buffalo Trace | 2 | IMG_001.jpeg, IMG_002.jpeg | High |
| Blanton's | 1 | IMG_003.jpeg | Medium |

## Extracted Metadata
[Detailed per-bottle metadata in tables]

## Rejected Items
[Items flagged for manual review]

## Instructions
[How to provide feedback]
```

**Alternatives Considered**:
- Markdown only: Limit workflow efficiency, no real-time interaction
- CLI prompts only: Harder to reference previous reports, lack version control history
- Web UI preview: Overkill for CLI tool, adds unnecessary complexity

**Impact**: Defines user workflow, testing strategy (both modes must be tested), documentation requirements

---

## Decision 7: Vendor Image Caching and Reference Link Collection Strategy

**Decision**: Optional, non-blocking feature using web search + image similarity verification + link collection

**Design**:

1. **Metadata-Based Search**:
   - Use extracted bottle name, distillery, proof, batch as search query
   - Query search via web search API (e.g., DuckDuckGo or direct site scraping)
   - Look for two key artifacts:
     - Official product images on distillery sites, retail listings
     - Vendor product pages specifically about this bottle/release

2. **Similarity Verification** (for images):
   - Compare candidate vendor image to original bottle images using feature matching
   - Confirm match above confidence threshold (e.g., 85%+ similarity)

3. **Link Collection and Verification** (for reference materials):
   - Extract URLs pointing to vendor pages with product-specific information
   - Prioritize: distillery official site → authorized retailers → reputable bourbon databases
   - Validate links (check HTTP status, avoid broken URLs)
   - Store links with metadata: source, confidence score, description (e.g., "Distillery product page", "Retailer listing")

4. **Caching Behavior** (images):
   - If matching vendor image found: Download and save as `vendor_ref.jpg` in bottle sub-folder
   - Mark in metadata as "vendor_reference": true
   - If no match or search fails: Continue processing normally, log as "no vendor image found"
   - Feature does NOT block processing: Bottle files generated regardless

5. **Markdown Output** (reference links):
   - Add "References" section to generated markdown file
   - Format:
     ```markdown
     ## References
     - [Distillery Product Page](URL) - Official distillery information about this release
     - [Retailer Listing](URL) - Product details and availability
     ```
   - Include all high-confidence reference links found
   - If no links found: omit section or note "No reference material located"

6. **Error Handling**:
   - Network failures: Skip vendor image/link lookup, continue processing
   - No matches: Document in markdown (e.g., "Vendor image: Not available", "References: None found")
   - Multiple ambiguous matches: Do not cache images, but may include links with low-confidence flags for manual review

**Alternatives Considered**:
- Require vendor image always: Too restrictive, processing fails if web unavailable
- Skip entirely (no vendor images): Less valuable for users who want reference photos
- Use paid API (Google Images, Bing Vision): Adds cost, not aligned with local-first approach

**Impact**: Defines optional feature implementation, error handling, testing rails

---

## Decision 8: Error Handling and Retry Mechanism

**Decision**: Three-tier strategy (automatic retry, error reports, hints.md)

**Design**:

1. **Automatic Retry** (up to 3 attempts):
   - Per-bottle timeout: 60 seconds max
   - On failure (OCR timeout, model error, network issue): Retry immediately
   - Backoff strategy: Exponential backoff (0s → 2s → 5s delays)
   - Different failure types handled differently:
     - Ollama API error: Retry with new connection
     - Timeout: Retry with smaller image (downsampled)
     - OCR failure: Retry with different orientation

2. **Error Report Generation** (if all retries exhausted):
   - Create `error_report.json` in bottle sub-folder
   - Include: error type, timestamp, failure details, retry attempts, suggested fixes
   - Retry-friendly format: User can re-run processor starting from failed items

3. **Hints.md Maintenance**:
   - Global `hints.md` file documenting hard-to-handle bottles
   - Per-bottle guidance: specific challenges, manual intervention notes
   - Examples: "Small print on label requires zoom", "Multiple angles confuse OCR"
   - Format:
     ```markdown
     # Processing Hints

     ## Blanton's
     - Challenge: Small batch number on back label
     - Suggestion: Use close-up of back label, or manually add batch number
     ```

4. **Retry Mechanism**:
   - Command-line flag: `--retry-from-report error_report.json`
   - Reads error report, applies suggested fixes, re-processes
   - User can update hints.md to guide retry behavior

**Alternatives Considered**:
- Manual user intervention only: Too manual, poor UX
- Crash on first failure: Unacceptable for batch processing
- Infinite retry loop: Resource waste, no progress mechanism

**Impact**: Defines reliability, user experience for handling edge cases

---

## Decision 9: Testing Approach

**Decision**: Three-tier testing strategy with pytest

**Strategy**:

1. **Unit Tests** (`tests/unit/`):
   - Test individual modules in isolation
   - Mock Ollama API calls
   - Use small synthetic images for orientation detection tests
   - Test quantity detection logic with pre-defined scenarios
   - Test batch lookahead logic with mock image queues

2. **Contract Tests** (`tests/contract/`):
   - Test CLI interface contracts (argument parsing, output formats)
   - Verify JSON schemas for error reports
   - Validate markdown structure of preview reports and bottle metadata files
   - Test configurable batch size behavior

3. **Integration Tests** (`tests/integration/`):
   - End-to-end tests with real images from test fixtures directory
   - Validate full pipeline from images to markdown output
   - Test Ollama model integration (requires Ollama service running)
   - Multi-bottle workflow tests with lookahead

**Test Organization**:
```text
tests/
├── fixtures/
│   ├── images/              # Sample bottle images for integration tests
│   ├── straightforward/     # Clearly labeled bottles
│   ├── rotated/            # Images with orientation issues
│   └── ambiguous/          # Edge cases
├── unit/
│   ├── test_bottle_analyzer.py
│   ├── test_image_organizer.py
│   ├── test_preview_generator.py
│   └── test_batch_processor.py
├── contract/
│   ├── test_cli_contracts.py
│   └── test_output_formats.py
└── integration/
    ├── test_full_pipeline.py
    └── test_batch_workflow.py
```

**Alternatives Considered**:
- Integration tests only: Too slow, no confidence in refactoring
- Unit tests only: Misses integration issues
- Manual testing only: Unacceptable for quality and sustainability

**Impact**: Defines code quality, confidence in deployments, CI/CD requirements

---

## Decision 10: Performance Optimization Strategy

**Decision**: Multi-pronged approach: concurrency, caching, image optimization

**Strategy**:

1. **Concurrent API Calls**:
   - Process multiple bottles in parallel using `asyncio` + `aiohttp`
   - Limit concurrency to avoid Ollama overload (default: 3 concurrent requests)
   - Batch API calls where possible (e.g., multiple images in one request if model supports)

2. **Image Optimization**:
   - Downsample images larger than 2048px on longest dimension for vision model
   - Preserve original images in output folder (no loss of detail)
   - Cache resized versions in memory during processing

3. **Model Keep-Alive**:
   - Configure Ollama to keep models loaded between requests (`OLLAMA_KEEP_ALIVE=-1`)
   - Avoid cold-start penalties for repeated model loading

4. **Incremental Processing**:
   - Process bottles incrementally (batch-by-batch)
   - Clear memory between batches to avoid accumulation
   - Reuse open HTTP connections to Ollama

5. **Target Metrics**:
   - 60 seconds per bottle set (including OCR + vision analysis + metadata generation)
   - 10 bottles in under 10 minutes
   - 95%+ accuracy on clearly labeled bottles

**Performance Bottlnecks to Watch**:
- Ollama model loading time (cold start: 10-30s, warm start: <2s per request)
- Image orientation detection (OpenCV feature extraction can be slow for large images)
- OCR time (Tesseract preprocessing + recognition)
- Large image file I/O

**Alternatives Considered**:
- GPU acceleration: Not available in all environments, adds deployment complexity
- Smaller models only: Loses accuracy, fails 95% accuracy requirement
- No concurrency: Too slow for 10+ bottle collections

**Impact**: Defines user experience, hardware requirements, meeting performance targets

---

## Decision 11: Project Structure and Module Organization

**Decision**: Domain-driven separation with shared utilities layer

**Organizational Principles**:
- Modules organized by business domain (bottle analysis, image organizing, preview generation)
- Shared utilities for cross-cutting concerns (Ollama client, configuration, error handling)
- Clear separation of concerns: CLI interface independent of business logic
- Existing code refactored gradually to fit new structure

**Final Structure**:
```text
src/
├── image_processor/          # Domain modules
│   ├── bottle_analyzer.py    # Bottle metadata extraction
│   ├── image_organizer.py   # Image grouping, quantity detection
│   ├── preview_generator.py # Markdown preview reports
│   ├── vendor_cache.py      # Optional vendor image caching
│   ├── text_extractor.py     # OCR, orientation correction
│   └── batch_processor.py    # Orchestration, lookahead, feedback loop
├── models/                   # Data model classes
│   └── bottle.py
├── cli/                      # CLI interface
│   └── commands.py
└── lib/                      # Shared utilities
    ├── ollama_client.py
    ├── config.py
    └── error_handling.py
```

**Alternatives Considered**:
- Monolithic script: Hard to test, difficult to reason about
- Microservices-style: Overkill for local CLI tool, deployment complexity
- Layered architecture (presentation/business/data): Clearer boundaries for web apps, overkill for CLI

**Impact**: Defines code organization, testability, maintainability

---

## Decision 12: Configuration Management

**Decision**: CLI arguments + environment variables + optional config file

**Configuration Sources** (priority order):
1. CLI arguments (highest priority, override everything)
2. `.env` or environment variables
3. Config file `(specify_config.toml)` (optional, lowest priority)
4. Default hardcoded values (fallback)

**Configuration Parameters**:
- `--image-folder PATH`: Folder containing bottle images (default: current directory)
- `--batch-size N | all`: Process N bottles or all images (default: 5)
- `--model NAME`: Ollama model for analysis (default: qwen3-vl:32b)
- `--interactive`: Enable CLI prompts (default: disabled)
- `--include-vendor-images`: Attempt to cache vendor images (default: disabled)
- `--ollama-url URL`: Custom Ollama endpoint (default: http://localhost:11434)
- `--output-dir PATH`: Output folder for processed files (default: processed/)
- `--retry-after-fail`: Enable automatic retry (default: enabled)

**Alternatives Considered**:
- CLI only: Simple but no persistent configuration
- Config file only: Hard to override per-invocation behavior
- Hardcoded defaults only: Inflexible, poor UX

**Impact**: Defines user experience, testing configuration, deployment flexibility

---

## Decision 13: CLI Framework

**Decision**: argparse (Python standard library) with sub-commands

**Rationale**:
- Built-in to Python, no additional dependency
- Sufficient for CLI tool complexity (no complex interactive menus)
- Easy to generate help text automatically
- Good argument validation support

**Command Structure**:
```bash
python -m cli.commands process [options]
python -m cli.commands retry --report error_report.json [options]
```

**Alternatives Considered**:
- click: More feature-rich (populates env variables, nicer help text), but adds dependency
- typer: Modern, typed CLI builder, but learning curve for team
- bare manual sys.argv parsing: Error-prone, no built-in help

**Impact**: Defines CLI UX, code style, dependency footprint

---

## Decision 14: Output File Formats

**Decision**: Markdown for metadata (readable, version controllable), JSON for errors/debugging

**Markdown Format**:
- Human-readable for direct website publication
- Version control friendly (diff, merge)
- Easy to format for Hugo/Jekyll/etc.
- Format:
  ```markdown
  # {Bottle Name}

  ## Distillery
  {Distillery Name}

  ## MSRP
  ${MSRP}

  ## Release Information
  - Proof: {X} proof
  - Age: {X} years
  - Batch: {Batch Number}

  ## Tasting Notes
  {Tasting notes...}
  ```

**JSON Format** (error reports, debug outputs):
- Machine-readable for retry mechanisms
- Easy to parse programmatically
- Schema: JSON Schema for validation

**Alternatives Considered**:
- HTML only: Less suitable for CLI tool, requires web browser
- JSON for everything: Not human-readable for website publication
- CSV: Too flat for hierarchical metadata

**Impact**: Defines file output strategy, data interchange formats, user experience

---

## Decision 15: Logging and Observability

**Decision**: Python logging module with configurable levels

**Strategy**:
- Default level: INFO (progress updates, batch summaries)
- Debug level: (--debug flag) detailed logs (API responses, OCR raw output)
- Error level: Critical failures
- Log format: Timestamp + level + message + context (e.g., image filename)
- Logs written to console (stdout/stderr) and optionally to file

**Logging Scenarios**:
- Processing start: "Processing 25 images in {folder}"
- Per-bottle progress: "Processing bottle set {N}/M: {images}"
- API call: "Calling Ollama with model {model}"
- Retry attempt: "Retry {N}/3: {error_type}"
- Success: "Successfully processed bottle set: {bottle_name} → {output_path}"

**Alternatives Considered**:
- print() statements only: No level filtering, no structured logging
- Structlog/JSON logging: Overkill for simple CLI tool
- No logging: Unacceptable for debugging, user transparency

**Impact**: Defines debuggability, user experience, troubleshooting

---

## Decision 16: Documentation Strategy

**Decision**: README.md + docstrings + inline comments

**Approach**:
1. **README.md**: Installation, usage, configuration, troubleshooting (already exists)
2. **API docstrings**: Google-style docstrings for all public functions
3. **Inline comments**: Explain non-obvious logic (orientation detection, quantity heuristics)
4. **Feature specs**: `specs/` directory for specification artifacts (this document, data-model, quickstart)
5. **Example outputs**: Sample processed folders in `examples/` directory

**Alternatives Considered**:
- Sphinx documentation: Overkill for CLI tool, adds build complexity
- No documentation: Unacceptable for maintainability, new developers

**Impact**: Defines maintainability, onboarding, knowledge transfer

---

## Open Questions for Planning Phase

None. All technical decisions resolved.

---

## Summary

- **Language**: Python 3.11+
- **Models**: Ollama (qwen3-vl:32b, gpt-oss:120b), Tesseract-OCR
- **Framework**: argparse CLI, asyncio for concurrency
- **Image Processing**: OpenCV (orientation), Pillow/PIL (manipulation), pytesseract (OCR)
- **Architecture**: Domain-driven modules with shared utilities
- **Testing**: pytest with unit/contract/integration tiers
- **Configuration**: CLI args + environment variables
- **Output**: Markdown metadata, JSON error reports
- **Scope**: Local-only, batch processing (default 5 bottles), optional vendor image caching (non-blocking)

Next Phase: Generate data-model.md, contracts/, and quickstart.md (Phase 1).
