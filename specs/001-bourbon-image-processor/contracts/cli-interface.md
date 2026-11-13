# CLI Interface Contract: Bourbon Image Processor

**Version**: 1.0
**Date**: 2026-11-09
**Status**: Complete

## Overview

This contract defines the command-line interface for the bourbon image processor. The CLI supports processing images, batch operations, retries, and configuration.

---

## Command: `process`

Primary command for processing bourbon bottle images from a folder.

### Synopsis

```bash
python -m cli.commands process [options]
```

### Required Arguments

None (all arguments optional with defaults)

### Options

| Short | Long | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| -f | --image-folder PATH | Folder containing bottle images to process | Current directory (`.`) | Must exist, be readable directory |
| -b | --batch-size N\|all | Number of bottles per batch, or "all" for full-folder processing | 5 | Integer ≥ 1, or literal "all" |
| -m | --model NAME | Ollama model for image analysis | qwen3-vl:32b | Must be valid Ollama model name |
| -i | --interactive | Enable interactive CLI prompts for preview review | disabled (false) | Boolean flag |
| -v | --include-vendor-images | Attempt to cache vendor reference images and collect reference links | disabled (false) | Boolean flag |
| -u | --ollama-url URL | Custom Ollama API endpoint | http://localhost:11434 | Valid URL, reachable |
| -o | --output-dir PATH | Output folder for processed files | processed/ | Must be writable path |
| -r | --allow-retry | Enable automatic retry on processing failures (max 3 attempts) | enabled (true) | Boolean flag |
| --lookahead N | Number of images to look ahead for multi-angle photos when batching | 3 | Integer 0-10 |
| --concurrency N | Maximum concurrent API requests to Ollama | 3 | Integer 1-10 |
| --debug | Enable debug logging (verbose output) | disabled (false) | Boolean flag |
| --dry-run | Show what would be processed without actually processing | disabled (false) | Boolean flag |
| -h | --help | Show help message and exit | N/A | N/A |

### Behavior

1. **Initialization**:
   - Validate input folder exists and contains images
   - Check Ollama service availability at configured URL
   - Verify specified model is available
   - Create output directory if not exists

2. **Image Discovery**:
   - Scan input folder recursively for supported image formats (.jpeg, .jpg, .png, .tiff, .heic)
   - Sort images alphabetically by filename
   - Group images using lookahead to detect multi-angle photo sequences

3. **Batch Processing**:
   - Process images in batches (size determined by `--batch-size`)
   - Apply lookahead (default 3 images) to include related multi-angle photos
   - For each image:
     - Correct orientation if needed
     - Extract text via OCR and vision model
     - Generate bottle metadata
     - Group related images into bottle sets
     - Detect quantity (duplicates, multiple bottles in frame)

4. **Preview Generation** (unless `--batch-size all`):
   - After each batch, generate markdown preview report
   - Show grouping decisions and extracted metadata
   - If `--interactive`: prompt user to accept/reject/modify per grouping
   - Apply corrections before writing output files

5. **Output Generation**:
   - Create sub-folder per bottle in output directory
   - Save original images, markdown metadata, extracted text
   - Optionally cache vendor reference image and collect reference links (if `--include-vendor-images`)
   - Generate error reports for failed attempts

6. **Retry Logic** (if `--allow-retry` enabled):
   - Automatic retry on failures (up to 3 attempts)
   - Backoff: 0s → 2s → 5s delays between retries
   - After all retries exhausted, generate error report

7. **Completion Reporting**:
   - Summary of processed/failed bottles
   - Paths to output files
   - Any warnings or errors encountered

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (all bottles processed or failed gracefully) |
| 1 | General error (invalid arguments, missing dependencies) |
| 2 | Ollama connection failed |
| 3 | No valid images found in input folder |
| 4 | Processing failed for all bottles |
| 5 | Output directory creation failed |

### Example Usage

```bash
# Basic processing (current directory, batch size 5)
python -m cli.commands process

# Process specific folder
python -m cli.commands process --image-folder ./bourbon_photos

# Batch processing with interactive feedback
python -m cli.commands process -f ./photos -b 5 -i

# Full-folder processing without preview reports
python -m cli.commands process -f ./photos --batch-size all --include-vendor-images

# Faster processing with lighter model
python -m cli.commands process -f ./photos -m qwen2-vl:7b --concurrency 5

# Debug mode to troubleshoot issues
python -m cli.commands process -f ./photos --dry-run --debug
```

---

## Command: `retry-from-report`

Re-process bottles that failed in a previous attempt, using error report guidance.

### Synopsis

```bash
python -m cli.commands retry-from-report --report PATH [options]
```

### Required Options

| Short | Long | Description | Validation |
|-------|------|-------------|------------|
| --report PATH | Path to error report JSON file | Must exist, valid JSON |

### Optional Options

All options from `process` command are supported. Options specified here override defaults but do not override values from error report.

### Behavior

1. **Load Error Report**:
   - Read error_report.json
   - Extract failed bottle ID and images
   - Extract error type, suggested fix, hint topic

2. **Apply Corrections**:
   - Search hints.md for hint topic (if present in error report)
   - Apply suggested fix (if implementable programmatically)
   - Allow user to specify corrections via prompts or flags

3. **Re-process**:
   - Re-run processing for failed bottle only
   - Use same options as original attempt (unless overridden)
   - Skip already-successful bottles

4. **Generate Output**:
   - Create/update bottle folder with new metadata
   - Mark as re-processed in status tracking
   - Update error report or delete if successful

### Exit Codes

Same as `process` command, plus:

| Code | Meaning |
|------|---------|
| 10 | Error report file not found or invalid |
| 11 | Bottle re-processing failed again after retry |

### Example Usage

```bash
# Retry from an error report
python -m cli.commands retry-from-report --report processed/error_report.json

# Retry with different model
python -m cli.commands retry-from-report --report processed/error_report.json -m qwen3-vl:32b
```

---

## Command: `feedback`

Provide user feedback on a generated preview report (alternative to interactive prompts).

### Synopsis

```bash
python -m cli.commands feedback --batch BATCH_ID [options]
```

### Required Options

| Short | Long | Description | Validation |
|-------|------|-------------|------------|
| --batch ID | Batch ID from preview report | Must exist, valid batch identifier |

### Optional Options

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| --corrections FILE | JSON file with corrections (alternative to interactive prompts) | None |
| -i | --interactive | Interactive prompt for corrections per grouping | disabled (false) |

### Behavior

1. **Load Batch**:
   - Load batch metadata using batch ID
   - Load preview report
   - Display current grouping decisions and metadata

2. **Collect Feedback**:
   - If interactive: prompt user for each grouping (accept/reject/modify)
   - If corrections file: read JSON with structured corrections
   - Accept corrections structured as:
     ```json
     {
       "bottle_id": "bottle_abc123",
       "correction_type": "MERGE_BOTTLES",
       "target_bottle_id": "bottle_def456",
       "reason": "Same batch number and proof"
     }
     ```

3. **Apply Corrections**:
   - Update bottle groupings
   - Merge/split bottles as requested
   - Update metadata fields if specified

4. **Regenerate Output**:
   - Create final output files with corrections applied
   - Mark batch as completed with feedback

### Exit Codes

Same as `process` command, plus:

| Code | Meaning |
|------|---------|
| 20 | Batch ID not found |
| 21 | Corrections JSON invalid |

### Example Usage

```bash
# Interactive feedback
python -m cli.commands feedback --batch batch_001 -i

# Feedback from corrections file
python -m cli.commands feedback --batch batch_001 --corrections my_feedback.json
```

---

## Environment Variables

Configurable via environment variables (lower priority than CLI arguments):

| Variable | Description | Default |
|----------|-------------|---------|
| `BORBON_IMAGE_MODEL` | Ollama model for image analysis | qwen3-vl:32b |
| `OLLLAMA_URL` | Ollama API endpoint | http://localhost:11434 |
| `BORBON_OUTPUT_DIR` | Output folder for processed files | processed/ |
| `BORBON_CONCURRENCY` | Max concurrent API requests | 3 |
| `BORBON_DEBUG` | Enable debug logging | false |
| `OLLLLAMA_MAX_LOADED_MODELS` | Max models kept in memory by Ollama | 3 |
| `OLLLLAMA_KEEP_ALIVE` | Keep-alive duration for models (seconds, -1 = indefinite) | -1 |

### Precedence Order

1. CLI arguments (highest)
2. Environment variables
3. Default values (lowest)

---

## Output Conventions

### Progress Messages

All progress messages written to stdout:
- Processing start: `Processing {N} images from {folder}`
- Bottle progress: `Processing bottle {current}/{total}: {images}`
- Batch progress: `Batch {current}/{total}: {N} bottles`
- Success: `✓ Successfully processed: {bottle_name} → {output_path}`
- Failure: `✗ Failed to process: {images} → {error_type}`

### Debug Messages (when `--debug` enabled)

Written to stdout with more detail:
- Ollama API calls: `POST {ollama_url}/api/generate with model {model}`
- Orientation detection: `Detected orientation: {detected} → correcting...`
- OCR output: `Extracted text: {raw_text}`
- Config values: `Using config: {key}={value}`

### Error Messages

Written to stderr:
- Invalid arguments: `Error: --batch-size must be positive integer or "all"`
- Missing dependencies: `Error: Required dependency not installed: {package}`
- Ollama error: `Error: Could not connect to Ollama at {url} - ensure service is running`

### Log Format

```
{timestamp} {level} {module}: {message}
```

Example:
```
2026-11-09 14:30:22 INFO image_processor.bottle_analyzer: Processing bottle 1/15
2026-11-09 14:30:45 DEBUG ollama_client: API call completed in 23.4s
```

---

## Interaction Modes

### Interactive Mode (`--interactive`)

Prompts user for each grouping decision:

```
Processing Batch 1/3

Group 1: Buffalo Trace (2 images: IMG_001.jpg, IMG_002.jpg)
Detected: 1 bottle, Quantity: 2, Confidence: High
Metadata: Buffalo Trace Distillery, 90 proof, $29.99, Batch BT24-07

Action? [A]ccept / [R]eject / [M]erge / [S]plit / [E]dit: _
```

**Actions**:
- `A` or `accept`: Accept grouping as-is
- `R` or `reject`: Reject grouping (mark for manual review, skip this set)
- `M` or `merge`: Merge with previous grouping (if applicable)
- `S` or `split`: Split into multiple bottles (specify split points)
- `E` or `edit`: Edit metadata interactively

### Non-Interactive Mode

Skip prompts, generate markdown preview report only:

```
Generating preview report: processed/batch_1_preview.md
Run `python -m cli.commands feedback --batch batch_001 -i` to review
```

---

## Configuration File Format

Optional config file at `.specify_config.toml` in root directory:

```toml
[processing]
default_batch_size = 5
lookahead_images = 3
concurrency = 3

[ollama]
url = "http://localhost:11434"
model = "qwen3-vl:32b"
max_loaded_models = 3
keep_alive = -1

[output]
directory = "processed"
include_timestamps = true

[vendor]
cache_images = false
collect_reference_links = false

[retry]
max_attempts = 3
enabled = true
```

---

## Validation Rules

### Option Validation

- `--image-folder`: Must exist, be directory, be readable
- `--batch-size`: Integer ≥ 1 OR literal "all"
- `--model`: Must be included in list from `ollama list`
- `--ollama-url`: Valid URL, must respond to GET `/api/tags`
- `--output-dir`: Must be writable (or creatable)
- `--lookahead`: Integer 0-10
- `--concurrency`: Integer 1-10

### Pre-flight Checks

Before processing starts:
1. Verify input folder contains at least 1 supported image
2. Verify Ollama service is reachable at configured URL
3. Verify specified model is available in Ollama
4. Verify Tesseract OCR is installed and executable
5. Verify output folder is writable (can create if not exists)

### Post-flight Checks

After processing completes:
1. Verify all bottles written to output folder
2. Verify error reports generated for failures
3. Verify preview reports generated for batches (unless `--batch-size all`)
4. Calculate and report success/failure statistics

---

## Error Handling

### Argument Errors

Exceptions during argument validation:
```bash
Error: Invalid value for --batch-size: '-1' (must be >= 1 or 'all')
Usage: python -m cli.commands process [options]
  --batch-size N  Number of bottles per batch, or 'all' for full-folder processing
```

### Dependency Errors

Missing required packages:
```bash
Error: Required dependency not installed: requests
Install with: pip install requests
```

### Ollama Connection Errors

Cannot reach Ollama:
```bash
Error: Could not connect to Ollama at http://localhost:11434
Start Ollama with: ollama serve
Check status with: curl http://localhost:11434/api/tags
```

### Processing Errors (Non-Fatal)

Individual bottle failures do not stop processing:
```
Warning: Failed to process bottle set {images} after 3 retries
  Error type: OCR_FAILURE
  Error: Tesseract unable to extract text
  Suggested fix: Check image quality, re-capture
  Error report: processed/{bottle_folder}/error_report.json
  Hints topic: rotated_images (see processed/hints.md)
Continuing with remaining bottles...
```

### Processing Errors (Fatal)

Errors that stop all processing:
- Output directory not writable
- Ollama service unavailable
- No valid images found
- Model not found

---

## Version Information

### Display Version

```bash
python -m cli.commands --version
```

Output:
```
Bourbon Image Processor v1.0.0
Python 3.11.0
Ollama client library built-in
```

---

## Testing the CLI

### Unit Tests (/tests/contract/test_cli_contracts.py)

Test argument parsing, validation, and help generation.

### Integration Tests (/tests/integration/test_cli_workflow.py)

Test end-to-end CLI commands with real fixtures.

### Contract Tests to Verify

- [ ] `--help` generates valid help text
- [ ] Invalid arguments exit with code 1 and show error message
- [ ] Required arguments reject missing values
- [ ] Pre-flight checks fail before processing
- [ ] Exit codes match specification
- [ ] Non-fatal errors allow continuation
- [ ] Fatal errors stop processing
- [ ] Interactive prompts work when `--interactive` enabled
- [ ] Markdown preview reports generated without `--batch-size all`
- [ ] Vendor image caching works when `--include-vendor-images` enabled

---

## Summary

The CLI provides:
- Single-command processing workflow (`process`)
- Retry mechanism for failed attempts (`retry-from-report`)
- Feedback loop for corrections (`feedback`)
- Batch processing with configurable sizing
- Interactive and non-interactive modes
- Comprehensive error handling and reporting

All commands are designed to fail explicitly with helpful error messages, use sensible defaults, and support configuration via CLI arguments, environment variables, and optional config files.
