# CLI Interface Contract: Bourbon Image Processor

> **Superseded in part (2026-11-23).** This document describes the original
> design. The code has since changed; [README.md](../../../README.md) describes what it
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

**Version**: 2.0 (updated 2026-11-23; version 1.0 was 2026-11-09)
**Status**: Matches the code as of 2026-11-23

## Overview

This contract defines the command-line interface of `bourbon-process`. The
console script is `bourbon-process = cli.commands:main`; `python -m
cli.commands` (with `src/` on the path) is equivalent. Subcommands:
`process`, `feedback`, `retry`, `config`. An argument list that names no
subcommand runs `process`; no arguments at all prints the help and exits 1.

Global options (before or after the subcommand): `--debug`, `--config PATH`,
`--no-config`, `-h/--help`. There is no `--version` option; the version is
recorded in each run's `run_<run_id>_settings.json`.

---

## Command: `process`

Primary command for processing bourbon bottle images from a folder.

### Synopsis

```bash
bourbon-process process [options]
```

### Required Arguments

None (all arguments optional with defaults)

### Options

Every option except `--dry-run`, `--refresh-web` (CLI only) and the global
ones also has a config key and, as listed in README.md, a `BOURBON_*`
variable. Argparse defaults are empty; values are validated by the settings
layer (invalid values exit 1, naming their source).

| Short | Long | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| -f | --image-folder PATH | Folder containing bottle images (scanned recursively) | `.` | Must exist, contain a supported image |
| -o | --output-dir PATH | Output folder | `processed` | Writable; not the image folder |
| -b | --batch-size N\|all | Bottles per batch | 5 | Integer >= 1, or "all" |
| | --concurrency N | Bottle groups analysed in parallel; output identical to sequential | 2 | 1-10 |
| | --group-gap-seconds S | Most seconds between photos of one bottle (EXIF capture time) | 12 | > 0, <= 600 |
| | --group-max-size N | Most photos per bottle (0 = no limit) | 3 | 0-20 |
| | --split-budget-ratio R | Extra split-check analyses per group over the run (retries not counted) | 0.35 | 0-2 |
| -m | --model TAG | Ollama vision model | qwen3.6:35b-a3b | Non-empty; vision-capable at pre-flight |
| -u | --ollama-url URL | Ollama endpoint | http://localhost:11434 | http(s) URL with a valid port |
| | --server-start-timeout S | Wait for a server this run starts | 30 | > 0 |
| | --request-timeout S | Per-request vision timeout | 600 | > 0 |
| | --keep-server / --no-keep-server | Leave a started server running | off | Boolean |
| | --server-autostart / --no-server-autostart | Start a local Ollama if none runs | on | Boolean |
| | --keep-models-loaded / --no-keep-models-loaded | Skip unloading models between the vision and text phases | off | Boolean |
| | --pull / --no-pull | Download missing models after a disk check | off | Boolean |
| | --min-free-gb GB | GiB to keep free after a pull | 5 | >= 0 |
| -i | --interactive / --no-interactive | Review each bottle before saving | off | Boolean |
| | --hints PATH | Hints TOML (notes and sourced field values) | none | Valid hints file |
| -r | --allow-retry | Repeat model requests that may pass | on | Boolean |
| | --no-retry | Make every model request once | | Boolean |
| | --retry-attempts N | Attempts per model request | 3 | 1-5 |
| | --verify / --no-verify | gpt-oss fact check | on | Boolean |
| | --text-model TAG | Fact-check model, or "none" | gpt-oss:120b | Non-empty; completion-capable |
| | --text-think LEVEL | Extraction reasoning effort | high | low, medium, high |
| | --text-timeout S | Read timeout of one text-model call | 900 | > 0 |
| | --web / --no-web | Web lookups | on | Boolean |
| | --web-provider NAME | Search provider | auto | auto, duckduckgo, brave, ollama, searxng, none; key or URL present |
| | --web-cache-dir PATH | Web cache | ${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web | Path |
| | --refresh-web | Ignore cached pages and searches for this run | off | Boolean |
| -v | --include-vendor-images / --no-include-vendor-images | Vendor reference image per bottle | off | Needs web lookups |
| | --dry-run | Check everything; start, pull, load and write nothing | off | Boolean |
| | --debug | Debug logging and the model stage report | off | Boolean |
| -h | --help | Show help and exit | | |

`--lookahead`, `--server-backend`, `--vision-timeout` and `--pull-missing`
were removed and are rejected by argparse (exit 2).

### Behavior

1. **Initialization (pre-flight)**: resolve settings (defaults, config files,
   environment, flags); check the image folder, the output folder, the hints
   file and Tesseract; reuse or start Ollama; check that the vision model and,
   when the fact check is on, the text model are installed (or pull them with
   `--pull`) and have the needed capability; check the text model against
   physical memory; warm the vision model.
2. **Image Discovery**: scan the folder recursively for `.bmp .heic .jpeg .jpg
   .png .tiff .webp`, skipping hidden folders and earlier outputs; group photos
   into bottles by EXIF capture time.
3. **Batch Processing**: per batch, OCR and one vision request per bottle (with
   the same-bottle split check), quantity from duplicate photos, hint values,
   then web research, vision-model unload, the fact check of each bottle and
   the text-model unload.
4. **Review**: with `--interactive`, approve, reject, edit or settle conflicts
   for each bottle.
5. **Preview Generation**: a Markdown preview report and a batch JSON for every
   batch.
6. **Output Generation**: one new folder per bottle (Markdown, JSON
   `schema_version` 2, OCR text, `evidence/W<n>.txt`, images, optional
   `images/vendor_ref.<ext>`); an `error_report_<bottle_id>.json` for each
   failed or rejected bottle; `run_<run_id>_settings.json` for the run.
7. **Retry Logic**: with retry on, each model request is tried up to
   `--retry-attempts` times (pause 2 s, doubling, at most 30 s); a timeout is
   retried once; the fact check makes at most 2 attempts per call and does
   not retry a text-model timeout.
8. **Completion Reporting**: processed and failed counts, the web provider,
   web and fact-check status counts, publish-ready count, output paths and the
   model usage summary.

Web lookups and the fact check never fail a bottle; their failures are
recorded in the bottle's `web_research` and `facts.verification`.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (also a successful `--dry-run`) |
| 1 | Invalid setting, invalid hints file, output folder equal to the image folder, Tesseract missing, some bottles failed, or no arguments |
| 2 | Model server or model unavailable, or text model larger than memory; also argparse usage errors |
| 3 | No valid images found in input folder |
| 4 | Processing failed for all bottles |
| 5 | Output directory not writable |
| 128+N | Stopped by signal N |

### Example Usage

```bash
# Basic processing (current directory, batch size 5)
bourbon-process process

# Process specific folder
bourbon-process process --image-folder ./bourbon_photos

# Batch processing with interactive review
bourbon-process process -f ./photos -b 5 -i

# One batch, vendor reference images
bourbon-process process -f ./photos --batch-size all --include-vendor-images

# Smaller fact-check model, no web
bourbon-process process -f ./photos --text-model gpt-oss:20b --no-web

# Check everything without processing
bourbon-process process -f ./photos --dry-run --debug
```

---

## Command: `retry`

Re-process one bottle that failed in a previous run, from its error report.
(Version 1.0 of this contract called it `retry-from-report`; that command
never existed.)

### Synopsis

```bash
bourbon-process retry --report PATH [options]
```

### Required Options

| Short | Long | Description | Validation |
|-------|------|-------------|------------|
| | --report PATH | Path to `error_report_<bottle_id>.json` | Must exist; JSON with `schema_version` 2, `bottle_id`, `images`, `error_message` |

### Optional Options

`-o/--output-dir`, `--hints`, and the model, server, retry, fact-check and web
options of `process` (`-m`, `-u`, `--server-start-timeout`,
`--request-timeout`, `--keep-server`, `--server-autostart`,
`--keep-models-loaded`, `-r`, `--no-retry`, `--retry-attempts`, `--verify`,
`--text-model`, `--text-think`, `--text-timeout`, `--web`, `--web-provider`,
`--web-cache-dir`, `--refresh-web`, `-v`). Not `--pull`.

### Behavior

1. **Load Error Report**: read the report; refuse a report that is not
   `schema_version` 2 or whose `suggested_fix` names a field that cannot be
   corrected.
2. **Re-process**: OCR and the vision request for the report's images, with
   the hint note; then the hint values; then the report's `suggested_fix`
   values as corrections for this bottle; then web research and the fact
   check.
3. **Generate Output**: a new bottle folder (existing output is never
   replaced). The error report is left in place.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error: invalid report or settings, model server unavailable, re-processing failed |
| 2 | Error report not found |

### Example Usage

```bash
bourbon-process retry --report processed/error_report_IMG_1234.json
bourbon-process retry --report processed/error_report_IMG_1234.json -m gemma4:26b
```

---

## Command: `feedback`

Apply corrections and approvals to a saved batch (alternative to interactive
review during `process`). Calls no model and no web site.

### Synopsis

```bash
bourbon-process feedback [--batch BATCH_ID] [options]
```

### Options

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| -o | --output-dir PATH | Folder with the batch files | processed |
| | --batch ID | Batch ID, e.g. `20261122_193000_001` for `batch_20261122_193000_001_preview.md` | Most recent batch |
| | --corrections FILE | JSON corrections file | None |
| -i | --interactive / --no-interactive | Interactive review of the batch's bottles | off |
| | --approve BOTTLE_ID | Approve a bottle for publishing (repeatable) | None |

### Behavior

1. **Load Batch**: read `batch_<id>.json`; refuse one that is not
   `schema_version` 2.
2. **Collect Feedback**: from the corrections file, or interactively
   (`[A]ccept`, `[R]eject`, `[E]dit`, `[C]onflicts`, `[N]ext`, `[Q]uit`), plus
   `--approve`. The corrections file is a JSON list:
   ```json
   [
     {"bottle_id": "IMG_4643", "field_name": "batch_number", "original_value": "C92", "corrected_value": "C923"}
   ]
   ```
   A correction applies only to that bottle, and only if its current value
   equals `original_value`.
3. **Apply**: corrections become user facts; the facts are re-reconciled
   without a model call; approvals are recorded and the publish gate is
   recomputed. Grouping is never changed (no merge or split).
4. **Regenerate Output**: each changed bottle is saved again in a new folder;
   the batch JSON is updated.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (also when there was nothing to apply) |
| 1 | Error: unreadable or older batch file, invalid corrections file, a correction or approval not applied, a changed bottle not saved |
| 2 | Batch not found |

### Example Usage

```bash
# Interactive review of the latest batch
bourbon-process feedback -i

# Corrections file and an approval for one batch
bourbon-process feedback --batch 20261122_193000_001 --corrections my_feedback.json --approve IMG_4643
```

---

## Command: `config`

```bash
bourbon-process config show [--json]   # config files searched and read, every value and its source
bourbon-process config template        # a commented bourbon.toml with every key
```

---

## Environment Variables

Lower priority than CLI arguments, higher than config files. Each setting
with a flag has one; the full list is in README.md. The most used:

| Variable | Description | Default |
|----------|-------------|---------|
| `BOURBON_MODEL` | Ollama vision model | qwen3.6:35b-a3b |
| `BOURBON_TEXT_MODEL` | Fact-check text model | gpt-oss:120b |
| `BOURBON_OLLAMA_URL` | Ollama API endpoint | http://localhost:11434 |
| `BOURBON_OUTPUT_DIR` | Output folder | processed |
| `BOURBON_CONCURRENCY` | Bottle groups analysed in parallel | 2 |
| `BOURBON_WEB_PROVIDER` | Search provider | auto |
| `BOURBON_DEBUG` | Debug logging | false |
| `BOURBON_CONFIG` / `BOURBON_NO_CONFIG` | Config file to read / read none | |
| `BRAVE_API_KEY`, `OLLAMA_API_KEY` | Search API keys (environment only) | unset |

### Precedence Order

1. CLI arguments (highest)
2. Environment variables
3. `./bourbon.toml`
4. `${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml`
5. Default values (lowest)

`--config PATH` replaces 3 and 4 with that file; `--no-config` skips both.

---

## Output Conventions

### Log Format

All log lines go to stdout:

```
{timestamp} {LEVEL} [{bottle id or -}] {module}: {message}
```

Example:
```
2026-11-23 14:30:22,123 INFO [IMG_4643] image_processor.bottle_analyzer: Processing bottle with 3 image(s)
2026-11-23 14:31:05,456 INFO [-] image_processor.enrichment: Unloaded vision model qwen3.6:35b-a3b
```

Configuration errors go to stderr, one line per invalid value, each naming
its source:

```
bourbon-process: configuration error:
  - concurrency=0 (from flag --concurrency): ...
```

---

## Interaction Modes

### Interactive Mode (`--interactive`)

For each bottle the terminal shows its facts, open conflicts and publish
blockers, then:

```
Action [A/R/E/C/N/Q]?
```

**Actions**:
- `A`: approve for publishing (the other publish checks still apply)
- `R`: reject (not saved; an error report is written)
- `E`: edit fields; each edit is a correction for this bottle
- `C`: settle each open conflict (pick a candidate, type a value, or skip)
- `N`: keep as it is, not approved
- `Q`: stop reviewing; remaining bottles are kept, not approved

Merge and split are not available.

### Non-Interactive Mode

The preview report is written and the run continues:

```
Saved batch preview report to processed/batch_20261123_120000_001_preview.md
```

The report ends with the `feedback` and `retry` commands for that batch.

---

## Configuration File Format

Sectioned TOML, read from `${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml`
and `./bourbon.toml` (or only `--config PATH`). Unknown sections and keys are
errors; API keys are refused. `bourbon-process config template` prints every
key with its default, environment variable and flag (the repository keeps
it as `bourbon.example.toml`). An excerpt:

```toml
[processing]
batch_size = 5
concurrency = 2

[model]
vision_model = "qwen3.6:35b-a3b"
keep_models_loaded = false

[text]
enabled = true
model = "gpt-oss:120b"

[server]
ollama_url = "http://localhost:11434"

[retry]
enabled = true
max_attempts = 3

[web]
enabled = true
provider = "auto"
vendor_images = false
deny_domains = []
```

---

## Validation Rules

### Option Validation

Done once by the settings layer for every source (file, environment, flag);
all errors are reported together and the command exits 1. Ranges are in the
options table above.

### Pre-flight Checks

Before processing starts:
1. Verify input folder contains at least 1 supported image
2. Verify the output folder is not the input folder and is writable
3. Verify the hints file, if any
4. Verify Tesseract OCR is installed
5. Verify Ollama is reachable (or can be started)
6. Verify the vision model and, when the fact check is on, the text model are available and capable
7. Check the text model against physical memory

### Post-flight Checks

After processing completes, the run summary reports processed and failed
bottles, error reports, web and fact-check status counts and the
publish-ready count.

---

## Error Handling

### Processing Errors (Non-Fatal)

Individual bottle failures do not stop processing: the bottle gets
`processed/error_report_<bottle_id>.json` and the run continues. Web lookup
and fact-check failures are recorded in the bottle and never fail it.

### Processing Errors (Fatal)

Errors that stop all processing:
- Output directory not writable
- Ollama service unavailable, or lost during the run
- No valid images found
- Model missing, not capable, or incompatible

---

## Testing the CLI

- `tests/contract/test_cli_contracts.py`: subprocess tests of parsing, help, exit codes and pre-flight.
- `tests/contract/test_cli_model_integration.py`: in-process runs with fake models, server, search and text model.
- `tests/contract/test_output_formats.py` and `tests/contract/test_example_sample.py`: output layout and the example sample.

---

## Summary

The CLI provides:
- Processing with web lookups and a fact check (`process`)
- Re-processing of failed bottles (`retry`)
- Corrections and approvals for saved batches (`feedback`)
- Effective configuration and a template (`config`)
- Batch processing with configurable sizing, interactive and non-interactive review
- Explicit errors with their source, and documented exit codes
