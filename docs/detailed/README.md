# Bourbon Image Processor - Complete User Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [CLI Reference](#cli-reference)
5. [Processing Modes](#processing-modes)
6. [Output Format](#output-format)
7. [Batch Processing](#batch-processing)
8. [Interactive Mode](#interactive-mode)
9. [Error Handling](#error-handling)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

---

## Overview

The Bourbon Image Processor is a local CLI tool that extracts bourbon bottle metadata from images using computer vision and OCR. It supports:

- **Batch processing** with configurable preview checkpoints
- **Interactive feedback** for correcting extractions
- **Multi-angle photo grouping** by EXIF capture time
- **Automatic retry** with error reports
- **Optional vendor reference caching**
- **Publish-ready markdown** output

The system uses Ollama for vision AI and Tesseract for OCR, running entirely on your machine (no cloud services required).

### Per-Stage Model Selection

Only the vision-analysis stage calls a model: the `-m/--model` vision model
(default `qwen3.6:35b-a3b`). There is no automatic fallback model, and no text
model (such as gpt-oss) is used.

| Stage | Primary | Notes |
|-------|---------|-------|
| OCR / Text Extraction | Tesseract-OCR | Hint text for the vision model |
| Vision Model Analysis | `-m` (default `qwen3.6:35b-a3b`) | The only model called |
| Image Grouping | EXIF capture time + vision same-bottle check | `--group-gap-seconds`, `--group-max-size`, `--split-budget-ratio` |
| Quantity Detection | Hash-based | |
| Metadata Generation | (not used) | Not implemented |
| Web Research | (not used) | |
| Error Classification | Rule-based | |
| Preview Generation | Template Markdown | |

### Model Serving Backends

- **Ollama** (the only backend, macOS and Linux): a running Ollama is reused and never stopped; otherwise `ollama serve` is started for the run and stopped afterwards

### Usage Tracking

At the end of processing, a summary is printed showing total calls, successful calls, fallback count, and duration — broken down by stage and model name.

---

## Installation

### Prerequisites

**Hardware Requirements:**
- CPU with AVX support (recommended)
- 8GB+ RAM (16GB+ recommended for vision models)
- 10-20GB disk space for models

**Software Requirements:**
- Python 3.11+
- Ollama service
- Tesseract OCR
- macOS or Linux (Windows experimental support)

### macOS Installation

```bash
# 1. Install Ollama with Homebrew
brew install ollama

# 2. Start Ollama service
ollama serve

# 3. Pull the default vision model (or let the tool do it: re-run with --pull)
ollama pull qwen3.6:35b-a3b

# Any other vision-capable model works with -m, e.g.:
# ollama pull gemma4:26b            # smaller

# 4. Install Tesseract OCR
brew install tesseract

# 5. Verify Tesseract
tesseract --version  # Should show v5.x+

# 6. Clone and install bourbon-image-processor
cd /path/to/your/projects
git clone <repo-url>
cd BourbonInventory/HighValueRoom

# 7. Run install script (installs Python dependencies)
./install.sh

# 8. Verify installation
process --help
```

### Linux Installation (Ubuntu/Debian)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama service
systemctl start ollama  # or: ollama serve

# 3. Pull the default vision model (or re-run the tool with --pull)
ollama pull qwen3.6:35b-a3b

# 4. Install Tesseract OCR
sudo apt update
sudo apt install tesseract-ocr

# 5. Install Python dependencies
cd /path/to/BourbonInventory/HighValueRoom
./install.sh

# 6. Verify installation
process --help
```

### Verify Installation

```bash
# Check Ollama models
ollama list  # Should show qwen3.6:35b-a3b (or the model you pass with -m)

# Check Ollama health
curl http://localhost:11434/api/tags  # Should return model list

# Check Tesseract
tesseract --version

# Check the processor
process --help
```

---

## Configuration

### Default Settings

The processor uses sensible defaults:

- **Batch size**: 5 bottles
- **Grouping**: photos at most 12 s apart (EXIF capture time), at most 3 per bottle
- **Ollama URL**: `http://localhost:11434`
- **Model**: `$BOURBON_MODEL`, else `qwen3.6:35b-a3b`
- **Output directory**: `processed/`
- **Retry limit**: 3 attempts per bottle
- **Max file size**: 5MB (pre-commit check)

### Environment Variables

Optional environment overrides:

```bash
# Default vision model when -m is not given
export BOURBON_MODEL=qwen3.6:35b-a3b

# Ollama model store (default ~/.ollama/models)
export OLLAMA_MODELS=/path/to/models
```

### File-Based Configuration

No configuration file required; all settings via CLI arguments. If you frequently use specific options, consider shell aliases:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias bourbon-batch='bourbon-process process --batch-size 5'
alias bourbon-all='bourbon-process process --batch-size all'
alias bourbon-interactive='bourbon-process process --interactive'
```

---

## CLI Reference

### Global Options

```bash
process --help           # Show help
process --version        # Show version
process --debug          # Enable debug logging
```

### Main Command: `process`

**Syntax:**
```bash
process [OPTIONS]
```

**Input/Output Options:**
```
-f, --image-folder PATH        Input folder containing images (default: current dir)
-o, --output-dir PATH          Output directory for processed bottles (default: processed/)
```

**Processing Options:**
```
-b, --batch-size STR|NUM        Bottles per batch: number or "all" (default: 5)
--group-gap-seconds SECS       Most seconds between photos of one bottle (default: 12)
--group-max-size N             Most photos per bottle, 0 = no limit (default: 3)
--split-budget-ratio RATIO     Extra analyses per group for splitting groups the model
                               reports as mixed; at least 4 per run, 0 = off (default: 0.35).
                               An analysis is one vision request (retried once if unusable) + OCR
--lookahead INT                Deprecated and ignored (warns); accepted (0-10) for old scripts
--concurrency INT              Accepted; processing is currently sequential (default: 3)
```

**Model Options:**
```
-m, --model STR                Vision model (Ollama tag) (default: $BOURBON_MODEL, else qwen3.6:35b-a3b)
-u, --ollama-url STR           Ollama service URL (default: http://localhost:11434)
--server-backend STR           Model serving backend; only "ollama" (default: ollama)
--server-start-timeout SECS    Wait for an `ollama serve` started by the run (default: 30)
--request-timeout SECS         Per-request model timeout (default: 600)
--keep-server                  Leave an `ollama serve` started by the run running
--no-server-autostart          Never start `ollama serve`; fail if Ollama is not running
```

**Behavior Options:**
```
-i, --interactive               Enable interactive preview and review
-v, --include-vendor-images    Include vendor/supply chain images
-r, --allow-retry               Allow automatic retry on failures (default: True)
--no-retry                      Disable automatic retry
--dry-run                       Run without processing (for testing)
```

### Subcommand: `feedback`

**Apply user corrections to processed batches.**

```bash
feedback [OPTIONS]
```

**Options:**
```
-o, --output-dir PATH          Directory containing processed batches (default: processed/)
--batch STR                    Batch ID (e.g., "1" for batch_1_preview.md)
--corrections PATH             Path to JSON corrections file
-i, --interactive              Interactive feedback collection mode
```

**Usage Examples:**
```bash
# Apply corrections from batch 1
feedback --batch 1

# Apply corrections from JSON file
feedback --corrections my_corrections.json

# Interactive correction mode
feedback --interactive
```

### Subcommand: `retry`

**Retry processing failed bottles from error reports.**

```bash
retry --report PATH [OPTIONS]
```

**Options:**
```
--report PATH                  Path to error_report.json (required)
-o, --output-dir PATH          Output directory for processed bottles (default: processed/)
```

**Usage Example:**
```bash
retry --report processed/bottle_name/error_report.json
```

---

## Processing Modes

### Mode 1: Basic Processing

Process all images in current directory:

```bash
process
```

**Behavior:**
- Scans current directory for supported images
- Groups images by bottle
- Processes 5 bottles at a time
- Generates preview reports between batches
- Saves results to `processed/`

### Mode 2: Specified Folder

Process images from a specific folder:

```bash
process --image-folder /path/to/bottle/photos
```

### Mode 3: Single Batch (All at Once)

Process entire collection without preview reports:

```bash
process --batch-size all
```

**When to use:**
- Small collections (< 10 bottles)
- High-confidence quality photos
- After tuning on smaller batches

### Mode 4: Custom Batch Size

Configure batch size for your workflow:

```bash
# Larger batches (10 bottles)
process --batch-size 10

# Smaller batches (2 bottles) for careful review
process --batch-size 2
```

### Mode 5: Dry Run

Test grouping without processing:

```bash
process --dry-run
```

Shows which images would be grouped as bottles, but doesn't process them.
Nothing is written to disk and no model is started, pulled or loaded.

---

## Output Format

### Directory Structure

```
processed/
├── batch_1_preview.md          # Preview report (batch mode only)
├── batch_1.json                # Batch metadata (if needed)
├── batch_2_preview.md
└── {bottle_name}/
    ├── {bottle_name}.md       # Markdown for website
    ├── {bottle_name}.json      # Full metadata
    ├── {bottle_name}.txt       # Raw OCR text
    ├── IMG_0001.jpg            # Original bottle images
    ├── IMG_0002.jpg
    └── vendor_image.jpg       # Optional vendor reference (if found)
```

### Markdown File Format

Generated `.md` files include:

```markdown
# {Bottle Name}

**Distillery:** {Distillery Name}

**MSRP:** ${price}

**Proof:** {proof}

**Age:** {age}

## Release Information
{specific_release, batch number, special designation}

## Tasting Notes
{tasting notes fromネル/labels or placeholders}

## Quantity
**Count:** {number}
**Price:** ${price}/bottle
**Total Value:** ${total}
*Source: extracted|manual*

## References
- [Vendor Page](url)
- [Review Site](url)
```

### JSON File Format

```json
{
  "name": "Bottle Name",
  "distillery": "Distillery",
  "specific_release": "Release info",
  "msrp": "89.99",
  "proof": "103",
  "age": "9 years",
  "tasting_notes": "...",
  "quantity": {
    "count": 1,
    "price": "89.99",
    "source": "extracted"
  },
  "extracted_at": "2026-11-13T12:00:00",
  "metadata": {
    "full_text": "Raw OCR text...",
    "selected_text": "Selected text subset...",
    "source_images": ["/path/to/IMG_0001.jpg", "/path/to/IMG_0002.jpg"]
  },
  "reference_links": [
    {"url": "https://...", "label": "Official Vendor Page"}
  ],
  "corrections": []
}
```

### Raw OCR Text

The `.txt` file contains concatenated OCR text from all bottle images.

---

## Batch Processing

### How Batching Works

The processor divides your images into batches for manageable review:

1. **Scan**: Reads all images from input folder
2. **Group**: Groups photos by bottle using EXIF capture time (see below)
3. **Slice**: Divides bottle groups into batches (default: 5)
4. **Lookahead**: Peek at next N images for multi-angle photos of current batch
5. **Process**: Extract metadata for each bottle in batch
6. **Preview**: Generate markdown preview report
7. **Feedback**: (Optional) Interactive correction opportunity
8. **Repeat**: Continue with next batch

### Grouping by Capture Time

Photos taken by one camera at most `--group-gap-seconds` apart (default 12)
are one bottle; runs longer than `--group-max-size` (default 3) are split at
their largest time gap. File names are never compared (consecutive numbers
are usually different bottles). Photos without EXIF time are single-photo
bottles unless named like `IMG_1_front` / `IMG_1_back`.

### Same-Bottle Check (replaces lookahead)

Grouping finishes before batching, so a bottle never spans two batches and
the old lookahead step is gone (`--lookahead` is still accepted, ignored, and
logs a deprecation warning).

When the vision model says a multi-photo group shows different products, the
group is split at its largest capture-time gap and each half is read again.
The split is kept only if the halves disagree on a field both show (name,
release, batch, proof within 0.5, age); a field missing on one half is a
wildcard, and so is a number in the name seen on one half only ("1792 Full
Proof" vs "1792 Full Proof 125"). Otherwise the group stays one bottle with a
review note and `same_bottle` cleared to `null`. Known limits: sibling bottles
of one product with no visible difference may be merged, and a name that is a
subset of the other ("Stagg" vs "Stagg Jr") matches unless another field read
on both halves differs. Parts the model still calls mixed are split again,
down to single photos; if such a part turns out to be one bottle that agrees
with the other half, the whole group stays one bottle. Each kept part is
saved as its own bottle folder with only its photos.

Bottles whose photos are at most 60 s apart and whose labels agree get a
"Possibly the same bottle as <id>" review note (never merged). When at least
25% of those notes involve a group of exactly `--group-max-size` photos, the
run suggests `--group-max-size N+1`.

Extra analyses are capped for the whole run, shared by all batches. The cap
counts analyses, not model requests (a retry after an unusable answer and the
OCR pass are not counted):

```bash
process --split-budget-ratio 0.35   # default: up to ceil(0.35 x groups) extra analyses (min 4)
process --split-budget-ratio 1      # check more mixed groups
process --split-budget-ratio 0      # turn the check off
```

### Preview Reports

Generated after each batch (unless `--batch-size all`):

```markdown
# Batch 1 Preview Report

## Processed: 5 bottles
## Failed: 0

### Bottle 1: {bottle_name}
**Images**: IMG_0001.jpg, IMG_0002.jpg
**Name**: {extracted_name}
**Distillery**: {distillery}
**MSRP**: ${msrp}
**Proof**: {proof}
**Age**: {age}

Review: [ ] OK / [ ] Needs correction

...
```

**File location:** `processed/batch_{N}_preview.md`

---

## Interactive Mode

### Enable Interactive Mode

```bash
process --interactive
```

### Workflow

1. Process batch (e.g., 5 bottles)
2. Display preview in terminal
3. For each bottle:
   ```
   Bottle 1: Eagle Rare 10 Year
   Name: Eagle Rare 10 Year
   Distillery: Buffalo Trace
   MSRP: $34.99
   Proof: 101
   Age: 10 years

   Actions:
     [a] Accept - Continue to next
     [e] Edit - Correct fields
     [s] Skip - Reject this bottle
     [q] Quit - Stop processing

   Your choice (a/e/s/q):
   ```

4. Apply corrections immediately or save JSON
5. System learns from corrections for remaining batches

### Correction Format

When editing, specify field corrections:

```
Field to correct: name
Original value: Eagle Rare 10 Year
Corrected value: Eagle Rare 10 Year Single Barrel

Field to correct: (or press Enter to finish)
```

---

## Error Handling

### Automatic Retry

The system automatically retries failed bottles up to 3 times:

```
[INFO] Processing bottle: IMG_1234
[INFO] Attempt 1/3 failed: Connection timeout
[INFO] Attempt 2/3 failed: Model inference error
[INFO] Attempt 3/3 succeeded
```

Disable automatic retry:
```bash
process --no-retry
```

### Error Reports

When a bottle fails completely (3 retry attempts), an error report is generated:

```json
{
  "bottle_id": "IMG_1234",
  "images": ["IMG_1234.jpg", "IMG_1235.jpg"],
  "error_message": "Failed to extract text: Image too blurry",
  "suggested_fix": {
    "name": "Manual review required",
    "notes": "Label text is illegible due to motion blur"
  },
  "timestamp": "2026-11-13T12:34:56"
}
```

**File location:** `processed/{bottle_id}/error_report.json`

### Retry Failed Bottles

Apply fixes and retry:

```bash
# Option 1: Edit error_report.json with corrections
# Option 2: Use hints.md for guidance (see below)
retry --report processed/IMG_1234/error_report.json
```

### hints.md

For difficult bottles, add guidance:

```markdown
# Processing Hints

## Bottle-Specific Hints

### Eagle Rare Batch A123

**Hint:** Handwritten batch number

**Context:** Batch number A123 is handwritten on bottom label, inconsistent font

**Suggested ACTION:** Manual verify batch number extracted from bottom
```

---

## Troubleshooting

### Ollama Issues

**Problem:** "Ollama service not reachable"

```bash
# Diagnose
curl http://localhost:11434/api/tags

# Fix 1: Start Ollama
ollama serve

# Fix 2: Check if running on different port
lsof -i :11434

# Fix 3: Specify custom URL
process --ollama-url http://localhost:11435
```

**Problem:** "Model not available"

```bash
# Check installed models
ollama list

# Pull the missing vision model (or re-run with --pull)
ollama pull qwen3.6:35b-a3b

# Verify model loaded
curl http://localhost:11434/api/show
```

**Problem:** Slow processing / timeouts

```bash
# Use a smaller vision model (faster but less accurate)
process --model gemma4:26b

# Allow more time per request (a cold model load can take minutes)
process --request-timeout 1200
```

### Tesseract OCR Issues

**Problem:** "Tesseract OCR not found"

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev

# Verify
tesseract --version  # Should output v5.x+
```

**Problem:** Poor OCR quality

```bash
# Check image quality (too small, blurry, compressed)
file IMG_0001.jpg
identify IMG_0001.jpg  # Requires ImageMagick

# Use PNG for better OCR (preprocess with ImageMagick)
convert input.tif -resample 300 output.png
```

**Problem:** grayscale images, rotated text**

```bash
# System auto-orientes, but verify with tesseract CLI
tesseract IMG_0001.jpg stdout
```

### Memory Issues

**Problem:** Out of memory during batch processing

```bash
# Reduce batch size
process --batch-size 3

# Reduce concurrency
process --concurrency 1

# Use a smaller vision model
process --model gemma4:26b

# Check system memory
free -h  # Linux
vm_stat  # macOS
```

### Permission Issues

**Problem:** "Output directory not writable"

```bash
# Check permissions on output directory
ls -la processed/

# Fix ownership
sudo chown $USER:$USER processed/
chmod u+w processed/

# Alternative: specify different output
process --output-dir /tmp/bourbon_output
```

### Image Issues

**Problem:** "No supported images found"

```bash
# Check supported formats
ls -la /path/to/images

# Supported: .bmp .heic .jpeg .jpg .png .tiff .webp
# Earlier outputs (folders with a .bourbon_output marker) and hidden folders are skipped

# Convert to supported format (ImageMagick)
convert photo.tiff photo.jpg
```

**Problem:** Images not grouped correctly

```bash
# See the groups without processing
process --dry-run

# Photos of one bottle taken further apart than 12 s
process --group-gap-seconds 20

# Four or more angles per bottle
process --group-max-size 5

# Photos without EXIF capture time are single-photo bottles; name them
# IMG_1_front.jpg / IMG_1_back.jpg to pair them
```

### Extraction Accuracy

**Problem:** Wrong bottle name or distillery

```bash
# Try another vision-capable model
process --model gemma4:26b

# Manual editing: Use interactive mode for correction
process --interactive

# Or edit markdown files directly
vim processed/bottle_name/bottle_name.md
```

**Problem:** Missing tasting notes

```bash
# Common if tasting notes are on website, not label
# Add manually to markdown:

vi processed/bottle_name/bottle_name.md
# Add under ## Tasting Notes section
```

**Problem:** Incorrect MSRP

```bash
# Manual correction
process feedback --batch 1 --corrections msrcp_fix.json

# Where msrcp_fix.json:
{
  "bottles": [
    {
      "name": "Eagle Rare 10 Year",
      "corrections": [
        {"field": "msrp", "old_value": "29.99", "new_value": "34.99"}
      ]
    }
  ]
}
```

### Debug Mode

For detailed troubleshooting:

```bash
process --debug
```

This enables:
- Disassembled step-by-step logging
- Image grouping decisions
- Similarity scores
- OCR text extraction logs
- Model inference logs

---

## Advanced Topics

### Model Selection

Available vision models (via Ollama):

Any vision-capable Ollama model can be passed with `-m`; text-only models
(such as `gpt-oss`) are rejected by pre-flight.

| Model | Notes |
|-------|-------|
| `qwen3.6:35b-a3b` | Default; tested end to end |
| `qwen3.8:27b` | Alternative (~18 GB) |
| `gemma4:26b` | Smallest of these (~16 GB) |

The old `llama3.2-vision` builds are listed by Ollama but fail on current
Ollama versions ("no longer compatible"); the run stops with exit code 2.

### Vendor Image Caching

The system can optionally cache official product photos:

```bash
process --include-vendor-images
```

**What it does:**
1. Searches web for official photos matching extracted metadata
2. Verifies match (name, distillery, proof, batch)
3. Saves `vendor_image.jpg` in bottle folder
4. Marks as vendor reference in JSON metadata

**When useful:**
- Creating a reference library
- Comparing different releases
- Quality control vs. official specs

**Note:** Caching failure does NOT block bottle processing.

### Quantity Detection

The system tracks quantity of identical items:

**Scenario 1: Multiple identical bottles in one image**

- Detects count via visual analysis / labels
- Sets `quantity.count = N`
- Saves single bottle record

**Scenario 2: Duplicate photo sets**

- Identical image files = same bottle
- Sets `quantity.count = number of duplicates`

**Scenario 3: Manual quantity**

- Use JSON editing to set manual quantity:

```json
{
  "quantity": {
    "count": 12,
    "price": "0.00",
    "source": "manual"
  }
}
```

### Custom Workflows

**Integration with website CMS:**

```bash
# Process bottles
process --batch-size all

# Sync markdown files to website CMS
./sync_to_site.sh processed/

# Or pipe via script for JSON API upload
cat processed/*/bottle.json | jq -r '.' | \
  curl -X POST https://your-cms.com/api/bottles -d @-
```

**Video preparation workflow:**

```bash
# Process bottles for video script
process --interactive

# Generate script from extracted data
python scripts/generate_video_script.py processed/

# Output: video_script.md with bottle descriptions, pricing, tasting notes
```

### Batch Automation with Crontab

Schedule automatic processing:

```bash
# Crontab entry: Process new photos daily at 10pm
0 22 * * * cd /path/to/bourbon && ./process --batch-size all --image-folder /photos/new_today >> /var/log/bourbon_process.log 2>&1
```

### Extract-Only Workflow

Skip OCR, just group images:

```bash
# This feature is currently under development
# Workaround: Use dry-run mode
process --dry-run
```

---

## Performance Benchmarks

Each bottle is one vision call with up to 4 photos (downscaled to 1280 px).
With `qwen3.6:35b-a3b` a warm call took a few seconds to under a minute per
bottle in testing; the first call also pays the model load (minutes on a cold
start). Times depend heavily on the model and hardware.

---

## Quick Reference Cheat Sheet

```bash
# Basic usage
process                                    # Process current dir
process -f /path/to/photos                 # Process specific folder
process -o /custom/output                  # Custom output dir

# Vision model (the only model used)
process -m qwen3.6:35b-a3b                 # Default
process -m gemma4:26b                      # Smaller
process --pull                             # Download the model if missing (disk-checked)

# Model serving
process -u http://gpu-box:11434            # Use Ollama on another host
process --no-server-autostart              # Require an already-running Ollama

# Batch control
process -b all                             # No batching
process -b 10                              # 10-bottle batches
process --split-budget-ratio 0             # No same-bottle split check

# Interactive & review
process -i                                 # Interactive mode
process --dry-run                          # Test grouping
process --debug                            # Verbose + model selector report

# Troubleshooting
retry --report processed/bottle/error.json  # Retry failed
feedback -b 1 -i                           # Correct previous batch

# Verify setup
ollama list                               # Check models
tesseract --version                       # Check OCR
process --help                            # All options
```

---

## Support

For issues, questions, or contributions:

- Check docs/adhd/README.md for 5-minute quick start
- Review this detailed guide for troubleshooting
- Review specs/001-bourbon-image-processor/spec.md for feature specification
- Check [README.md](../../README.md) for project overview
