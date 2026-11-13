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
- **Multi-angle photo grouping** via visual similarity detection
- **Automatic retry** with error reports
- **Optional vendor reference caching**
- **Publish-ready markdown** output

The system uses Ollama for vision AI and Tesseract for OCR, running entirely on your machine (no cloud services required).

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

# 3. Pull recommended vision model
ollama pull qwen3-vl:32b

# Or pull alternative models:
# ollama pull gpt-oss:120b  # Slower, more accurate
# ollama pull qwen2-vl:7b   # Faster, less accurate

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

# 3. Pull vision model
ollama pull qwen3-vl:32b

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
ollama list  # Should show qwen3-vl:32b

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
- **Lookahead**: 3 images for multi-angle grouping
- **Similarity threshold**: 0.7 (70% visual similarity)
- **Ollama URL**: `http://localhost:11434`
- **Model**: `qwen2-vl:7b` (default), recommended: `qwen3-vl:32b`
- **Output directory**: `processed/`
- **Retry limit**: 3 attempts per bottle
- **Max file size**: 5MB (pre-commit check)

### Environment Variables

Optional environment overrides:

```bash
# Set Ollama URL (default: http://localhost:11434)
export OLLAMA_BASE_URL=http://localhost:11434

# Set maximum concurrent operations
export MAX_CONCURRENCY=3

# Set model override
export DEFAULT_MODEL=qwen3-vl:32b
```

### File-Based Configuration

No configuration file required; all settings via CLI arguments. If you frequently use specific options, consider shell aliases:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias bourbon-process='process --model qwen3-vl:32b --batch-size 5'
alias bourbon-all='process --model qwen3-vl:32b --batch-size all'
alias bourbon-interactive='process --model qwen3-vl:32b --interactive'
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
--lookahead INT                Images to peek ahead for multi-angle photos (default: 3)
--concurrency INT              Max concurrent operations (default: 3)
```

**Model Options:**
```
-m, --model STR                Ollama vision model (default: qwen2-vl:7b)
-u, --ollama-url STR           Ollama service URL (default: http://localhost:11434)
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
2. **Group**: Groups related images by bottle (visual + temporal similarity)
3. **Slice**: Divides bottle groups into batches (default: 5)
4. **Lookahead**: Peek at next N images for multi-angle photos of current batch
5. **Process**: Extract metadata for each bottle in batch
6. **Preview**: Generate markdown preview report
7. **Feedback**: (Optional) Interactive correction opportunity
8. **Repeat**: Continue with next batch

### Lookahead Feature

The lookahead mechanism ensures multi-angle photos stay together:

```
Batch 1 (5 bottles)
  ├─ Bottle A (front.jpg)
  ├─ Bottle B (front.jpg)
  ├─ Bottle C (front.jpg)
  └─ Bottle D (front.jpg)

Lookahead: Checks next 3 images
  Found: Bottle A (side.jpg) → moves into Batch 1
  Found: Bottle C (back.jpg) → moves into Batch 1
  Found: Bottle E (front.jpg) → stays for Batch 2

Result: Batch 1 now has 7 bottle groups with multi-angle sets
```

Configure lookahead:
```bash
process --lookahead 5   # Peek at next 5 images
process --lookahead 0   # Disable lookahead
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

# Pull missing model
ollama pull qwen3-vl:32b

# Verify model loaded
curl http://localhost:11434/api/show
```

**Problem:** Slow processing / timeouts

```bash
# Use smaller model (faster but less accurate)
process --model qwen2-vl:7b

# Reduce concurrency
process --concurrency 1

# Use smaller batch size
process --batch-size 2
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

# Use smaller model
process --model qwen2-vl:7b

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

# Supported: .jpeg .jpg .png .tiff .heic

# Convert to supported format (ImageMagick)
convert photo.tiff photo.jpg
```

**Problem:** Images not grouped correctly

```bash
# Use smaller lookahead (aggressive grouping)
process --lookahead 1

# Disable lookahead (no grouping)
process --lookahead 0

# Manual review: Organize photos into folders per bottle before processing
./organize_photos.sh  # Custom script to sort by timestamp/group
```

### Extraction Accuracy

**Problem:** Wrong bottle name or distillery

```bash
# Use more accurate model
process --model gpt-oss:120b  # Slower but smarter

# Or try qwen3-vl:32b (sweet spot)
process --model qwen3-vl:32b

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

| Model | Accuracy | Speed | Recommended For |
|-------|----------|-------|-----------------|
| `qwen3-vl:32b` | High | Medium | Standard use (sweet spot) |
| `gpt-oss:120b` | Very High | Slow | Difficult bottles, low-quality photos |
| `qwen2-vl:7b` | Medium | Fast | Large collections, speed priority |
| `llava:7b` | Low | Very Fast | Preview/testing only |

Recommendation: Start with `qwen3-vl:32b`, switch to `gpt-oss:120b` only if accuracy issues persist.

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

Typical processing times (qwen3-vl:32b model):

| Collection Size | Bottles | Images | Time (M1 Mac) | Time (i7 Linux) |
|----------------|---------|--------|---------------|-----------------|
| Small | 1 | 4 | 45 sec | 60 sec |
| Small | 5 | 20 | 3 min | 4 min |
| Medium | 10 | 40 | 7 min | 10 min |
| Large | 50 | 200 | 35 min | 50 min |

Multiplied by ~1.5x for gpt-oss:120b model, divided by ~0.6x for qwen2-vl:7b model.

---

## Quick Reference Cheat Sheet

```bash
# Basic usage
process                                    # Process current dir
process -f /path/to/photos                 # Process specific folder
process -o /custom/output                  # Custom output dir

# Model selection
process -m qwen3-vl:32b                    # Best balance
process -m gpt-oss:120b                    # Most accurate
process -m qwen2-vl:7b                     # Fastest

# Batch control
process -b all                             # No batching
process -b 10                              # 10-bottle batches
process --lookahead 5                      # Aggressive grouping

# Interactive & review
process -i                                 # Interactive mode
process --dry-run                          # Test grouping

# Troubleshooting
process --debug                            # Verbose logging
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
