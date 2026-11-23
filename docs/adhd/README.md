# Bourbon Image Processor - Quick Start (5 Minutes)

## What It Does

Processes photos of bourbon bottles to extract:
- Bottle name, distillery, MSRP
- Release info (proof, age, batch)
- Tasting notes
- Organizes everything into publishable markdown files

## Prerequisites

```bash
# 1. Install Ollama (macOS only)
brew install ollama

# 2. Pull a vision model
ollama pull qwen3-vl:32b  # Recommended: fast & accurate
# OR: ollama pull gpt-oss:120b  # Slower but smarter

# 3. Install Tesseract OCR
# macOS: brew install tesseract
# Linux: apt install tesseract-ocr

# 4. Install Python dependencies (run install.sh)
./install.sh
```

## Model Selector

The system uses different models for different pipeline stages:

| Stage | Primary Model | Fallback |
|-------|--------------|----------|
| OCR / Text Extraction | Tesseract-OCR | qwen2-vl:7b |
| Vision Model Analysis | qwen3-vl:32b | qwen2-vl:7b |
| Metadata Generation | gpt-oss:120b | qwen3-vl:32b |

Override the global default:

```bash
# Use a faster model for all AI stages
process --model qwen2-vl:7b
```

## Key Commands

**Process one bottle** (current folder images → processed/ folder):

```bash
# Basic - process all images in current directory
process

# Specify folder
process --image-folder /path/to/bottle/images

# Choose model
process --model qwen3-vl:32b
```

**Process multiple bottles** (batch mode):

```bash
# Default: 5 bottles per batch with preview reports
process

# Custom batch size
process --batch-size 10

# Process all without stopping
process --batch-size all
```

**Interactive mode** (review and correct between batches):

```bash
process --interactive
```

**Retry failed bottles**:

```bash
process retry --report processed/bottlename/error_report.json
```

## Output Structure

```
processed/
  ├── bottle_name_1/
  │   ├── bottle_name_1.md        # Markdown for website
  │   ├── bottle_name_1.json       # Full metadata
  │   ├── bottle_name_1.txt        # Raw OCR text
  │   ├── IMG_0001.jpg             # Original images
  │   ├── IMG_0002.jpg
  │   └── vendor_image.jpg        # Optional cached vendor photo
  └── bottle_name_2/
      └── ...
```

## Common Errors & Fixes

**"Ollama service not reachable"**

```bash
# Start Ollama if not running
ollama serve

# Or check if wrong URL
process --ollama-url http://localhost:11434
```

**"Model not available"**

```bash
# List available models
ollama list

# Pull the model
ollama pull qwen3-vl:32b
```

**"No supported images found"**

```bash
# Check folder has jpg/png/jpeg/tiff/heic files
ls -la /your/image/folder

# Supported formats: .jpeg .jpg .png .tiff .heic
```

**"Tesseract OCR not found"**

```bash
# macOS
brew install tesseract

# Linux
# Ubuntu/Debian
sudo apt install tesseract-ocr

# Fedora
sudo dnf install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Output directory not writable**

```bash
# Check permissions
ls -la

# Create processed folder manually
mkdir -p processed
chmod u+w processed
```

## Tips

- **Quality matters**: Bright, sharp photos with clear labels work best
- **Multiple angles**: Take front, back, label close-ups for better extraction
- **Organize input**: Put each bottle's photos in separate folders for best results
- **Review output**: Check the .md files after first batch to tune extraction
- **Batch mode**: Use `--batch-size all` for large collections, but review first

## Next Steps

For detailed configuration, troubleshooting, and advanced usage, see [../detailed/README.md](../detailed/README.md).
