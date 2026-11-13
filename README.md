# Bourbon Inventory Image Scraper

AI-powered tool to analyze bourbon bottle images, extract metadata, and generate structured documentation using local Ollama models.

## Features

- Image analysis with Qwen3-VL multimodal model
- Metadata extraction with GPT-OSS text model
- Generates markdown files with bottle information (name, distillery, MSRP, release notes, tasting notes)
- Batch processing of image folders
- Local-only processing (no cloud API calls)

## Documentation

- [Quick Start Guide (5 minutes)](docs/adhd/README.md) - Get up and running quickly
- [Detailed Documentation](docs/detailed/README.md) - Comprehensive reference with troubleshooting, batch processing, and advanced topics

## Prerequisites

### 1. Ollama Installation

Install Ollama from [ollama.com](https://ollama.com) and ensure the Ollama service is running:

```bash
# macOS/Linux
ollama serve &
```

### 2. Pull Required Models

```bash
# Vision model for image analysis (primary)
ollama pull qwen3-vl:32b

# Text model for metadata generation
ollama pull gpt-oss:120b

# Embedding model (optional, for future features)
ollama pull nomic-embed-text
```

Alternative models (lighter/faster options):
- `qwen2-vl:7b` - Faster, less detailed analysis
- `llama3.2-vision:11b` - Open-source alternative
- `gpt-oss:13b` - Faster metadata generation

### 3. System Dependencies

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx
```

**Windows:**
Download from [tesseract-ocr.github.io](https://github.com/tesseract-ocr/tesseract)

## Installation

### Using uv (Recommended)

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or install from pyproject.toml
uv pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
```

This installs:
- `requests` - HTTP client for Ollama API
- `Pillow` - Image processing
- `opencv-python` - Computer vision utilities
- `numpy` - Numerical operations
- `pytesseract` - OCR integration

## Usage

### Basic Usage

Process all JPEG/PNG images in the current directory:

```bash
python scraper.py
```

### Specify Image Folder

```bash
python scraper.py --image-folder ./bourbon_photos
```

### Use Different Model

```bash
python scraper.py --image-folder ./photos --model qwen2-vl:7b
```

### Help

```bash
python scraper.py --help
```

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | `.` | Folder containing bottle images |
| `--model` | `qwen3-vl:32b` | Ollama model for image analysis |

### Environment Variables

The script auto-configures Ollama internally:
- `OLLAMA_MAX_LOADED_MODELS=3` - Keep 3 models in memory
- `OLLAMA_KEEP_ALIVE=-1` - Keep models loaded indefinitely

Adjust models in `scraper.py` if needed.

## Output

Creates `.md` files in `bottle_metadata/` with structure:

```
bottle_metadata/
└── {bottle_name}_{timestamp}.md
```

Each file contains:
- Raw image analysis from vision model
- Extracted metadata (name, distillery, MSRP, release info, tasting notes)

## Troubleshooting

For comprehensive troubleshooting, see [Detailed Documentation](docs/detailed/README.md#troubleshooting).

### "ModuleNotFoundError: No module named 'requests'"

```bash
uv pip install requests pillow
```

### "Ollama connection refused"

Ensure Ollama is running:
```bash
ollama serve
```

Check Ollama status:
```bash
curl http://localhost:11434/api/tags
```

### Model not found

Pull the required model:
```bash
ollama pull qwen3-vl:32b
```

### Out of memory errors

Use smaller models:
```bash
ollama pull qwen2-vl:7b
python scraper.py --model qwen2-vl:7b
```

## Examples

### Process high-value room inventory

```bash
# Images in root directory
python scraper.py

# Images in subfolder
python scraper.py --image-folder ./high_value_spirits

# Faster processing with lighter model
python scraper.py --model qwen2-vl:7b
```

### Batch processing workflow

```bash
# Move all inventory photos to a folder
mkdir inventory_photos
mv IMG_*.jpeg inventory_photos/

# Run scraper
python scraper.py --image-folder inventory_photos

# Review generated metadata
ls -la bottle_metadata/
```

## Development

### Run tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=. --cov-report=html
```

## License

Personal bourbon inventory project.
