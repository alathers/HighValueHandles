# Bourbon Inventory Image Scraper

AI-powered tool to analyze bourbon bottle images, extract metadata, and generate structured documentation using local Ollama models.

## Features

- **Per-stage model selection** — Different models for OCR, vision analysis, metadata generation, etc.
- **Automatic fallback chains** — When the preferred model is unavailable, tries the next in chain automatically
- **Usage tracking & reporting** — See which models were used, how many fallbacks, total processing time per stage
- **MPLTX model serving** — Native process management (start, stop, pull, load/unload, resource monitoring)
- **Metadata extraction** with GPT-OSS text model
- Generates markdown files with bottle information (name, distillery, MSRP, release notes, tasting notes)
- Batch processing of image folders
- Local-only processing (no cloud API calls)

## Model Selector

The model selector assigns the right model to each pipeline stage, manages fallback chains, and tracks usage.

| Stage | Primary Model | Fallback |
|-------|--------------|----------|
| OCR / Text Extraction | Tesseract-OCR | qwen2-vl:7b |
| Vision Model Analysis | qwen3-vl:32b | qwen2-vl:7b |
| Image Grouping (Similarity) | ORB (OpenCV) | Filename heuristics |
| Quantity Detection | Hash-based | Visual analysis |
| Metadata Generation | gpt-oss:120b | qwen3-vl:32b |
| Web Research | DuckDuckGo API | Skip (non-blocking) |
| Error Classification | Rule-based | N/A |
| Preview Generation | Template Markdown | N/A |

### Model Serving Backends

- **MPLTX** (preferred): Native process management with configurable port, GPU layers, context size, and keep-alive
- **Ollama** (backward compatible): Auto-starts Ollama if not running

### Usage Reporting

At the end of processing, a summary is printed showing:
- Total/successful/fallback model calls
- Total processing duration
- Breakdown by stage and by model name

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

### Per-Stage Model Selection

By default, each pipeline stage uses its recommended model (see table above). Override the global default:

```bash
python scraper.py --model llama3.2
```

This sets `llama3.2` as the primary model for all AI stages. Per-stage overrides via environment variables:

```bash
# Override vision analysis stage specifically
BOURBON_VISION_ANALYSIS_MODEL=qwen2-vl:7b python scraper.py
```

### Model Serving Backend

Choose between MPLTX (preferred) and Ollama:

```bash
# Use Ollama backend (backward compatible)
python scraper.py --server-backend ollama

# Configure MPLTX settings
python scraper.py --server-backend mpltx \
  --mpltx-home ~/.mpltx \
  --mpltx-port 11435 \
  --mpltx-gpu-layers 32 \
  --mpltx-context-size 4096
```

### Debug Mode — Model Selector Report

```bash
python scraper.py --debug
```

In debug mode, the model selector prints a stage report showing primary models, fallback chains, and descriptions for each stage.

### Help

```bash
python scraper.py --help
```

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | `.` | Folder containing bottle images |
| `--model` | `qwen3-vl:32b` | Global default model (sets primary for all AI stages) |
| `--server-backend` | `mpltx` | Model serving backend: `mpltx` or `ollama` |
| `--mpltx-home` | `~/.mpltx` | MPLTX data directory |
| `--mpltx-port` | `11435` | MPLTX HTTP API port |
| `--mpltx-gpu-layers` | `-1` | MPLTX GPU offload layers (`-1` = all) |
| `--mpltx-context-size` | `4096` | MPLTX context window size |

### Environment Variables

The script auto-configures Ollama internally:
- `OLLAMA_MAX_LOADED_MODELS=3` - Keep 3 models in memory
- `OLLAMA_KEEP_ALIVE=-1` - Keep models loaded indefinitely

Model server settings:
- `BOURBON_SERVER_BACKEND` - Override backend: `mpltx` or `ollama`
- `BOURBON_MPLTX_HOME` - Override MPLTX data directory
- `BOURBON_MPLTX_PORT` - Override MPLTX port
- `BOURBON_MPLTX_GPU_LAYERS` - Override GPU offload layers
- `BOURBON_MPLTX_CONTEXT_SIZE` - Override context window size
- `BOURBON_<STAGE>_MODEL` - Per-stage model override (e.g., `BOURBON_VISION_ANALYSIS_MODEL`)

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

### Wrong model for a stage

The model selector uses the best model for each stage by default. See the [Model Selector](#model-selector) table above. To override:

```bash
# Global override
python scraper.py --model qwen2-vl:7b

# Per-stage override
BOURBON_VISION_ANALYSIS_MODEL=qwen2-vl:7b python scraper.py
```

### Using Ollama backend instead of MPLTX

```bash
python scraper.py --server-backend ollama --ollama-url http://localhost:11434
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
