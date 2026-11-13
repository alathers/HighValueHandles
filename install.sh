#!/usr/bin/bash
# Installation script for Bourbon Image Processor
# Supports macOS and Linux

set -e

echo "🥃 Bourbon Image Processor Installation"
echo "=========================================="
echo ""

# Detect OS
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    echo "Detected: macOS"
elif [[ "$OS" == "Linux" ]]; then
    echo "Detected: Linux"
else
    echo "Error: Unsupported operating system: $OS"
    echo "Supported: macOS, Linux"
    exit 1
fi
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "✓ uv found: $(which uv)"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [[ -f "pyproject.toml" ]]; then
    uv sync
    echo "✓ Dependencies installed from pyproject.toml"
elif [[ -f "requirements.txt" ]]; then
    uv pip install -r requirements.txt
    echo "✓ Dependencies installed from requirements.txt"
else
    echo "❌ Neither pyproject.toml nor requirements.txt found"
    exit 1
fi

if [[ -f "dev-requirements.txt" ]]; then
    echo "Installing development dependencies..."
    uv pip install -r dev-requirements.txt
    echo "✓ Development dependencies installed"
fi
echo ""

# Install Tesseract OCR
echo "🔍 Installing Tesseract OCR..."
if [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    brew install tesseract tesseract-lang
    echo "✓ Tesseract OCR installed via Homebrew"
elif [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx
        echo "✓ Tesseract OCR installed via apt-get"
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y tesseract tesseract-langpacks
        echo "✓ Tesseract OCR installed via dnf"
    elif command -v yum &> /dev/null; then
        sudo yum install -y tesseract
        echo "✓ Tesseract OCR installed via yum"
    else
        echo "❌ Unsupported Linux package manager. Please install Tesseract OCR manually."
        exit 1
    fi
fi
echo ""

# Verify installations
echo "🔎 Verifying installations..."
if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n 1)
    echo "✓ Tesseract: $TESSERACT_VERSION"
else
    echo "❌ Tesseract not found in PATH"
    exit 1
fi

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo "✓ uv: $UV_VERSION"
else
    echo "❌ uv not found"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "✓ Python: $PYTHON_VERSION"
echo ""

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✓ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found. Run 'uv pip install pre-commit' to enable git hooks."
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Install and start Ollama: https://ollama.com"
echo "2. Pull required models: ollama pull qwen3-vl:32b && ollama pull gpt-oss:120b"
echo "3. Run: bourbon-process --help"
echo ""
