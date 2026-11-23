# Bourbon Image Processor - Quick Start (5 Minutes)

## What It Does

Processes photos of bourbon bottles to extract:
- Bottle name, distillery, MSRP
- Release info (proof, age, batch)
- Tasting notes
- Organizes everything into publishable markdown files

## Install (once)

**macOS**

```bash
brew install uv tesseract tesseract-lang ollama
brew services start ollama          # or open the Ollama desktop app
```

**Linux (Debian/Ubuntu)** (Fedora: `sudo dnf install -y tesseract tesseract-langpack-eng mesa-libGL`)

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh    # installs the `ollama` systemd service
sudo systemctl enable --now ollama
```

**Both** (from the repository root):

```bash
uv sync                              # installs the bourbon-process command
ollama pull qwen3.6:35b-a3b          # the default vision model (~23 GB)
```

Skip the `ollama pull` if you like: add `--pull` to a run and the tool
downloads the model after checking there is enough free disk space. Without
`--pull` nothing is ever downloaded.

## Model

One vision model reads the labels; Tesseract OCR text goes along as a hint.
Default: `qwen3.6:35b-a3b`. Other vision models that work: `qwen3.8:27b`,
`gemma4:26b` (smaller). Text-only models are rejected.

```bash
uv run bourbon-process process -f photos/ -m gemma4:26b
```

Ollama is the only backend. If Ollama is not running, the tool starts
`ollama serve` for the run and stops it afterwards (an Ollama you started is
never stopped). On Linux with the systemd service, start the service instead:
`sudo systemctl start ollama`.

## Key Commands

```bash
# Check everything first (starts nothing, downloads nothing, writes nothing)
uv run bourbon-process process -f photos/ --dry-run

# Process photos/ -> processed/
uv run bourbon-process process -f photos/ -o processed/

# Batch size (default 5 bottles per batch with a preview report)
uv run bourbon-process process -f photos/ --batch-size 10
uv run bourbon-process process -f photos/ --batch-size all

# Review and correct each batch before it is saved
uv run bourbon-process process -f photos/ --interactive

# Retry a failed bottle
uv run bourbon-process retry --report processed/error_report_<bottle_id>.json

# Apply corrections to the latest batch
uv run bourbon-process feedback -o processed/ --interactive
```

Photos of one bottle are grouped by EXIF capture time (taken at most 12 s
apart, at most 3 per bottle). Tune with `--group-gap-seconds` and
`--group-max-size`. If the vision model says a group mixes bottles, it is
split and re-read (cap: `--split-budget-ratio`, 0 = off). `--lookahead` is
deprecated and ignored.

## Output Structure

```
processed/
  ├── bottle_name_1/
  │   ├── bottle_name_1.md          # Markdown page
  │   ├── bottle_name_1.json        # Full metadata
  │   ├── bottle_name_1.txt         # OCR text
  │   └── images/                   # Copies of the photos (originals untouched)
  ├── batch_<run>_001_preview.md    # Batch preview report
  ├── batch_<run>_001.json          # Batch data for `feedback`
  └── error_report_<bottle_id>.json # One per failed bottle, for `retry`
```

Exit codes: 0 ok, 1 bad argument / Tesseract missing / some bottles failed,
2 Ollama or model problem, 3 no photos, 4 every bottle failed, 5 output
folder not writable.

## Common Errors & Fixes

**"Model server unavailable" (exit 2)**

```bash
curl http://localhost:11434/api/tags   # is Ollama answering?
ollama serve                           # start it (macOS or Linux)
sudo systemctl start ollama            # Linux systemd service
uv run bourbon-process process -f photos/ -u http://other-host:11434   # Ollama elsewhere
```

**"... is not installed on the ollama server" (exit 2)**

```bash
ollama list                        # what is installed
ollama pull qwen3.6:35b-a3b        # or re-run with --pull
```

**"Not enough disk space to pull"**: free some space, lower the margin with
`--min-free-gb`, or use a smaller model (`-m gemma4:26b`).

**"No supported images found"**

```bash
# Check folder has image files
ls -la /your/image/folder

# Supported formats: .bmp .heic .jpeg .jpg .png .tiff .webp
# Earlier output folders (with a .bourbon_output marker) are skipped
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
- **Shoot each bottle in one go**: photos taken within 12 s of each other become one bottle; pause before the next bottle
- **Review output**: Check the .md files after first batch to tune extraction
- **Batch mode**: Use `--batch-size all` for large collections, but review first

## Next Steps

For detailed configuration, troubleshooting, and advanced usage, see [../detailed/README.md](../detailed/README.md).
