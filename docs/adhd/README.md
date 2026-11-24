# Bourbon Image Processor - Quick Start (5 Minutes)

## What It Does

Reads photos of bourbon bottles and writes one folder per bottle with:
- Bottle name, distillery, release, batch, proof, ABV, age, tasting notes
- MSRP only when a source states it (otherwise "Unknown" with the reason)
- The sources of every value: label, OCR text, web pages, or you
- A Markdown page and a JSON record, plus a list of every website consulted

Nothing is marked publish-ready until the checks pass **and you approve it**.

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
ollama pull qwen3.6:35b-a3b          # vision model (~23 GB, 32 GB+ RAM)
ollama pull gpt-oss:120b             # fact-check model (~65 GB, ~70 GB free memory)
```

Skip the `ollama pull` if you like: add `--pull` to a run and the tool
downloads the missing models after checking there is enough free disk space.
Without `--pull` nothing is ever downloaded.

Smaller machine? Use `--text-model gpt-oss:20b` (~14 GB) instead of
`gpt-oss:120b`, or `--no-verify` to skip the fact check (then nothing can be
publish-ready).

## Models

- **Vision** (`-m`, default `qwen3.6:35b-a3b`) reads the label. Other vision
  models that work: `qwen3.8:27b`, `gemma4:26b` (smaller).
- **Fact check** (`--text-model`, default `gpt-oss:120b`) quotes the label,
  OCR text and web pages; Python checks every quote and decides each field.

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

# Review each bottle before it is saved (approve, edit, settle conflicts)
uv run bourbon-process process -f photos/ --interactive

# One batch for the whole run (the models swap only once)
uv run bourbon-process process -f photos/ --batch-size all

# Approve a bottle later, or apply a corrections file
uv run bourbon-process feedback -o processed/ --approve IMG_4643
uv run bourbon-process feedback -o processed/ --corrections fix.json

# Retry a failed bottle
uv run bourbon-process retry --report processed/error_report_<bottle_id>.json

# See every setting and where it comes from
uv run bourbon-process config show
```

Photos of one bottle are grouped by EXIF capture time (taken at most 12 s
apart, at most 3 per bottle). Tune with `--group-gap-seconds` and
`--group-max-size`. If the vision model says a group mixes bottles, it is
split and re-read (cap: `--split-budget-ratio`, 0 = off).

## Web Lookups (on by default)

- Works with no API key: DuckDuckGo. With `BRAVE_API_KEY` or `OLLAMA_API_KEY`
  set, or a SearXNG URL in the config, that provider is used instead.
- Polite: obeys robots.txt, waits between requests to a site, sends no
  cookies. Results are cached in `~/.cache/bourbon-process/web`;
  `--refresh-web` ignores the cache.
- Social media, forums and auction sites are never used.
- Every website consulted is listed in the bottle's **Web Research Log**,
  with what it was used for or why it was skipped.
- A failed lookup never fails a bottle. `--no-web` turns lookups off.
- `-v` also fetches a vendor product photo; it is for your reference only,
  **not licensed for publication**.

## Fact Check and Approval

- Each field is `supported` (sources agree), `user` (you said so),
  `conflict` (sources disagree) or `unknown` (with the reason).
- A price on the label or a shop tag is retail, **never MSRP**.
- Conflicts are listed under **Needs Review** in the bottle's Markdown page.
  Settle them with `[C]onflicts` in interactive mode, or with a corrections
  file.
- The page starts with **Publish-ready** or **NOT publish-ready: <reasons>**.
  It is ready only when the fact check verified it, there are no open
  conflicts or low-confidence values, the core fields are supported, a
  verified description exists and is current, and you approved it.
- Approve with `[A]ccept` in `--interactive`, or with
  `feedback --approve <bottle_id>`.
- Interactive keys: `[A]ccept` `[R]eject` `[E]dit` `[C]onflicts` `[N]ext`
  (keep, not approved) `[Q]uit`.
- Expect about 1-2 minutes per bottle for the fact check with `gpt-oss:120b`.

## Output Structure

```
processed/
  ├── run_<run_id>_settings.json    # Settings of the run
  ├── bottle_name_1/
  │   ├── bottle_name_1.md          # Markdown page (status, facts, sources, web research log)
  │   ├── bottle_name_1.json        # Full record (schema_version 2)
  │   ├── bottle_name_1.txt         # OCR text + web page count
  │   ├── evidence/W1.txt           # Web excerpts the fact check read
  │   └── images/                   # Copies of the photos (originals untouched)
  ├── batch_<run>_001_preview.md    # Batch preview report
  ├── batch_<run>_001.json          # Batch data for `feedback`
  └── error_report_<bottle_id>.json # One per failed bottle, for `retry`
```

See [../../examples/processed_sample/](../../examples/processed_sample/) for a
complete example.

Exit codes: 0 ok, 1 bad setting / Tesseract missing / some bottles failed,
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
ollama pull gpt-oss:120b           # or use --text-model gpt-oss:20b / --no-verify
```

**"Not enough disk space to pull"**: free some space, lower the margin with
`--min-free-gb`, or use smaller models (`-m gemma4:26b`,
`--text-model gpt-oss:20b`).

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
- **Tell it what you know**: a `--hints` file records values you checked on the bottle, with their source (see `examples/hints.example.toml`)
- **Review output**: Check the .md files after the first batch

## Next Steps

For configuration, every option, and troubleshooting, see the
[README](../../README.md) and [../detailed/README.md](../detailed/README.md).
