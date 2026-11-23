# Bourbon Image Processor

`bourbon-process` reads photos of bourbon bottles with a local vision model
served by [Ollama](https://ollama.com), and writes one folder per bottle with
Markdown and JSON metadata (name, distillery, release, proof, ABV, age, batch,
tasting notes). Everything runs locally; no cloud APIs are called.

It runs on macOS and Linux. Ollama is the only model backend.

## Features

- **One local vision model** (Ollama, default `qwen3.6:35b-a3b`) reads each bottle's label photos
- **Bottle grouping by capture time**: photos of one bottle taken seconds apart are analysed together
- **Ollama server lifecycle**: reuses a running Ollama or starts `ollama serve` itself, warms the model, and never downloads a model without `--pull`
- **Disk check before downloads**: `--pull` compares the model's download size with free space on the Ollama model store
- **Usage report**: model calls, durations and per-stage totals are logged at the end of every run
- Batch processing with Markdown preview reports, interactive review, and `feedback` / `retry` commands

## Documentation

- [Quick Start Guide (5 minutes)](docs/adhd/README.md)
- [Detailed Documentation](docs/detailed/README.md): troubleshooting, batch processing and advanced topics

## Pipeline stages and models

Only one stage calls an AI model: vision analysis, with the `-m/--model`
vision model. There is no automatic fallback to another model. The model
selector also lists the models planned for the other stages; `--debug` prints
that table and marks every stage except vision analysis "not used by the
pipeline".

| Stage | Status | What it uses |
|-------|--------|--------------|
| OCR / text extraction | Implemented | Tesseract OCR (local, no model). The text is sent to the vision model as a hint only. |
| Vision analysis | Implemented | `--model` (default `$BOURBON_MODEL`, else `qwen3.6:35b-a3b`). The only model that is called. |
| Image grouping | Implemented | EXIF capture time (see [How photos are grouped](#how-photos-are-grouped)), checked by the vision model's same-bottle answer (mixed groups are split). No visual similarity matching. |
| Quantity detection | Implemented | File hashes (exact duplicate photos of a bottle). |
| Error classification | Implemented | Rule-based; drives the error reports that `retry` reads. |
| Preview generation | Implemented | Markdown templates (no model). |
| Metadata generation (text model) | Not implemented | The selector lists `gpt-oss:120b`, but nothing calls it and it is never required or pulled. |
| Web research / vendor images | Not implemented | The selector lists DuckDuckGo; `--include-vendor-images` is accepted but has no effect yet. |

## Installation

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), Tesseract OCR and
Ollama.

### macOS (Homebrew)

```bash
brew install uv tesseract tesseract-lang
brew install ollama           # CLI + server; or install the desktop app from https://ollama.com/download
brew services start ollama    # or run `ollama serve` in a terminal, or open the desktop app
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1   # libgl1 is needed by OpenCV
curl -LsSf https://astral.sh/uv/install.sh | sh                  # uv
curl -fsSL https://ollama.com/install.sh | sh                    # Ollama + systemd service
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y tesseract tesseract-langpack-eng mesa-libGL
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh
```

The Ollama install script sets up an `ollama` systemd service that runs as the
`ollama` user and keeps its models in `/usr/share/ollama/.ollama/models`:

```bash
sudo systemctl enable --now ollama   # start now and at boot
systemctl status ollama
```

If the service is installed but stopped, `bourbon-process` will not start its
own `ollama serve` (that would use a different model store). Start the service,
or set `OLLAMA_MODELS` if you run Ollama as your own user.

### Python package

From the repository root:

```bash
uv sync                    # creates .venv and installs the `bourbon-process` command
uv run bourbon-process --help
```

`uv sync --extra dev` also installs the test and lint tools.

### Vision model

Only one model is needed: the vision model passed with `-m/--model`
(default `qwen3.6:35b-a3b`). Install it yourself:

```bash
ollama pull qwen3.6:35b-a3b
```

or let the tool download it after a disk-space check by adding `--pull` to a
run. Nothing is ever downloaded without `--pull`.

Vision-capable models that have been used with this tool:

| Model | Approx. size | Notes |
|-------|--------------|-------|
| `qwen3.6:35b-a3b` | 23 GB | Default |
| `qwen3.8:27b` | 18 GB | |
| `gemma4:26b` | 16 GB | Smallest of the three |

Any other Ollama model whose capabilities include `vision` can be passed with
`-m`. Text-only models (for example `gpt-oss:120b`) are rejected in pre-flight.
Models that Ollama lists but can no longer run (for example the old
`llama3.2-vision` builds on current Ollama versions) fail on the first request
and stop the run with exit code 2.

## Usage

```bash
# Process the photos in ./photos; output goes to ./processed
uv run bourbon-process process -f ./photos -o ./processed

# The `process` subcommand can be left out
uv run bourbon-process -f ./photos

# Check inputs, the server and the model without processing anything
uv run bourbon-process process -f ./photos --dry-run

# Use a different vision model, downloading it if it is missing
uv run bourbon-process process -f ./photos -m gemma4:26b --pull

# Print the model selector stage report and debug logging
uv run bourbon-process --debug process -f ./photos
```

Running `bourbon-process` with no arguments prints the help and exits with 1.
(Inside the activated virtualenv, `uv run` can be left out.)

### Subcommands

| Command | Purpose |
|---------|---------|
| `process` | Scan a folder, group photos into bottles, analyse each bottle and save the results |
| `feedback` | Apply corrections (`--corrections file.json` or `-i`) to a saved batch (`--batch <id>`, default the most recent) |
| `retry` | Re-analyse a failed bottle from its `error_report_<bottle_id>.json` (`--report`) |

`retry` takes the same model and server options as `process` (`-m`, `-u`,
`--request-timeout`, `--keep-server`, `--no-server-autostart`, ...) but not
`--pull`.

### `process` options

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --image-folder` | `.` | Folder with the bottle photos (scanned recursively) |
| `-o, --output-dir` | `processed` | Output folder; must not be the photo folder itself, and is best kept outside it |
| `-b, --batch-size` | `5` | Bottles per batch, or `all` (no preview reports) |
| `--group-gap-seconds` | `12` | Most seconds between consecutive photos of one bottle |
| `--group-max-size` | `3` | Most photos per bottle (0 = no limit) |
| `--split-budget-ratio` | `0.35` | Extra analyses per group the run may spend splitting groups the model reports as mixed (at least 4 per run; 0 = off, max 2). An analysis is one vision request (retried once if unusable) plus OCR. See [How photos are grouped](#how-photos-are-grouped) |
| `--lookahead` | none | **Deprecated, ignored** (logs a warning). Still accepted (0-10) so old scripts keep working; grouping by capture time already keeps a bottle in one batch |
| `-m, --model` | `$BOURBON_MODEL`, else `qwen3.6:35b-a3b` | Vision model (Ollama tag); must be vision-capable |
| `-u, --ollama-url` | `http://localhost:11434` | Ollama URL |
| `--server-backend` | `$BOURBON_SERVER_BACKEND`, else `ollama` | Model server backend (only `ollama`) |
| `--server-start-timeout` | `30` | Seconds to wait for a server this run starts |
| `--request-timeout` (`--vision-timeout`) | `600` | Per-request model timeout in seconds, including model load |
| `--keep-server` | off | Leave a server started by this run running |
| `--no-server-autostart` | off | Never start a server; fail if none is running |
| `--pull` (`--pull-missing`) | off | Download the vision model if missing, after a disk-space check |
| `--min-free-gb` | `5` | GiB to keep free on the model store after a `--pull` |
| `-i, --interactive` | off | Review and correct each batch before it is saved |
| `--dry-run` | off | Check everything without starting a server, pulling, loading or analysing |
| `--debug` | off | Debug logging and the model selector stage report |
| `--concurrency` | `3` | Validated (1-10) but has no effect: bottles are processed one at a time |
| `-v, --include-vendor-images` | off | Accepted but has no effect yet |
| `-r, --allow-retry` / `--no-retry` | on | Accepted but has no effect yet |

### Model server (Ollama)

```bash
# Ollama at http://localhost:11434 (the default)
uv run bourbon-process process -f photos/

# Ollama on another machine
uv run bourbon-process process -f photos/ -u http://gpu-box:11434
```

- A server that is already running is reused and is **never stopped** by this
  tool. That covers the desktop app, `brew services`, the Linux systemd
  service and a manual `ollama serve`.
- If nothing is running on a local URL (`localhost`, `127.0.0.1`, `::1`), the
  tool starts `ollama serve` and stops it when the run ends. Pass
  `--keep-server` to leave it running, or `--no-server-autostart` to never
  start one. A remote URL is never started.
- Logs of servers started by the tool go to `$BOURBON_LOG_DIR`, else
  `$XDG_CACHE_HOME/bourbon-processor/logs` (default
  `~/.cache/bourbon-processor/logs`).
- Before processing, the vision model is checked: it must be installed (or
  pulled with `--pull`) and report the `vision` capability. It is then loaded
  into memory; a failed warm-up is only a warning.
- A missing model makes pre-flight fail with exit code 2. The error shows the
  download size, the installed vision-capable models, and the exact
  `ollama pull <model>` command.

### `--pull` and the disk check

Models are **never downloaded** unless you pass `--pull`. With `--pull`:

1. The download size is read from the registry manifest. Blobs that are
   already complete in the local store are not counted.
2. Free space is measured on the Ollama model store: `$OLLAMA_MODELS` if set,
   else `~/.ollama/models`, or on Linux the systemd service store
   `/usr/share/ollama/.ollama/models` when that exists and `~/.ollama/models`
   does not.
3. The pull is refused unless the download plus a safety margin fits. The
   margin is the larger of `--min-free-gb` (default 5 GiB) and 10% of the
   download. It is also refused if the size cannot be determined, or if the
   model is not in the registry.
4. During the download the pull is aborted if free space drops below
   `--min-free-gb`. Ollama keeps partially downloaded blobs and resumes them
   on the next pull; this tool never deletes anything.

For a remote server (`-u http://other-host:11434`) the free space cannot be
measured; the pull goes ahead with a warning that states the download size.

`--dry-run --pull` reports what would be downloaded and whether it fits,
without downloading.

### `--dry-run`

A dry run only checks: arguments, the photo folder, that the output folder
could be created, Tesseract, and the model server. It probes the server but
never starts it, never pulls or loads a model, writes nothing to disk, and
lists the bottle groups it found. It still fails (exit 2) when a real run could
not start the server, for example with `--no-server-autostart` or a stopped
Linux systemd service.

### How photos are grouped

Photos are grouped into bottles by their EXIF capture time, never by file
name (consecutive names like `IMG_4653`/`IMG_4654` are usually different
bottles):

- Photos from one camera taken at most `--group-gap-seconds` apart (default
  12) belong to one bottle.
- A run of more than `--group-max-size` photos (default 3; 0 = no limit) is
  split at its largest time gap. Raise it if you always shoot four or more
  angles of a bottle.
- Photos without EXIF capture time are single-photo bottles, except names
  like `IMG_1_front.jpg` / `IMG_1_back.jpg`, which pair up. File modification
  times are used only when other photos prove they are the original camera
  times (copies and downloads rewrite them).
- Exact duplicates join their original; burst retakes taken a few seconds
  apart are kept but not sent to the vision model twice.
- `--dry-run` lists the groups without calling the model.

**Same-bottle check.** For a group of several photos the vision model also
says whether they all show one product. If it says no:

1. The group is split at its largest time gap and each half is read again
   (2 extra vision calls). Duplicates and burst retakes stay with their
   original.
2. If the halves disagree on a field both of them show (name, release, batch,
   proof within 0.5, age), the split is kept: each part is saved as its own
   bottle with only its own photos, and a part the model still calls mixed is
   split again, down to single photos.
3. Otherwise it was a false alarm and the group stays one bottle (its JSON
   `same_bottle` is cleared to `null`). A field or number seen on only one
   half (say the batch on the back label, or "125" after "1792 Full Proof")
   never counts as a difference, so a front/back pair is not torn apart. A
   half the model still calls mixed is checked again; if it turns out to be
   one bottle that agrees with the other half, the whole group stays one.
   The downsides: two bottles of the same product that show no different
   field (e.g. two Elmer T. Lee bottles) stay merged, and a name that is a
   shorter form of the other ("Stagg" vs "Stagg Jr", "Weller" vs "Weller
   Special Reserve") matches unless proof, batch, age or release differ. The
   bottle gets a review note saying so.
4. Extra analyses are capped for the whole run at `--split-budget-ratio` x
   number of groups (default 0.35, at least 4). The cap counts analyses, not
   model requests: each may retry an unusable answer once and re-runs OCR
   on its photos. Once used up, a mixed group is kept whole with a review
   note. `--split-budget-ratio 0` turns the check off.

**Possibly the same bottle.** When a bottle's first photo was taken at most
60 s after the previous bottle's last photo and both read the same label
(nothing read on both differs), it gets the note "Possibly the same bottle
as <id> (adjacent photos, identical label)". The bottles are never merged
automatically, since sibling bottles with identical labels look the same.
This catches a box and bottle split by `--group-max-size`. If at least a
quarter of these notes involve a group of exactly `--group-max-size`
photos, the run ends with a warning suggesting `--group-max-size N+1`.

Split and false-alarm notes appear as **Warning** lines in the bottle's
Markdown, as **Review** lines in the batch preview, and in the JSON
`metadata.warnings`. The run ends with a one-line summary of the check.

Supported formats: `.bmp .heic .jpeg .jpg .png .tiff .webp`. HEIC needs the
optional `pillow-heif` package (`uv pip install pillow-heif`). Hidden folders
and earlier output folders (marked with a `.bourbon_output` file) are skipped.

### How bottles are read

- Each bottle gets one vision call with up to 4 of its photos (receipts are
  skipped unless nothing else exists). Photos are rotated using their EXIF
  orientation and downscaled to 1280 px. Tesseract OCR text goes along as a
  hint only.
- The model is asked for JSON (name, distillery, release, proof, ABV, age,
  batch, tasting notes). It must report only what is printed on the label.
  MSRP is filled only when a price is visible in a photo; otherwise it is
  `null` in the JSON and "Unknown" in the Markdown (never $0.00), and the total
  value is unknown too.
- With one photo the model is told to ignore bottles in the background; the
  "photos show different products" warning is only raised for groups of
  several photos.
- A vision failure is reported as a failure, never as an "Unknown Bottle".
  This covers timeouts, empty or unparseable answers, and answers without a
  name, distillery or release. The bottle gets an error report and the run
  continues.
- A model that is missing, cannot read images, or is incompatible with your
  Ollama version stops the run with exit code 2, as does an unreachable
  server.

## Output

```
processed/
├── .bourbon_output                         # marks the folder as output (skipped by later scans)
├── <bottle_name>/
│   ├── <bottle_name>.md                    # Markdown page
│   ├── <bottle_name>.json                  # full metadata
│   ├── <bottle_name>.txt                   # OCR text
│   └── images/                             # copies of the source photos (originals are only read)
├── batch_<run>_<NNN>_preview.md            # batch preview report (not with --batch-size all)
├── batch_<run>_<NNN>.json                  # batch data read by `feedback`
└── error_report_<bottle_id>.json           # one per failed bottle, read by `retry`
```

Existing folders are never overwritten: a second bottle (or a re-run) with
the same name is saved as `<bottle_name>_2`, `_3`, and so on.

At the end of a run the usage summary lists the model calls, total duration
and a breakdown by stage and by model.

## Exit codes

`process`:

| Code | Meaning |
|------|---------|
| 0 | Success (also a successful `--dry-run`) |
| 1 | Invalid argument, output folder equal to the photo folder, Tesseract missing, some bottles failed, or no arguments given |
| 2 | Model server or model unavailable (not running and cannot be started, model missing, not vision-capable, refused pull, incompatible model, invalid `--ollama-url` port) |
| 3 | Photo folder missing or contains no supported images |
| 4 | Every bottle failed |
| 5 | Output folder not writable |
| 128+N | Stopped by signal N (SIGTERM, SIGHUP, SIGQUIT); a server started by the run is stopped first |

`feedback`: 0 success, 1 error, 2 batch not found.
`retry`: 0 success, 1 error (including model server unavailable), 2 report not found.

## Environment variables

| Variable | Effect |
|----------|--------|
| `BOURBON_MODEL` | Default vision model when `-m` is not given |
| `BOURBON_SERVER_BACKEND` | Default `--server-backend` (only `ollama`) |
| `BOURBON_LOG_DIR` | Log folder for servers started by the tool |
| `OLLAMA_BIN` | Executable used to start `ollama serve` (default `ollama` on `PATH`) |
| `OLLAMA_MODELS` | Ollama model store used for the disk check (same meaning as for Ollama itself) |

## Troubleshooting

See also [Detailed Documentation](docs/detailed/README.md#troubleshooting).

**Model server unavailable (exit 2)**

```bash
curl http://localhost:11434/api/tags        # is Ollama answering?
ollama serve                                # start it by hand (macOS or Linux)
sudo systemctl start ollama                 # Linux systemd service
```

**Model not installed (exit 2)**

```bash
ollama list                                  # installed models
ollama pull qwen3.6:35b-a3b                  # or re-run with --pull
```

**Not enough disk space to pull**: free space on the model store, lower the
margin with `--min-free-gb`, or choose a smaller model such as `gemma4:26b`.

**Timeouts on the first bottle**: a cold model load can take minutes; raise
`--request-timeout`.

**Out of memory**: use a smaller vision model, for example `-m gemma4:26b`.

**Tesseract OCR not found (exit 1)**: install it (`brew install tesseract`,
`sudo apt-get install tesseract-ocr`, or `sudo dnf install tesseract`).

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest                    # hermetic: network access is blocked in tests
uv run --extra dev pytest --cov=src --cov-report=html
```

Live tests are skipped unless enabled. They never pull models:

```bash
BOURBON_LIVE_TESTS=1 uv run --extra dev pytest tests/integration/test_model_server_live.py
BOURBON_LIVE_TESTS=1 BOURBON_LIVE_VISION=1 uv run --extra dev pytest tests/integration/test_vision_live.py
```

The legacy single-file `scraper.py` is superseded by `bourbon-process`.

## License

Personal bourbon inventory project.
