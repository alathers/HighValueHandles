# Bourbon Image Processor - Complete User Guide

This guide explains how a run works and how to review its results. The
[README](../../README.md) is the reference for installation, every option,
the configuration file, environment variables and exit codes.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [CLI Reference](#cli-reference)
5. [Processing Modes](#processing-modes)
6. [Output Format](#output-format)
7. [Batch Processing](#batch-processing)
8. [Web Lookups and the Fact Check](#web-lookups-and-the-fact-check)
9. [Interactive Mode](#interactive-mode)
10. [Feedback and Approval](#feedback-and-approval)
11. [Error Handling](#error-handling)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Topics](#advanced-topics)

---

## Overview

`bourbon-process` is a local CLI tool that reads bourbon bottle photos and
writes one documented record per bottle. It:

- groups photos into bottles by EXIF capture time;
- reads each bottle's label with a local vision model (Ollama) and Tesseract OCR;
- looks each bottle up on the web (on by default, no API key needed) and logs every site consulted;
- fact-checks every field with a local text model: the model proposes quoted claims, Python verifies each quote and decides;
- writes Markdown and JSON with the sources, confidence and status of every value, and a publish gate that needs a person's approval;
- retries model requests that may pass on a second try, and writes an error report for each bottle that still fails.

Accuracy and provenance come first: a value that cannot be verified is
"Unknown" with the reason, and a disagreement between sources is a conflict
for a person to settle.

### Stages and Models

| Stage | Uses | Notes |
|-------|------|-------|
| OCR | Tesseract | Hint text for the vision model; evidence (`O1`...) for the fact check |
| Vision analysis | `-m` (default `qwen3.6:35b-a3b`) | Reads the label; also answers whether a group's photos show one product |
| Grouping | EXIF capture time + the vision same-bottle answer | `--group-gap-seconds`, `--group-max-size`, `--split-budget-ratio` |
| Quantity | File hashes | Byte-identical copies of a photo count as more bottles |
| Web research | Search provider + polite HTTP fetches | No model (`-v` also uses the vision model to compare a vendor photo) |
| Fact check | `--text-model` (default `gpt-oss:120b`) | Extracts quoted claims and writes a cited description; Python verifies |
| Reconciliation, confidence, publish gate | Python rules | No model |
| Preview and output files | Templates | No model |

There is no automatic fallback model. Ollama is the only backend: a running
Ollama is reused and never stopped; otherwise `ollama serve` is started for
the run and stopped afterwards.

### Usage Tracking

At the end of a run the summary lists the bottles processed and failed, the
web search provider, the web research and fact-check status counts, how many
bottles are publish-ready, and the model calls and durations by stage and by
model.

---

## Installation

Follow [README.md#installation](../../README.md#installation). It covers
macOS (Homebrew) and Linux (Debian/Ubuntu, Fedora/RHEL): `uv`, Tesseract,
Ollama, and `libgl1` (Debian/Ubuntu) or `mesa-libGL` (Fedora) for OpenCV.

Requirements:

- Python 3.11+, [uv](https://docs.astral.sh/uv/), Tesseract OCR and Ollama
  ([ollama.com](https://ollama.com)), on macOS or Linux.
- Vision model `qwen3.6:35b-a3b`: about 23 GB of disk and 32 GB+ of RAM.
- Fact-check model `gpt-oss:120b`: about 65 GB of disk and about 70 GB of
  free unified (Apple Silicon) or GPU memory. On smaller machines use
  `--text-model gpt-oss:20b` (about 14 GB), or `--no-verify` to skip the
  fact check.

```bash
uv sync                               # installs the bourbon-process command
ollama pull qwen3.6:35b-a3b           # or add --pull to a run
ollama pull gpt-oss:120b
uv run bourbon-process --help
uv run bourbon-process process -f photos/ --dry-run   # checks Tesseract, Ollama and both models
```

---

## Configuration

Settings come from built-in defaults, then
`${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml`, then
`./bourbon.toml`, then `BOURBON_*` environment variables, then flags (each
layer overrides the one before). `--config PATH` reads only that file;
`--no-config` reads none.

```bash
uv run bourbon-process config template > bourbon.toml   # commented template of every key
uv run bourbon-process config show                      # each value and where it came from
```

Some settings exist only in the file, for example `text.think_compose`,
`text.num_ctx`, `web.max_pages`, `web.deny_domains` and `web.allow_domains`.
API keys (`BRAVE_API_KEY`, `OLLAMA_API_KEY`) go in environment variables
only. See [README.md#configuration](../../README.md#configuration) and
[README.md#environment-variables](../../README.md#environment-variables).

---

## CLI Reference

```bash
bourbon-process [--debug] [--config PATH | --no-config] {process,feedback,retry,config} [options]
```

`process` can be left out: `bourbon-process -f photos/` runs `process`.
With no arguments at all the help is printed (exit 1). There is no
`--version` flag; the version is recorded in each run's
`run_<run_id>_settings.json`.

### `process`

The full option table is in [README.md#process-options](../../README.md#process-options). By group:

- Input and output: `-f/--image-folder`, `-o/--output-dir`.
- Batching and grouping: `-b/--batch-size N|all`, `--concurrency N`,
  `--group-gap-seconds`, `--group-max-size`, `--split-budget-ratio`.
- Vision model and server: `-m/--model`, `-u/--ollama-url`,
  `--server-start-timeout`, `--request-timeout`, `--keep-server`,
  `--no-server-autostart`, `--keep-models-loaded`, `--pull`, `--min-free-gb`.
- Fact check: `--verify/--no-verify`, `--text-model TAG|none`,
  `--text-think low|medium|high`, `--text-timeout`.
- Web: `--web/--no-web`, `--web-provider auto|duckduckgo|brave|ollama|searxng|none`,
  `--web-cache-dir`, `--refresh-web`, `-v/--include-vendor-images`.
- Review and retries: `-i/--interactive`, `--hints PATH`, `-r/--allow-retry`,
  `--no-retry`, `--retry-attempts N`.
- Checks: `--dry-run`, `--debug`.

### `feedback`

```bash
bourbon-process feedback [-o DIR] [--batch ID] [--corrections FILE | -i] [--approve BOTTLE_ID ...]
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir` | Folder with the batch files (default `processed`) |
| `--batch ID` | The batch ID in the file name: `20261122_193000_001` for `batch_20261122_193000_001_preview.md`. Default: the most recent batch |
| `--corrections FILE` | JSON corrections file (see [Feedback and Approval](#feedback-and-approval)) |
| `-i, --interactive` | Review the batch's bottles interactively (when no `--corrections` is given) |
| `--approve BOTTLE_ID` | Approve a bottle for publishing; repeat for more bottles |

`feedback` has no `-b` option and calls no model, server or web site.

### `retry`

```bash
bourbon-process retry --report processed/error_report_<bottle_id>.json [-o DIR] [--hints PATH] [options]
```

`retry` takes the model, server, retry, fact-check and web options of
`process` (`-m`, `-u`, `--server-start-timeout`, `--request-timeout`,
`--keep-server`, `--no-server-autostart`, `--keep-models-loaded`, `-r`,
`--no-retry`, `--retry-attempts`, `--verify`/`--no-verify`, `--text-model`,
`--text-think`, `--text-timeout`, `--web`/`--no-web`, `--web-provider`,
`--web-cache-dir`, `--refresh-web`, `-v`), but not `--pull`.

### `config`

```bash
bourbon-process config show [--json]
bourbon-process config template
```

---

## Processing Modes

```bash
# Current folder, output to processed/ (the output folder is skipped when scanning)
bourbon-process process

# A specific folder
bourbon-process process -f /path/to/bottle/photos

# Everything in one batch: one preview, one model swap
bourbon-process process -f photos/ -b all

# Smaller batches for careful review
bourbon-process process -f photos/ -b 2 -i

# Check settings, photos, grouping, Tesseract, server and models; process nothing
bourbon-process process -f photos/ --dry-run
```

A dry run writes nothing, starts and pulls nothing, loads no model and makes
no web request. It lists the bottle groups and logs the active stages, the
text model, the chosen search provider and the text model's memory check.

---

## Output Format

### Directory Structure

```
processed/
├── .bourbon_output                         # marks the folder as output (skipped by later scans)
├── run_<run_id>_settings.json              # effective settings, tool version and models of the run (no API keys)
├── <bottle_name>/
│   ├── <bottle_name>.md                    # Markdown page: status, facts, prices, sources, web research log
│   ├── <bottle_name>.json                  # full record, schema_version 2
│   ├── <bottle_name>.txt                   # OCR text, then "web: N pages consulted, M cited"
│   ├── evidence/
│   │   └── W1.txt ...                      # each web excerpt exactly as the fact check saw it
│   └── images/                             # copies of the source photos (originals are only read)
│       └── vendor_ref.<ext>                # -v only: vendor reference image, not licensed for publication
├── batch_<run_id>_<NNN>_preview.md         # batch preview report
├── batch_<run_id>_<NNN>.json               # batch data read by `feedback`
└── error_report_<bottle_id>.json           # one per failed or rejected bottle, read by `retry`
```

A complete generated example is in
[examples/processed_sample/](../../examples/processed_sample/).

### Markdown Page

In this order:

1. Title, then **Publish-ready** or **NOT publish-ready: <blockers>**.
2. **Needs Review**: each conflict with its candidate values and quotes
   (resolved ones say which value won).
3. **Facts**: Field | Value | Confidence | Sources. Unknown rows give the reason.
4. **Prices**: MSRP (`$X` with the page, quote and date; or user-stated; or
   `Unknown: <reason>`), then retail prices. A label price is shown as
   "Price visible on label (retail, not MSRP)".
5. **Description (draft)** with `[n]` markers for the facts each sentence
   relies on, or the reason there is none.
6. **Sources**: numbered; label, OCR, web pages (with class, domain, date and
   the excerpt file) and user statements, with the quotes used.
7. **Verification**: model, think level, prompt version, accepted and
   rejected counts, duration.
8. **References**: the web pages, cited ones first.
9. **Web Research Log**: provider, queries, and one row per site consulted
   with its class, date, outcome and what it was used for or why it was
   skipped. Shown also when the lookup was skipped or failed.
10. Quantity, images (a vendor image is marked "vendor reference, not for
    publication"), corrections, and processing details.

Text from web pages is escaped, and only fetched URLs become links.

### JSON File

The JSON is `Bottle.to_dict()` plus the save details. An excerpt of the
example bottle (`facts.fields` shows two of its twelve fields, and
`web_research.pages` two of four):

```json
{
  "schema_version": 2,
  "bottle_id": "IMG_0001",
  "run_id": "20261123_120000",
  "name": "Example Creek Single Barrel",
  "distillery": "Example Creek Distillery",
  "specific_release": "Single Barrel",
  "proof": "100 Proof",
  "abv": null,
  "age": null,
  "batch_number": "C923",
  "tasting_notes": null,
  "msrp": null,
  "label_price": "64.99",
  "total_value": null,
  "review_required": false,
  "quantity": {"count": 1, "price": null, "source": "extracted"},
  "facts": {
    "fields": {
      "proof": {
        "value": 100.0,
        "status": "supported",
        "confidence": "high",
        "confidence_basis": "label(high)+ocr",
        "citations": [
          {"source_id": "L1", "quote": "100 Proof", "quote_kind": "reading", "derived": false},
          {"source_id": "O1", "quote": "100 PROOF", "quote_kind": "verbatim", "derived": false}
        ],
        "candidates": ["..."],
        "reason": null
      },
      "msrp": {
        "value": null,
        "status": "unknown",
        "confidence": null,
        "confidence_basis": null,
        "citations": [],
        "candidates": [],
        "reason": "price on label ($64.99) recorded as retail, not MSRP; 2 page(s) fetched; 1 retail price(s) found (W2 $69.99); no source stated an MSRP"
      }
    },
    "sources": ["..."],
    "conflicts": ["..."],
    "prices": [
      {"kind": "retail", "amount": "64.99", "currency": "USD",
       "citation": {"source_id": "L1", "quote": "price visible on label", "quote_kind": "reading", "derived": false},
       "confidence": null},
      {"kind": "retail", "amount": "69.99", "currency": "USD",
       "citation": {"source_id": "W2", "quote": "Our price $69.99", "quote_kind": "verbatim", "derived": false},
       "confidence": null}
    ],
    "verification": {"status": "verified", "text_model": "gpt-oss:120b", "think": "high",
                     "prompt_version": "facts-1", "duration_s": 74.2, "rejected": ["..."]},
    "copy": {"...": "..."},
    "publish": {"ready": false, "blockers": ["not approved by a reviewer"], "group_warnings": []},
    "review": {"approved": false, "by": null, "at": null}
  },
  "web_research": {
    "status": "ok",
    "reason": null,
    "provider": "duckduckgo",
    "queries": ["\"Example Creek Single Barrel\" Example Creek Distillery c923 MSRP",
                "\"Example Creek Single Barrel\" Example Creek Distillery Single Barrel"],
    "searched_at": "2026-11-23T12:00:00+00:00",
    "pages": [
      {"url": "https://www.examplecreek.test/single-barrel", "title": "Single Barrel | Example Creek",
       "domain": "examplecreek.test", "source_class": "official", "retrieved_at": "2026-11-23T12:00:04+00:00",
       "outcome": "cited", "cited_for": ["distillery"], "reason": null, "from_cache": false},
      {"url": "https://forum.test/t/example-creek", "title": "Example Creek thread",
       "domain": "forum.test", "source_class": null, "retrieved_at": null,
       "outcome": "skipped", "cited_for": [], "reason": "deny-listed", "from_cache": false}
    ],
    "evidence": ["..."],
    "vendor_image": null
  },
  "reference_links": ["..."],
  "corrections": ["..."],
  "vendor_image": null,
  "metadata": {"...": "..."},
  "extracted_at": "2026-11-23T12:00:00+00:00",
  "model_used": "qwen3.6:35b-a3b",
  "confidence": "HIGH",
  "source_images": ["/path/to/photos/IMG_0001.jpg"],
  "images": ["images/IMG_0001.jpg"],
  "missing_images": []
}
```

Flat fields are `null` when unknown; a price is never `0`. `feedback` and
`retry` read only `schema_version` 2 files.

### Raw OCR Text

The `.txt` file holds the OCR text of every photo of the bottle, then the
line `web: N pages consulted, M cited`.

---

## Batch Processing

### How Batching Works

1. **Scan**: read all supported images from the input folder (the output
   folder, earlier outputs and hidden folders are skipped).
2. **Group**: group photos into bottles by EXIF capture time.
3. **Slice**: divide the bottle groups into batches (`-b`, default 5).
4. **Analyse**: OCR and the vision request for each bottle in the batch
   (`--concurrency` at once), with the same-bottle check.
5. **Quantity and hints**: count duplicate photos; apply `--hints` values.
6. **Enrich**: web research for the batch, unload the vision model, fact
   check each bottle, unload the text model.
7. **Review**: with `-i`, approve, reject, edit or settle conflicts.
8. **Preview**: write the batch preview report and batch JSON.
9. **Save**: write each bottle's folder; failed and rejected bottles get an
   error report.
10. **Repeat** with the next batch.

Because the models are swapped in every batch, `-b all` loads each model
only once per run.

### Grouping by Capture Time

Photos taken by one camera at most `--group-gap-seconds` apart (default 12)
are one bottle; runs longer than `--group-max-size` (default 3) are split at
their largest time gap. File names are never compared (consecutive numbers
are usually different bottles). Photos without EXIF time are single-photo
bottles unless named like `IMG_1_front` / `IMG_1_back`.

### Same-Bottle Check

Grouping finishes before batching, so a bottle never spans two batches.

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
saved as its own bottle folder with only its photos. A split or mixed-group
warning blocks publishing until a person has looked.

Bottles whose photos are at most 60 s apart and whose labels agree get a
"Possibly the same bottle as <id>" review note (never merged). When at least
25% of those notes involve a group of exactly `--group-max-size` photos, the
run suggests `--group-max-size N+1`.

Extra analyses are capped for the whole run, shared by all batches. The cap
counts analyses (OCR plus a vision request); retries of a request do not
count against it:

```bash
bourbon-process process --split-budget-ratio 0.35   # default: up to ceil(0.35 x groups) extra analyses (min 4)
bourbon-process process --split-budget-ratio 1      # check more mixed groups
bourbon-process process --split-budget-ratio 0      # turn the check off
```

### Preview Reports

Written for every batch as `processed/batch_<run_id>_<NNN>_preview.md`, with
the batch data in `batch_<run_id>_<NNN>.json`. The report has a summary
table (name, quantity, images, extraction confidence, Ready Y/N, open
Conflicts, Unknown fields, Verification status, MSRP with its source), then
per bottle its ID, fields, label price (retail), publish status and
blockers, web count, review notes and an OCR preview, then the feedback
commands for that batch.

---

## Web Lookups and the Fact Check

The README has the full rules:
[Web lookups](../../README.md#web-lookups) and
[Fact check and provenance](../../README.md#fact-check-and-provenance). In
short:

- **Search**: `auto` uses Brave (`BRAVE_API_KEY`), else Ollama web search
  (`OLLAMA_API_KEY`), else SearXNG (`web.searxng_url`), else DuckDuckGo
  (no key). Queries are built from the label reading only.
- **Fetching**: robots.txt is honoured, at least 2 s between requests to a
  host, a generic User-Agent, no cookies, a cache in
  `${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web` (`--refresh-web`
  bypasses it). Social, forum, video and auction sites are never used;
  `web.deny_domains` and `web.allow_domains` narrow it further.
- **Logged**: every URL considered is in the bottle's `web_research.pages`
  and its Web Research Log, with its outcome and reason.
- **Never fatal**: a failed lookup is recorded and the bottle continues.
- **Fact check**: the text model quotes; Python verifies each quote against
  the source text and each value against the quote, reconciles the fields
  (`supported`, `user`, `conflict`, `unknown`), assigns confidence by fixed
  rules and computes the publish gate. A label or shop price is never an
  MSRP.
- **Time**: about 1-2 minutes per bottle for the fact check with
  `gpt-oss:120b`.

---

## Interactive Mode

```bash
bourbon-process process -f photos/ --interactive
bourbon-process feedback --interactive            # the most recent saved batch
```

For each bottle the terminal shows its facts table, open conflicts and
publish blockers, then asks for an action:

| Key | Action |
|-----|--------|
| `A` | Accept: approve the bottle for publishing (the other publish checks still apply) and go to the next bottle |
| `R` | Reject: the bottle is not saved; an error report is written so it can be retried |
| `E` | Edit fields by name; each edit is a correction for this bottle (a `user` fact) |
| `C` | Settle each open conflict: pick a candidate by number, type a value (`=` first for a number), or press Enter to skip |
| `N` | Next: keep the bottle as it is, not approved |
| `Q` | Quit the review; the remaining bottles are kept, not approved |

After `E` or `C` the same bottle is offered again, so it can be approved
with its edits. Anything else asks again, so a stray key never approves a
bottle. If the review itself fails (for example end of input), nothing from
it is applied and no bottle is approved.

When the review ends, the corrections are applied by bottle ID, each
changed bottle's facts are re-reconciled without any model call (a
description that cites a changed field is marked stale), and the approvals
are recorded. Corrections apply only to the bottle they were made for; the
tool does not learn from them for other bottles.

---

## Feedback and Approval

`feedback` works on a saved batch (`batch_<run_id>_<NNN>.json`); it never
calls a model or the web.

### Corrections File

A JSON list:

```json
[
  {"bottle_id": "IMG_4643", "field_name": "batch_number", "original_value": "C92", "corrected_value": "C923"},
  {"bottle_id": "IMG_5120", "field_name": "msrp", "original_value": "", "corrected_value": "59.99"}
]
```

```bash
bourbon-process feedback --batch 20261122_193000_001 --corrections fix.json
```

- A correction applies only to the bottle with that `bottle_id`, and only
  when the bottle's current value equals `original_value` (empty for
  unknown). Otherwise it is reported with the reason ("unknown bottle_id",
  "stale original value", "field is not correctable") and `feedback` exits 1.
- Correctable fields: `name`, `distillery`, `bottler`, `specific_release`,
  `batch_number`, `proof`, `abv`, `age`, `release_year`, `mash_bill`,
  `msrp`, `tasting_notes`.
- Each correction becomes a `user` fact citing the file. A corrected MSRP
  is shown as user-asserted.

### Approving

```bash
bourbon-process feedback --approve IMG_4643 --approve IMG_5120
```

Approval records who and when (`facts.review`) and recomputes the publish
gate; it never clears the other blockers. Changed and approved bottles are
saved again in new folders (`<name>_2`...), and the batch JSON is updated so
the next round starts from the corrected values. A batch file that is not
`schema_version` 2 is refused (exit 1).

---

## Error Handling

### Retries

With retry on (`-r`, the default):

- each model request is tried up to `--retry-attempts` times (default 3),
  waiting 2 s, then doubling, at most 30 s;
- unusable output, HTTP 5xx/408/429 and refused connections are retried;
  a timeout is retried only once;
- a bottle's OCR runs once and is reused for every attempt, and its
  `--hints` note goes with every attempt;
- the fact check makes at most 2 attempts per call; a text-model timeout
  is not retried (the check is recorded as `failed`, or `partial` without a
  description when it hit the description step).

`--no-retry` makes every request once. A model that is missing, cannot read
images or is incompatible, or a server that is gone, stops the run (exit 2)
instead of failing each bottle.

### Error Reports

A bottle that fails (analysis or saving) or is rejected in interactive
review gets `processed/error_report_<bottle_id>.json`:

```json
{
  "schema_version": 2,
  "bottle_id": "IMG_1234",
  "images": ["/abs/path/IMG_1234.jpg", "/abs/path/IMG_1235.jpg"],
  "error_message": "VisionResponseError: ...",
  "error_type": "...",
  "stage": "analysis",
  "hint": "...",
  "details": "...",
  "timestamp": "2026-11-23T12:34:56+00:00",
  "retry_count": 0,
  "suggested_fix": {}
}
```

`stage` is `analysis`, `save` or `rejected`. An existing report is never
overwritten (`_2`, `_3`... are added).

### Retry Failed Bottles

```bash
bourbon-process retry --report processed/error_report_IMG_1234.json
bourbon-process retry --report processed/error_report_IMG_1234.json --hints hints.toml -m gemma4:26b
```

Before retrying you can fill `suggested_fix` with field values you checked
on the bottle, for example `{"batch_number": "C923"}`; they are applied to
the fresh reading as corrections. Only correctable fields are allowed; an
invalid `suggested_fix` stops the retry before any model work. The bottle
then gets the hint values, web research and the fact check, and is saved
in a new folder.

### Hints

A `--hints` TOML file gives per-bottle reading notes for the vision model
and field values a person checked, each with a required `source`. See
[README.md#hints](../../README.md#hints) and
[examples/hints.example.toml](../../examples/hints.example.toml).

---

## Troubleshooting

### Ollama Issues

**Problem:** "Model server unavailable"

```bash
curl http://localhost:11434/api/tags                       # is Ollama answering?
ollama serve                                               # start it
lsof -i :11434                                             # something else on the port?
bourbon-process process -f photos/ -u http://localhost:11435   # Ollama on another port
```

**Problem:** "Model not available"

```bash
ollama list
ollama pull qwen3.6:35b-a3b          # or re-run with --pull
ollama pull gpt-oss:120b             # or --text-model gpt-oss:20b, or --no-verify
```

**Problem:** Slow processing or timeouts

```bash
bourbon-process process -f photos/ -m gemma4:26b            # smaller vision model
bourbon-process process -f photos/ --request-timeout 1200   # a cold vision load can take minutes
bourbon-process process -f photos/ --text-timeout 1800      # slower fact check
bourbon-process process -f photos/ --text-think medium      # faster, less careful extraction
bourbon-process process -f photos/ -b all                   # load each model once
```

### Tesseract OCR Issues

**Problem:** "Tesseract OCR not found"

```bash
brew install tesseract                  # macOS
sudo apt install tesseract-ocr          # Debian/Ubuntu
sudo dnf install tesseract              # Fedora
tesseract --version
```

**Problem:** Poor OCR quality

```bash
tesseract IMG_0001.jpg stdout           # what Tesseract reads from one photo
```

Sharp, well-lit label close-ups help most.

### Memory Issues

**Problem:** Out of memory

```bash
bourbon-process process -f photos/ --text-model gpt-oss:20b   # smaller fact-check model
bourbon-process process -f photos/ --no-verify                # skip the fact check
bourbon-process process -f photos/ -m gemma4:26b              # smaller vision model
```

Do not use `--keep-models-loaded` on a machine that cannot hold both models.
Pre-flight warns when the text model uses more than 75 % of memory and stops
(exit 2) when it is larger than memory.

### Permission Issues

**Problem:** "Output directory not writable"

```bash
ls -la processed/
chmod u+w processed/
bourbon-process process -f photos/ --output-dir /tmp/bourbon_output
```

### Image Issues

**Problem:** "No supported images found"

Supported: `.bmp .heic .jpeg .jpg .png .tiff .webp` (HEIC needs
`pillow-heif`). Earlier outputs (folders with a `.bourbon_output` marker) and
hidden folders are skipped.

**Problem:** Images not grouped correctly

```bash
bourbon-process process -f photos/ --dry-run                  # see the groups
bourbon-process process -f photos/ --group-gap-seconds 20     # photos of one bottle further apart
bourbon-process process -f photos/ --group-max-size 5         # four or more angles per bottle
```

Photos without EXIF capture time are single-photo bottles; name them
`IMG_1_front.jpg` / `IMG_1_back.jpg` to pair them.

### Extraction Accuracy

**Problem:** A wrong value

Correct it in `-i` review (`[E]dit`) or with `feedback --corrections`; do
not edit the Markdown by hand, since the JSON and the fact sheet would not
change. A value you know in advance can go in a `--hints` file.

**Problem:** MSRP is Unknown

The MSRP comes only from a page that states an MSRP, or from you. The reason
on the Facts and Prices lines says what was found (for example only retail
prices). Supply it with a sourced hint (`msrp = 59.99`) or a corrections
file.

**Problem:** Web lookups skipped

Check the bottle's Web Research Log and the run summary for the reason:
`--no-web`, `--web-provider none`, a throttled DuckDuckGo, a placeholder
name or low vision confidence. Set `BRAVE_API_KEY` or `OLLAMA_API_KEY`, or
point `web.searxng_url` at your own SearXNG, for more reliable search.

### Debug Mode

```bash
bourbon-process --debug process -f photos/
```

Adds debug logging (grouping decisions, OCR, model requests) and the
model stage report, and lists settings the command ignores.

---

## Advanced Topics

### Model Selection

Any vision-capable Ollama model can be passed with `-m`, and any model with
the `completion` capability with `--text-model`; pre-flight rejects a model
without the needed capability.

| Model | Role | Notes |
|-------|------|-------|
| `qwen3.6:35b-a3b` | vision | Default; tested end to end (~23 GB) |
| `qwen3.8:27b` | vision | Alternative (~18 GB) |
| `gemma4:26b` | vision | Smallest of these (~16 GB) |
| `gpt-oss:120b` | text | Default fact-check model (~65 GB) |
| `gpt-oss:20b` | text | For smaller machines (~14 GB) |

The old `llama3.2-vision` builds are listed by Ollama but fail on current
Ollama versions ("no longer compatible"); the run stops with exit code 2.

### Vendor Images

```bash
bourbon-process process -f photos/ -v
```

For each bottle the tool takes the product photo of an official or retailer
page it fetched, and asks the vision model whether it shows the same product
as your photo. Only a high-confidence match is kept, as
`images/vendor_ref.<ext>`, with its page, verdict and the note "reference
only - not licensed for publication" in the JSON. A failed vendor lookup
never blocks the bottle. `-v` needs web lookups (`-v --no-web` is an error).

### Quantity

The quantity is the number of byte-identical copies of the bottle's photos
(at least 1). It is not read from the label or the photo content. The total
value is the quantity times the MSRP, and unknown while the MSRP is.

### Concurrency

`--concurrency 2` (the default) overlaps the next bottle's OCR with the
current vision request; results are merged in input order. Use 1 for
evaluation runs, and more than 2 only when Ollama serves requests in
parallel (`OLLAMA_NUM_PARALLEL`).

### Scheduled Runs

```bash
# Crontab entry: process new photos daily at 10pm (no review; nothing becomes publish-ready unattended)
0 22 * * * cd /path/to/HighValueRoom && uv run bourbon-process process -b all -f /photos/new_today >> /var/log/bourbon_process.log 2>&1
```

---

## Performance

Each bottle is one vision request with up to 4 photos (downscaled to
1280 px). With `qwen3.6:35b-a3b` a warm request took a few seconds to under a
minute per bottle in testing; the first request also pays the model load
(minutes on a cold start). The fact check with `gpt-oss:120b` adds about 1-2
minutes per bottle, plus a model load per batch. Times depend heavily on the
models and hardware.

---

## Quick Reference Cheat Sheet

```bash
# Basic usage
bourbon-process process                          # process the current folder
bourbon-process process -f /path/to/photos       # a specific folder
bourbon-process process -o /custom/output        # custom output folder

# Models
bourbon-process process -m gemma4:26b            # smaller vision model
bourbon-process process --text-model gpt-oss:20b # smaller fact-check model
bourbon-process process --no-verify              # no fact check
bourbon-process process --pull                   # download missing models (disk-checked)

# Web
bourbon-process process --no-web                 # no web lookups
bourbon-process process --refresh-web            # ignore the web cache
bourbon-process process -v                       # vendor reference images

# Model serving
bourbon-process process -u http://gpu-box:11434  # Ollama on another host
bourbon-process process --no-server-autostart    # require a running Ollama

# Batch control
bourbon-process process -b all                   # one batch
bourbon-process process -b 10                    # 10-bottle batches
bourbon-process process --split-budget-ratio 0   # no same-bottle split check

# Review
bourbon-process process -i                                    # interactive review
bourbon-process feedback -i                                   # review the latest batch again
bourbon-process feedback --batch <id> --corrections fix.json  # corrections file
bourbon-process feedback --approve <bottle_id>                # approve for publishing
bourbon-process retry --report processed/error_report_<bottle_id>.json

# Checks
bourbon-process process --dry-run                # check everything, process nothing
bourbon-process config show                      # effective settings and their sources
bourbon-process --debug process                  # debug logging and stage report
```

---

## Support

- [docs/adhd/README.md](../adhd/README.md): 5-minute quick start
- [README.md](../../README.md): reference for options, configuration and exit codes
- [specs/001-bourbon-image-processor/spec.md](../../specs/001-bourbon-image-processor/spec.md): the original feature specification (see its banner for what changed)
