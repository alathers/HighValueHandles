# Bourbon Image Processor

`bourbon-process` reads photos of bourbon bottles with a local vision model
served by [Ollama](https://ollama.com), looks each bottle up on the web,
fact-checks every field against the label, the OCR text and the pages it
found with a local text model, and writes one folder per bottle with
Markdown and JSON metadata. Every value carries its sources; anything that
cannot be verified is written as "Unknown" with the reason, and conflicts go
to a person.

All models run locally in Ollama. The only network traffic besides Ollama is
the web lookup (search queries and page fetches), which `--no-web` turns off.

It runs on macOS and Linux. Ollama is the only model backend.

## Features

- **Local vision model** (default `qwen3.6:35b-a3b`) reads each bottle's label photos
- **Bottle grouping by capture time**: photos of one bottle taken seconds apart are analysed together
- **Web lookups** (on by default, no API key needed): DuckDuckGo, Brave, Ollama web search or SearXNG; polite fetching; every page consulted is logged in the bottle's documents
- **Fact check** with a local text model (default `gpt-oss:120b`): it proposes claims with verbatim quotes; Python checks every quote and value and decides each field
- **Provenance and a publish gate**: each field has a status, a confidence and its sources; nothing is publish-ready until the checks pass and a person approves it
- **Ollama server lifecycle**: reuses a running Ollama or starts `ollama serve` itself, and never downloads a model without `--pull`
- **Config file, environment variables and flags** for every setting, with `bourbon-process config show`
- Batch preview reports, interactive review, and `feedback` / `retry` commands

## Documentation

- [Quick Start Guide (5 minutes)](docs/adhd/README.md)
- [Detailed Documentation](docs/detailed/README.md): troubleshooting, batch processing and advanced topics
- [Example output](examples/README.md) and an [example hints file](examples/hints.example.toml)

## Pipeline stages and models

Two stages call a model. There is no automatic fallback to another model.

| Stage | Model (default) | What it does |
|-------|-----------------|--------------|
| Vision analysis (`VISION_ANALYSIS`) | `-m/--model` (`qwen3.6:35b-a3b`) | Reads the label fields (name, distillery, release, batch, proof, ABV, age, tasting notes, any visible price) from up to 4 photos of a bottle, and says whether the photos show one product. With `-v` it also compares a vendor's product photo with your photo. |
| Fact check (`METADATA_GENERATION`) | `--text-model` (`gpt-oss:120b`) | Reads the label reading, the OCR text and the web excerpts. It extracts claims, each with a verbatim quote and a source ID, and writes a short description that cites the verified fields. It decides nothing: Python verifies every quote and value against the source text, reconciles the fields, assigns confidence by fixed rules and computes the publish gate. |

Steps that use no model:

| Step | How |
|------|-----|
| OCR | Tesseract, on every photo. The text goes to the vision model as a hint and to the fact check as evidence (`O1`, `O2`, ...). |
| Grouping | EXIF capture time (see [How photos are grouped](#how-photos-are-grouped)), checked by the vision model's same-bottle answer. |
| Quantity | File hashes: byte-identical copies of a photo count as more bottles. |
| Web search and fetch | Search provider API or HTML results, then polite HTTP fetches; no model. |
| Reconciliation, confidence, publish gate | Deterministic Python rules. |
| Previews and output files | Markdown and JSON templates. |

`--debug` prints a stage report that marks each stage active or inactive.

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

### Models

A default run needs two models: the vision model and the fact-check text
model. Install them yourself:

```bash
ollama pull qwen3.6:35b-a3b    # vision, about 23 GB
ollama pull gpt-oss:120b       # fact check, about 65 GB
```

or let the tool download the missing ones after a disk-space check by adding
`--pull` to a run. Nothing is ever downloaded without `--pull`.

Hardware: the vision model needs about 23 GB of disk and 32 GB+ of RAM.
`gpt-oss:120b` needs about 65 GB of disk and about 70 GB of free unified (Apple
Silicon) or GPU memory. On a smaller machine use `--text-model gpt-oss:20b`
(about 14 GB), or skip the fact check with `--no-verify` (then no bottle can
become publish-ready). Pre-flight compares the installed text model's size
with physical memory: above 75 % it warns, larger than memory it stops with
exit code 2.

Vision-capable models that have been used with this tool:

| Model | Approx. size | Notes |
|-------|--------------|-------|
| `qwen3.6:35b-a3b` | 23 GB | Default |
| `qwen3.8:27b` | 18 GB | |
| `gemma4:26b` | 16 GB | Smallest of the three |

Any other Ollama model whose capabilities include `vision` can be passed with
`-m`; the text model must report the `completion` capability. A model without
the needed capability is rejected in pre-flight. Models that Ollama lists but
can no longer run (for example the old `llama3.2-vision` builds on current
Ollama versions) fail on the first request and stop the run with exit code 2.

## Usage

```bash
# Process the photos in ./photos; output goes to processed/
uv run bourbon-process process -f ./photos -o processed

# The `process` subcommand can be left out
uv run bourbon-process -f ./photos

# Check inputs, settings, the server and the models without processing anything
uv run bourbon-process process -f ./photos --dry-run

# Smaller machine: the 20b text model, or no fact check at all
uv run bourbon-process process -f ./photos --text-model gpt-oss:20b
uv run bourbon-process process -f ./photos --no-verify

# Offline: no web lookups
uv run bourbon-process process -f ./photos --no-web

# One batch, so the models are swapped only once; review every bottle
uv run bourbon-process process -f ./photos -b all -i

# Show the effective settings and where each value comes from
uv run bourbon-process config show
```

Running `bourbon-process` with no arguments prints the help and exits with 1.
(Inside the activated virtualenv, `uv run` can be left out.)

### Subcommands

| Command | Purpose |
|---------|---------|
| `process` | Scan a folder, group photos into bottles, read each bottle, look it up on the web, fact-check it and save the results |
| `feedback` | Apply corrections (`--corrections file.json` or `-i`) and approvals (`--approve <bottle_id>`) to a saved batch (`--batch <id>`, default the most recent) |
| `retry` | Re-run one failed bottle from its `error_report_<bottle_id>.json` (`--report`) |
| `config` | `config show [--json]`: config files, every value and its source. `config template`: a commented config file |

`--debug`, `--config PATH` and `--no-config` work before or after the
subcommand.

### `process` options

Every option can also be set in a config file or an environment variable;
see [Configuration](#configuration). Defaults are the built-in values.

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --image-folder` | `.` | Folder with the bottle photos (scanned recursively) |
| `-o, --output-dir` | `processed` | Output folder; must not be the photo folder itself, and is best kept outside it |
| `-b, --batch-size` | `5` | Bottles per batch, or `all`. Each batch is read, enriched, reviewed and saved before the next starts |
| `--concurrency` | `2` | Bottle groups analysed in parallel (1-10). Output is identical to a sequential run. See [Concurrency](#concurrency) |
| `--group-gap-seconds` | `12` | Most seconds between consecutive photos of one bottle |
| `--group-max-size` | `3` | Most photos per bottle (0 = no limit) |
| `--split-budget-ratio` | `0.35` | Extra analyses per group the run may spend splitting groups the model reports as mixed (at least 4 per run; 0 = off, max 2). Retries of a request do not count against it. See [How photos are grouped](#how-photos-are-grouped) |
| `-m, --model` | `qwen3.6:35b-a3b` | Vision model (Ollama tag); must be vision-capable |
| `-u, --ollama-url` | `http://localhost:11434` | Ollama URL |
| `--server-start-timeout` | `30` | Seconds to wait for a server this run starts |
| `--request-timeout` | `600` | Per-request vision timeout in seconds, including a cold model load |
| `--keep-server` / `--no-keep-server` | off | Leave a server started by this run running |
| `--server-autostart` / `--no-server-autostart` | on | Start a local Ollama if none is running; `--no-server-autostart` fails instead |
| `--keep-models-loaded` / `--no-keep-models-loaded` | off | Keep the vision and text models loaded together instead of unloading one before the other loads |
| `--pull` / `--no-pull` | off | Download missing models (vision and, when the fact check is on, text) after a disk-space check |
| `--min-free-gb` | `5` | GiB to keep free on the model store after a `--pull` |
| `-i, --interactive` / `--no-interactive` | off | Review each bottle before it is saved: approve, reject, edit, settle conflicts |
| `--hints PATH` | none | Hints TOML file: reading notes and sourced field values per bottle. See [Hints](#hints) |
| `-r, --allow-retry` | on | Repeat model requests that fail in a way that may pass. See [Retries](#retries) |
| `--no-retry` | | Make every model request once |
| `--retry-attempts N` | `3` | Attempts per model request when retry is on (1-5) |
| `--verify` / `--no-verify` | on | Run the fact check with the text model |
| `--text-model TAG` | `gpt-oss:120b` | Text model of the fact check, or `none` (same as `--no-verify`) |
| `--text-think low\|medium\|high` | `high` | Reasoning effort of the fact-check extraction call |
| `--text-timeout SECONDS` | `900` | Read timeout of one text-model call |
| `--web` / `--no-web` | on | Look bottles up on the web and cite the pages used |
| `--web-provider` | `auto` | `auto`, `duckduckgo`, `brave`, `ollama`, `searxng` or `none`. See [Web lookups](#web-lookups) |
| `--web-cache-dir PATH` | `${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web` | Web cache folder |
| `--refresh-web` | off | Ignore cached pages and search results for this run (new results are still cached) |
| `-v, --include-vendor-images` / `--no-include-vendor-images` | off | Fetch a vendor reference image per bottle (needs web lookups; `-v` with `--no-web` is an error) |
| `--dry-run` | off | Check everything without starting a server, pulling, loading or analysing |
| `--debug` | off | Debug logging and the model stage report |
| `--config PATH` | | Read only this config file |
| `--no-config` | | Read no config file |

`-v` is short for `--include-vendor-images`, not "verbose"; use `--debug` for
more output.

### `feedback` options

| Option | Description |
|--------|-------------|
| `-o, --output-dir` | Folder that holds the batch files (default `processed`) |
| `--batch ID` | Batch ID, e.g. `20261122_193000_001` for `batch_20261122_193000_001_preview.md` (default: the most recent batch) |
| `--corrections FILE` | Corrections file; see [Corrections file](#corrections-file) |
| `-i, --interactive` / `--no-interactive` | Review the batch's bottles interactively (used when no `--corrections` is given) |
| `--approve BOTTLE_ID` | Approve a bottle for publishing; repeatable |
| `--debug`, `--config`, `--no-config` | As above |

### `retry` options

`retry --report error_report_<bottle_id>.json` takes `-o`, `--hints` and the
same model, server, retry, fact-check and web options as `process` (`-m`,
`-u`, `--server-start-timeout`, `--request-timeout`, `--keep-server`,
`--server-autostart`, `--keep-models-loaded`, `-r`, `--no-retry`,
`--retry-attempts`, `--verify`, `--text-model`, `--text-think`,
`--text-timeout`, `--web`, `--web-provider`, `--web-cache-dir`,
`--refresh-web`, `-v`), but not `--pull`, `--min-free-gb`, `--dry-run`,
`-i`, `-b`, `--concurrency` or the grouping options. The bottle goes through
the same steps as in `process`: OCR, the vision request, hint values, the
report's `suggested_fix`, web lookup and fact check, then a new output
folder.

## Configuration

Every setting is resolved from these layers, lowest precedence first:

1. built-in defaults;
2. the user file `${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml`;
3. the project file `./bourbon.toml` (current directory only);
4. `BOURBON_*` environment variables (an empty value counts as unset);
5. command-line flags.

`--config PATH` (or `BOURBON_CONFIG=PATH`) reads only that file instead of 2
and 3, and the file must exist. `--no-config` (or `BOURBON_NO_CONFIG=1`)
reads no file, which makes a run reproducible from its flags alone. The two
cannot be combined.

Config files are sectioned TOML: `[input]`, `[output]`, `[processing]`,
`[grouping]`, `[model]`, `[text]`, `[server]`, `[retry]`, `[web]`,
`[logging]`. Unknown sections or keys and wrong types are errors, and all
errors are listed together, each naming the file, variable or flag it came
from (exit code 1). Relative paths in a file are relative to that file.

```bash
uv run bourbon-process config template > bourbon.toml   # every key, commented, with its env var and flag
uv run bourbon-process config show                      # files read, every value and its source
uv run bourbon-process config show --json
```

[`bourbon.example.toml`](bourbon.example.toml) is the generated template; a
test keeps it identical to `config template`. Some settings exist only in
the file: `text.think_compose`, `text.num_ctx`, the retry delays and the web
limits (`max_queries`, `max_results`, `max_pages`, `max_llm_pages`,
`fetch_timeout_seconds`, `max_page_bytes`, `max_image_bytes`,
`per_host_interval_seconds`, `max_requests_per_run`, `cache_ttl_hours`,
`search_cache_ttl_hours`, `deny_domains`, `allow_domains`) and
`web.searxng_url` (also `BOURBON_SEARXNG_URL`).

**API keys go in environment variables only** (`BRAVE_API_KEY` or
`BOURBON_BRAVE_API_KEY`, and `OLLAMA_API_KEY`). A key-like name in a config
file is an error. Key values are never logged or written; `config show` and
the run settings file say only "set" or "unset".

Each `process` run writes its effective settings, the tool version and the
model names to `<output_dir>/run_<run_id>_settings.json` (never
overwritten; no API keys). The `run_id` is also in every bottle's JSON.

## Web lookups

Web lookups are on by default. For each bottle the tool builds at most two
search queries from the label reading only (name, distillery, release,
batch), searches, fetches up to 5 pages, and hands excerpts of up to 4 of
them to the fact check. It never fails a bottle: a search or fetch problem
is recorded as `web_research.status` `failed` or `skipped` with the reason,
and the bottle is fact-checked with what it has. A bottle whose name is a
placeholder, or whose label reading has low vision confidence, is not
searched.

**Search providers.** `--web-provider auto` (the default) picks the first
that applies:

1. Brave Search API if `BRAVE_API_KEY` (or `BOURBON_BRAVE_API_KEY`) is set;
2. Ollama web search if `OLLAMA_API_KEY` is set (a free ollama.com account);
3. SearXNG if `web.searxng_url` (or `BOURBON_SEARXNG_URL`) is set. The
   instance must allow JSON output (`search.formats` includes `json` in its
   `settings.yml`);
4. otherwise DuckDuckGo, which needs no key.

Naming a provider that lacks its key or URL is a configuration error
(exit 1), as is asking for web lookups (`--web` or `-v`) with
`--web-provider none`. With the default settings and `--web-provider none`
the run logs one warning and records web research as skipped.

DuckDuckGo is queried through its HTML results page (with the lite page as a
fallback) at most once every 2 seconds. A throttled answer (HTTP 202, 403 or
429, or an anomaly or captcha page) is retried once after a pause; after 3
throttled queries in a row DuckDuckGo is turned off for the rest of the run
with one warning, and later bottles record web research as skipped. Every
provider's results are cached for 24 hours (`web.search_cache_ttl_hours`).

**Polite fetching.**

- robots.txt is honoured for every host. A missing robots.txt (404/410)
  allows fetching; any other error, a timeout or a crawl delay over 30 s
  means the host is skipped.
- At least 2 s between requests to one host (`web.per_host_interval_seconds`,
  or the host's crawl delay if longer), and at most 500 web requests per run
  (`web.max_requests_per_run`).
- A generic User-Agent, `bourbon-process/<version> (label-research; respects
  robots.txt)`, no cookies and no Referer.
- Only HTML and plain text pages are read, at most 2 MB of each
  (`web.max_page_bytes`). Redirects are followed by hand (at most 5), and
  every hop is checked: local, private and non-public addresses are refused.
- HTTP 429 or 503 is retried once, honouring `Retry-After` up to 30 s.

**Cache.** Pages are cached for 7 days (`web.cache_ttl_hours`), refusals
(robots.txt, wrong type, HTTP 4xx) for 24 hours, in
`${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web/` (`search/`, `pages/`,
`robots/`, `images/`). `--web-cache-dir` moves it; `--refresh-web` skips
cached entries for one run.

**Which sites.** Social media, video, forum, Q&A, auction and resale sites
are never fetched or cited (built-in list in
`image_processor.web_research.DENY_DOMAINS`). Add more with
`web.deny_domains = ["example.com"]` (subdomains included). With
`web.allow_domains` set, only those domains, plus the brand's own site, are
used. Each page is classed `official` (the domain contains a brand or
distillery word), `retailer` (the page has product data with offers) or
`reference`; pages that do not
mention every distinctive word of the bottle's name are dropped.

**Every site consulted is logged.** Each bottle's JSON `web_research` lists
the provider, every query, and one `pages[]` entry per URL considered, with
its class, retrieval time, outcome (`cited`, `fetched_not_cited` or
`skipped`), the fields it supports, and the reason it was skipped or not
cited. The Markdown page shows the same as a **Web Research Log** table,
also when the lookup was skipped or failed. Each URL and its outcome are
also logged at INFO level during the run.

**Vendor images** (`-v`). For each bottle the tool looks for a product photo
(the page's `og:image` or its product data) on the official and retailer
pages it fetched, tries at most 2 (JPEG, PNG or WebP, at least 300 px on the
short edge, at most 5 MB), and asks the vision model whether each shows the
same product as your photo. Only a high-confidence "same product" answer that
does not say "different release" is kept, as `images/vendor_ref.<ext>`.
**Vendor images are a reference for you only and are not licensed for
publication**; the JSON records `"usage": "reference only - not licensed
for publication"`.

## Fact check and provenance

After the vision pass of a batch, the tool unloads the vision model (unless
`--keep-models-loaded`) and runs the text model on each bottle in turn:

1. **Extract.** The model sees the label reading (`L1`), the OCR text of
   each photo (`O1`...) and the web excerpts (`W1`...), and proposes claims:
   field, value, source ID and a verbatim quote. It also says whether each
   page is about this exact bottle.
2. **Verify.** Python checks every claim: the quote must appear in that
   source's text, and the value must match the quote (numbers, units, proof
   derived from ABV, age, batch tokens, "distilled by" versus "bottled by").
   Batch-specific claims need a page about this exact release. Rejected
   claims are kept in the JSON with a reason code.
3. **Reconcile.** Each field gets a value, a status, a confidence and its
   citations. Disagreements become conflicts; code never settles one.
4. **Compose.** When the name and at least one other field are supported,
   the model writes a short description in which every sentence cites the
   facts it uses. Python checks the description against the facts; if a
   sentence fails, there is no description and the reason is recorded.
5. **Publish gate.** See below.

**Field statuses.**

| Status | Meaning |
|--------|---------|
| `supported` | Backed by the label reading, OCR text and/or verified web quotes |
| `user` | Stated by a person (hint, interactive edit or corrections file); it wins over the sources, and any disagreement is kept as a resolved conflict |
| `conflict` | Sources disagree; the value is empty until a person decides |
| `unknown` | Nothing verifiable; the `reason` says what was searched and found |

Only `supported` and `user` values reach the bottle's flat fields; the rest
are written as "Unknown" with the reason.

**Confidence** follows fixed rules; the model's own confidence is never used:

- Label reading backed by OCR text, or by a verified web quote: **high**.
- Label reading alone: **medium** when the vision confidence was high,
  otherwise **low**.
- Web only: **high** with two agreeing domains including an official one;
  **medium** with one official domain, or two other domains; **low** with
  one non-official domain (raised to medium if OCR agrees).
- OCR text alone: **low**.
- MSRP: only the web rules apply (an OCR or label amount never raises it).
- User-stated: **high**.

**Prices.** A price seen on the label or a shop tag is recorded as a
*retail* price, never as the MSRP. Retail prices on shop pages, including
structured product data, are retail too. The MSRP comes only from a page
that states it with MSRP wording near the amount (then its confidence
follows the web rules above), or from you (hint, edit or corrections file).
Otherwise MSRP is "Unknown" with a reason such as "W1 $64.99 ... no source
stated an MSRP". The total value is unknown while the MSRP is.

**Conflicts and review.** The Markdown page lists open conflicts under
**Needs Review** with each candidate value and its quotes. Settle them:

- in `process -i` or `feedback -i`: `[C]onflicts` shows each open conflict;
  pick a candidate by number, type a value, or skip. `[E]dit` changes any
  field. Each choice is a correction (a `user` fact) for that bottle;
- with a corrections file: `bourbon-process feedback --corrections fix.json`.

After a correction the facts are re-reconciled without any model call; a
description that cites a changed field is marked stale.

**Publish gate.** A bottle is publish-ready (`facts.publish.ready`) only
when there are no blockers. Blockers are:

- the fact check did not finish as `verified` (failed, skipped, `--no-verify`);
- an unresolved conflict;
- a field with a value at low confidence;
- no name, distillery or bottler, or proof or ABV, supported at medium
  confidence or better;
- a split or mixed-group warning from grouping;
- no verified description, or a stale one;
- no approval by a person.

Approve a bottle with `[A]ccept` in `process -i` or `feedback -i`, or with
`bourbon-process feedback --approve <bottle_id>` (repeatable). The approval
is recorded with who and when, and the gate is recomputed; an approval
never clears the other blockers. The Markdown page starts with
**Publish-ready** or **NOT publish-ready: <blockers>**.

**Smaller machines and speed.**

- `--text-model gpt-oss:20b` uses the smaller model; `--no-verify` (or
  `--text-model none`) skips the check: fields then come from the label,
  OCR and your corrections only, and no bottle can be publish-ready.
- `--text-think low|medium` makes extraction faster and less careful.
- `--keep-models-loaded` keeps both models in memory (for a machine with
  enough memory for both, such as a 128 GB Mac). Otherwise the vision model
  is unloaded before the fact check and the text model after it, in every
  batch. Use `-b all` to load each model only once per run.
- Expect about 1-2 minutes per bottle for the fact check with
  `gpt-oss:120b`, on top of the vision step.

## Hints

`--hints PATH` (or `processing.hints_file`) reads a TOML file with one table
per bottle ID (the first photo's file name without extension):

```toml
[bottle.IMG_4643]
note = "Batch number is on the neck label in the third photo."
source = "Owner inspected the bottle, 2026-11-20"

[bottle.IMG_4643.fields]
batch_number = "C923"
```

- `note` is reading guidance for the vision model (at most 500
  characters); it is told never to copy values from it.
- `fields` are values a person established on the bottle. `source` is
  required with them. They are applied after the vision step as corrections
  recorded with the file, the bottle ID and the source, and the fact check
  treats them as `user` facts. When the label reading differs, the hint
  wins and both values are kept as a resolved conflict.
- Allowed fields: `name`, `distillery`, `bottler`, `specific_release`,
  `batch_number`, `proof`, `abv`, `age`, `release_year`, `msrp`.
- An invalid file stops the run before any model work (exit 1). A hint for
  a bottle ID that the run never saw is logged as a warning.

See [`examples/hints.example.toml`](examples/hints.example.toml).

## Corrections file

`feedback --corrections fix.json` reads a JSON list:

```json
[
  {"bottle_id": "IMG_4643", "field_name": "batch_number", "original_value": "C92", "corrected_value": "C923"},
  {"bottle_id": "IMG_5120", "field_name": "msrp", "original_value": "", "corrected_value": "59.99"}
]
```

A correction is applied only to the bottle with that `bottle_id`, and only
when the bottle's current value equals `original_value` (empty for an
unknown value). Allowed fields are those of the fact sheet: `name`,
`distillery`, `bottler`, `specific_release`, `batch_number`, `proof`, `abv`,
`age`, `release_year`, `mash_bill`, `msrp`, `tasting_notes`. Every
correction that is not applied is logged with its reason and makes
`feedback` exit with 1. Changed and approved bottles are saved again in new
folders, and the batch JSON is updated so the next round starts from the
corrected values. `feedback` reads only batch files with `schema_version` 2.

## Retries

With retry on (`-r`, the default), each model request is tried up to
`--retry-attempts` times (default 3) with a growing pause (2 s, doubling,
at most 30 s). Unusable output, HTTP 5xx/408/429 and refused connections are
retried; a timeout is retried only once, because each wait can take
minutes. A bottle's OCR runs once and is reused for every attempt, and its
hint note goes with every attempt. The fact check makes at most 2 attempts
per call (1 with `--no-retry`); an empty, unparseable or cut-off reply is
retried (a cut-off one at a lower think level), but a text-model timeout is
not: the check is recorded as `failed` (or as `partial`, without a
description, when the timeout hit the description step). `--no-retry` makes every request once. A bottle that still fails
gets an `error_report_<bottle_id>.json` for `retry`.

## Concurrency

`--concurrency N` (default 2) analyses up to N bottle groups at once: with 2,
the next bottle's OCR overlaps the current bottle's vision request. Results
are merged in input order, so the output is the same as a sequential run.

- Use `--concurrency 1` for evaluation runs that must be reproducible step by step.
- Use more than 2 only when Ollama serves requests in parallel
  (`OLLAMA_NUM_PARALLEL`); otherwise the extra requests queue on the
  server and their wait counts toward `--request-timeout`.

The fact check always runs one bottle at a time.

## Model server (Ollama)

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
- Before processing, the required models are checked: the vision model,
  plus the text model when the fact check is on. Each must be installed (or
  pulled with `--pull`) and report its capability (`vision` or
  `completion`). Only the vision model is loaded at this point; a failed
  warm-up is only a warning.
- A missing model makes pre-flight fail with exit code 2. The error shows the
  download size, the installed models with that capability, and the exact
  `ollama pull <model>` command.

## `--pull` and the disk check

Models are **never downloaded** unless you pass `--pull`. With `--pull`:

1. The download size of every missing model is read from the registry
   manifest. Blobs that are already complete in the local store are not
   counted.
2. Free space is measured on the Ollama model store: `$OLLAMA_MODELS` if set,
   else `~/.ollama/models`, or on Linux the systemd service store
   `/usr/share/ollama/.ollama/models` when that exists and `~/.ollama/models`
   does not.
3. The pull is refused unless all downloads plus a safety margin fit. The
   margin is the larger of `--min-free-gb` (default 5 GiB) and 10% of the
   download. It is also refused if a size cannot be determined, or if a
   model is not in the registry.
4. During the download the pull is aborted if free space drops below
   `--min-free-gb`. Ollama keeps partially downloaded blobs and resumes them
   on the next pull; this tool never deletes anything.

For a remote server (`-u http://other-host:11434`) the free space cannot be
measured; the pull goes ahead with a warning that states the download size.

`--dry-run --pull` reports what would be downloaded and whether it fits,
without downloading.

## `--dry-run`

A dry run only checks: the settings, the photo folder, that the output
folder could be created, the hints file, Tesseract, and the model server. It
logs the active stages, the text model, the chosen web search provider and
the text model's memory check. It probes the server but never starts it,
never pulls or loads a model, writes nothing to disk, makes no web request,
and lists the bottle groups it found. It still fails (exit 2) when a real
run could not start the server, for example with `--no-server-autostart` or
a stopped Linux systemd service, or when a required model is missing.

## How photos are grouped

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
   (2 extra analyses). Duplicates and burst retakes stay with their
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
   number of groups (default 0.35, at least 4). The cap counts analyses
   (OCR plus a vision request), not attempts: retries of a request do not
   count against it. Once used up, a mixed group is kept whole with a review
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
`metadata.warnings`. A split or mixed-group warning also blocks publishing
until a person has looked. The run ends with a one-line summary of the
check.

Supported formats: `.bmp .heic .jpeg .jpg .png .tiff .webp`. HEIC needs the
optional `pillow-heif` package (`uv pip install pillow-heif`). Hidden folders
and earlier output folders (marked with a `.bourbon_output` file) are skipped.

## How bottles are read

- Each bottle gets one vision request with up to 4 of its photos (receipts
  are skipped unless nothing else exists). Photos are rotated using their
  EXIF orientation and downscaled to 1280 px. Tesseract OCR text goes along
  as a hint only.
- The model is asked for JSON (name, distillery, release, proof, ABV, age,
  batch, tasting notes, and any price visible in a photo). It must report
  only what is printed on the label. A visible price is kept as
  `label_price` (retail), never as the MSRP.
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

The bottle JSON (`schema_version` 2) holds the flat fields (`name`,
`distillery`, `specific_release`, `proof`, `abv`, `age`, `tasting_notes`,
`msrp`, `batch_number`; `null` when unknown), `label_price`, `quantity`,
`total_value`, `run_id`, `reference_links` (cited pages first),
`corrections`, `facts` (fields with status, confidence, citations and
reasons; sources; conflicts; prices; verification; the description; the
publish gate; the review), `web_research` (provider, queries, every page
consulted with its outcome), `vendor_image`, `review_required`, `metadata`,
and the image lists. Each `evidence/W<n>.txt` matches the `content_sha256`
of its source in `facts.sources`. See
[`examples/processed_sample/`](examples/processed_sample/) for a complete
example.

The batch preview has one row per bottle with Ready (Y/N), open Conflicts,
Unknown fields, the Verification status and the MSRP with its source,
followed by each bottle's blockers and web count.

Existing folders are never overwritten: a second bottle (or a re-run) with
the same name is saved as `<bottle_name>_2`, `_3`, and so on.

Log lines have the form `<time> <LEVEL> [<bottle id or ->] <module>:
<message>`. At the end of a run the summary lists the bottles processed and
failed, the web search provider, the web research and fact-check status
counts, how many bottles are publish-ready, and the model usage (calls and
durations by stage and by model).

## Exit codes

`process`:

| Code | Meaning |
|------|---------|
| 0 | Success (also a successful `--dry-run`) |
| 1 | Invalid setting (any source, including an invalid `--ollama-url` port), invalid hints file, output folder equal to the photo folder, Tesseract missing, some bottles failed, or no arguments given |
| 2 | Model server or model unavailable (not running and cannot be started, model missing, lacking the needed capability, refused pull, incompatible model), or the text model larger than physical memory. An unknown flag (argparse) also exits 2 |
| 3 | Photo folder missing or contains no supported images |
| 4 | Every bottle failed |
| 5 | Output folder not writable |
| 128+N | Stopped by signal N (SIGTERM, SIGHUP, SIGQUIT); a server started by the run is stopped first |

`feedback`: 0 success, 1 error (including a batch file that is not schema
version 2, or a correction or approval that was not applied), 2 batch not
found.
`retry`: 0 success, 1 error (including model server unavailable or a report
that is not schema version 2), 2 report not found.
`config`: 0, or 1 for an invalid setting.

## Environment variables

Each setting with a flag also has a `BOURBON_*` variable; flags win over
variables, and variables over config files.

| Variable | Setting | Flag |
|----------|---------|------|
| `BOURBON_IMAGE_FOLDER` | `input.image_folder` | `-f` |
| `BOURBON_OUTPUT_DIR` | `output.output_dir` | `-o` |
| `BOURBON_BATCH_SIZE` | `processing.batch_size` | `-b` |
| `BOURBON_CONCURRENCY` | `processing.concurrency` | `--concurrency` |
| `BOURBON_INTERACTIVE` | `processing.interactive` | `-i` |
| `BOURBON_HINTS_FILE` | `processing.hints_file` | `--hints` |
| `BOURBON_GROUP_GAP_SECONDS` | `grouping.gap_seconds` | `--group-gap-seconds` |
| `BOURBON_GROUP_MAX_SIZE` | `grouping.max_group_size` | `--group-max-size` |
| `BOURBON_SPLIT_BUDGET_RATIO` | `grouping.split_budget_ratio` | `--split-budget-ratio` |
| `BOURBON_MODEL` | `model.vision_model` | `-m` |
| `BOURBON_PULL` | `model.pull` | `--pull` |
| `BOURBON_MIN_FREE_GB` | `model.min_free_gb` | `--min-free-gb` |
| `BOURBON_KEEP_MODELS_LOADED` | `model.keep_models_loaded` | `--keep-models-loaded` |
| `BOURBON_VERIFY` | `text.enabled` | `--verify` / `--no-verify` |
| `BOURBON_TEXT_MODEL` | `text.model` | `--text-model` |
| `BOURBON_TEXT_THINK` | `text.think_extract` | `--text-think` |
| `BOURBON_TEXT_TIMEOUT` | `text.timeout_seconds` | `--text-timeout` |
| `BOURBON_OLLAMA_URL` | `server.ollama_url` | `-u` |
| `BOURBON_SERVER_AUTOSTART` | `server.autostart` | `--server-autostart` |
| `BOURBON_KEEP_SERVER` | `server.keep_running` | `--keep-server` |
| `BOURBON_SERVER_START_TIMEOUT` | `server.start_timeout` | `--server-start-timeout` |
| `BOURBON_REQUEST_TIMEOUT` | `server.request_timeout` | `--request-timeout` |
| `BOURBON_RETRY` | `retry.enabled` | `-r` / `--no-retry` |
| `BOURBON_RETRY_ATTEMPTS` | `retry.max_attempts` | `--retry-attempts` |
| `BOURBON_WEB` | `web.enabled` | `--web` / `--no-web` |
| `BOURBON_WEB_PROVIDER` | `web.provider` | `--web-provider` |
| `BOURBON_SEARXNG_URL` | `web.searxng_url` | (none) |
| `BOURBON_INCLUDE_VENDOR_IMAGES` | `web.vendor_images` | `-v` |
| `BOURBON_WEB_CACHE_DIR` | `web.cache_dir` | `--web-cache-dir` |
| `BOURBON_DEBUG` | `logging.debug` | `--debug` |
| `BOURBON_CONFIG` | config file to read (like `--config`) | `--config` |
| `BOURBON_NO_CONFIG` | read no config file (like `--no-config`) | `--no-config` |

Boolean variables take `1/true/yes/on` or `0/false/no/off`.

Other variables:

| Variable | Effect |
|----------|--------|
| `BRAVE_API_KEY` (or `BOURBON_BRAVE_API_KEY`) | Brave Search API key; makes `auto` pick Brave |
| `OLLAMA_API_KEY` | Ollama web search key; makes `auto` pick Ollama when no Brave key is set |
| `BOURBON_LOG_DIR` | Log folder for servers started by the tool |
| `OLLAMA_BIN` | Executable used to start `ollama serve` (default `ollama` on `PATH`) |
| `OLLAMA_MODELS` | Ollama model store used for the disk check (same meaning as for Ollama itself) |
| `XDG_CONFIG_HOME`, `XDG_CACHE_HOME` | Base folders of the user config file and the web cache |

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
ollama pull gpt-oss:120b                     # or --text-model gpt-oss:20b, or --no-verify
```

**Not enough disk space to pull**: free space on the model store, lower the
margin with `--min-free-gb`, or choose smaller models (`-m gemma4:26b`,
`--text-model gpt-oss:20b`).

**Timeouts on the first bottle**: a cold model load can take minutes; raise
`--request-timeout` (vision) or `--text-timeout` (fact check).

**Out of memory**: use `--text-model gpt-oss:20b` or `--no-verify`, do not
use `--keep-models-loaded`, or use a smaller vision model (`-m gemma4:26b`).

**Web lookups always skipped**: check the run summary and each bottle's Web
Research Log for the reason. DuckDuckGo may throttle a long run; set
`BRAVE_API_KEY` or `OLLAMA_API_KEY`, or point `web.searxng_url` at your own
SearXNG.

**Tesseract OCR not found (exit 1)**: install it (`brew install tesseract`,
`sudo apt-get install tesseract-ocr`, or `sudo dnf install tesseract`).

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest                    # hermetic, see below
uv run --extra dev pytest --cov=src --cov-report=html
```

The default suite is hermetic. A guard in `tests/conftest.py` blocks model
pulls and inference requests, spawning `ollama`, and every connection to a
host that is not loopback. Each test also runs in its own temporary folder
with its own config and cache folders, and without `BOURBON_*` variables or
API keys. Tests use injected fakes for search, HTTP and the models.

Live tests are skipped unless enabled. They never pull models:

```bash
# Model server: reuse the running Ollama; add BOURBON_LIVE_SPAWN=1 to also start
# a dedicated `ollama serve` on 127.0.0.1:11500
BOURBON_LIVE_TESTS=1 uv run --extra dev pytest tests/integration/test_model_server_live.py

# Vision on real photos (BOURBON_MODEL=<tag> picks another model)
BOURBON_LIVE_TESTS=1 BOURBON_LIVE_VISION=1 uv run --extra dev pytest tests/integration/test_vision_live.py

# Fact check against the installed text model (BOURBON_LIVE_TEXT_MODEL=<tag> picks another)
BOURBON_LIVE_TESTS=1 BOURBON_LIVE_TEXT=1 uv run --extra dev pytest tests/integration/test_fact_check_live.py

# Real web search and fetches (DuckDuckGo unless a key or SearXNG URL is set)
BOURBON_LIVE_TESTS=1 BOURBON_LIVE_WEB=1 uv run --extra dev pytest tests/integration/test_web_live.py
```

Two tests skip on case-sensitive file systems (the usual Linux setup),
because they check case-insensitive path matching:
`test_exclude_differing_only_in_case_matches_on_case_insensitive_fs`
(`tests/unit/test_image_organizer.py`) and
`test_output_dir_differing_only_in_case_is_rejected`
(`tests/integration/test_cli_save_path.py`).

`scripts/regen_example_sample.py` regenerates
[`examples/processed_sample/`](examples/processed_sample/) from the real
output code, with no model and no network; a contract test fails when the
sample no longer matches.

## License

Personal bourbon inventory project.
