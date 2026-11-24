# ToDo

Open work for `bourbon-process`. The web-lookup, fact-check and configuration
build (WP1-WP7) closed most of the earlier list: config files, every no-op
flag, the deprecated shims, the dead code, the stale docs and the test
hygiene are done. What is left: items from the earlier list that this build
did not touch, follow-ups the build reported, and planned improvements.

- Code is named by module and symbol, since line numbers drift. Anchors are as
  of commit `f7148d1` (2026-11-23) plus the uncommitted WP6/WP7 integration.
- Effort: **S** is under an hour, **M** is about half a day, **L** is a day or more.
- 🔶 marks an item that needs an owner decision before anyone starts on it.
- Constraints for every item:
  - Ollama is the only backend.
  - Everything must stay portable to Linux.
  - Tests stay hermetic: no network, no pulls, injected fakes for HTTP, search and models.
  - Accuracy and provenance beat speed: never invent a value; "Unknown + reason" beats a guess; conflicts go to a person.

---

## 1. Decisions needed

- [ ] 🔶 **The 96 committed photos in `HighValueRoom/` (~259 MB)** (L)
  - Options: track them with git-lfs (`git lfs migrate import`), or move them out of the repo.
  - Getting the space back means rewriting history and force-pushing. That changes every commit SHA and needs explicit sign-off.
  - Low-risk first step, doable on its own: make the photo folder configurable (`BOURBON_PHOTO_DIR`, defaulting to `./HighValueRoom`).
    - These tests hard-code the path: `tests/integration/test_ocr_real_photos.py`, `tests/integration/test_vision_live.py`.
  - `.pre-commit-config.yaml` would already reject files over 5 MB, but the pre-commit hooks were never installed.
- [ ] 🔶 **Capture-grouping tie band was waived** (M)
  - `GAP_TIE_SECONDS=0.05` and `GAP_TIE_FRACTION=0.0` (`image_processor.capture_grouping`).
  - Waiving it kept pairwise F1 at 0.827. Turning the band on drops it to 0.796, below the 0.80 floor (`tests/fixtures/grouping/highvalueroom_capture_times.json`).
  - Revisit with a second labelled shooting session. An alternative is to settle near-ties with the vision same-bottle answer.
  - Update the lock test in `tests/unit/test_capture_grouping.py` to match whatever is decided.
- [ ] 🔶 **Label-only `tasting_notes` blocks publishing** (S)
  - A label-only field read at medium or low vision confidence gets low confidence, and any low-confidence value is a publish blocker (`fact_reconcile.compute_gate`). OCR rarely backs tasting notes on one line, so they often block.
  - Options: exempt `tasting_notes` from the low-confidence blocker, leave it Unknown unless backed, or keep it and fix with `[E]dit`.
- [ ] 🔶 **Configurable `publish_core_fields`** (S)
  - The core fields are fixed in `fact_reconcile.CORE_FIELD_GROUPS` (name; distillery or bottler; proof or ABV). Make the list a setting if the social account needs, say, age or batch on every post.

## 2. Accuracy and provenance

- [ ] **Accuracy evaluation harness (owner work)** (L)
  - Truth labels, first for about 20 groups, then all 55: name, distillery or bottler, proof, abv, age, batch, release_year, msrp plus its source URL. Store them next to `tests/fixtures/grouping/highvalueroom_capture_times.json`.
  - Recorded web-evidence replay: save the search results and page excerpts of one live run, then replay them so every later evaluation is deterministic and offline.
  - Metrics:
    - published-wrong rate (target about 0): a publish-ready bottle with a wrong value;
    - precision and coverage per field;
    - false-Unknown rate;
    - unsupported-claim rate by reason code (`facts.verification.rejected`);
    - page-match accuracy;
    - time per bottle.
  - Tune with it: `text.think_extract`/`--text-think`, `web.max_llm_pages`, and temperature 0 vs 1.
  - Fold in the old release/batch item: per-field accuracy of the vision reading, the same-bottle false-alarm, miss and precision rates, `DEFAULT_SPLIT_BUDGET_RATIO` (0.35 has no measured basis), and deterministic post-processing (drop a `batch_number` that equals the proof, strip release tokens already in the name, normalise BiB and Barrel Proof). Record the model tags with the results.
- [ ] **Quotes can span two excerpt segments** (S)
  - `fact_verify._quote_problem`, `_cue_near`, `_release_near` and `_msrp_cue_binds` normalise the whole evidence text, which turns the blank line between excerpt segments into a space. A quote or an MSRP cue can then join the end of one passage and the start of another that are not adjacent on the page ("Example Creek" + "$59.99").
  - Fix: match quotes and cue windows within each `text.split("\n\n")` segment (page text from `html_text` never has empty lines, so a blank line always marks a segment boundary). Add a test.
- [ ] **Vision schema: per-field confidence and printed-text quotes** (M)
  - Today a label citation is a *reading* (`quote_kind: reading`), not verbatim text. Ask the vision model for the printed text behind each field and a per-field confidence, so label citations become verbatim and confidence is per field rather than per bottle.
- [ ] **`verify_copy` gaps** (M)
  - It does not catch a role verb that does not match the field ("produced by X distillery" for the distillery field), numbers written as words, or proper names outside capitalised phrases (`unsupported_name` is a heuristic).
- [ ] **Store the page_match verdicts on the fact sheet** (S)
  - Only failed verdicts appear in `verification.rejected`. Keep all of them for audit and for a later compose-only re-run.
- [ ] **Tighten web corroboration of product-level fields** (S)
  - Name, distillery, bottler and mash bill from a page judged `same_product_other_release` or `unclear` still count as web support (only batch-specific fields need an exact page match). Decide whether to require `exact`.
- [ ] **Release-token proximity and MSRP cue use only the first occurrence of a quote** (S)
  - This can reject a true claim but never accepts a false one. Check every occurrence.
- [ ] **Curated official-domain list** (M)
  - `web_research.is_official_domain` counts any brand or distillery word of 4+ characters in the domain's first label (`trace` in a hypothetical tracesupply.com would be "official"). Replace the heuristic with a curated list of official domains, keeping the heuristic only as a review note.
- [ ] **Photo categories are never assigned** (M)
  - `ImageCategory` BILL and VENDOR exist; only VENDOR is used (for `images/vendor_ref.*`). Receipts and box shots are not classified. Classify only from evidence (vision answer or OCR), never from file names.

## 3. Features

- [ ] **`verify` subcommand** (M)
  - Re-run the fact check on saved schema-v2 outputs without the vision step; refresh stale descriptions; `--relookup` re-runs the web search first.
- [ ] **Social-post drafts** (M)
  - From verified fields only, at most 280 characters, with a citation, behind `--social-drafts`. Only for publish-ready bottles.
- [ ] **More search providers: Tavily and SerpAPI** (M)
  - SerpAPI takes its key in the URL, so redact keys from every logged or recorded URL first.
- [ ] **Parse PDF press releases** (M)
  - Distillers often publish MSRP and batch details as PDFs; `web_fetch` reads only HTML and plain text.
- [ ] **User price CSV** (M)
  - A CSV of known MSRPs with `msrp_source=manual`, applied as user facts with the file as the source.
- [ ] **LLM response cache and web cache pruning** (M)
  - Cache fact-check replies keyed by model, prompt version and the evidence sha256s, so a re-run of unchanged evidence costs no model time.
  - Prune the web cache by size and age, including rejected or unused vendor-image downloads (`images/`), which today stay until removed by hand.
- [ ] **Quantity should come from the user** (M)
  - Quantity counts byte-identical copies of a photo, which is not the number of bottles owned. Make it user-supplied (hint or edit) with `quantity_source: assumed|user`.
- [ ] **Interactive merge and split were removed** (L)
  - Re-add them only with real regrouping and a fresh analysis of the new groups.
- [ ] **Feedback does not tune grouping** (M)
  - If it should, retarget corrections at `gap_seconds` and `max_group_size` per session; today feedback only corrects fields and approves.
- [ ] **Start the next batch while interactive review runs** (M)
  - Review blocks the run; the next batch's vision pass could run meanwhile (mind the model swaps).
- [ ] **Per-stage concurrency limits; `OLLAMA_NUM_PARALLEL` when this tool spawns the server** (M)
  - `--concurrency` applies to vision analysis and web research alike. When the tool starts `ollama serve` itself it could set `OLLAMA_NUM_PARALLEL` to match.
- [ ] **Web tasting notes with attribution: deliberately out of scope**
  - Tasting notes from reviews are opinion and copyrighted text. `tasting_notes` cannot be claimed from the web (`fact_prompt`). Revisit only with explicit attribution and short quotes.

## 4. Known limitations

- [ ] **A re-run skips nothing; it writes `<name>_2` folders** (M)
  - `preview_generator._unique_path` is used for every save; README documents this as intended.
  - Fix:
    - Index existing `<out>/*/<name>.json` files by resolved source path plus MD5 (`lib/duplicates.py`).
    - Skip groups that are already done, and add `--force`/`--reprocess` to override.
    - Store absolute paths and hashes so renamed folders still match.
    - Failed bottles stay eligible for reprocessing.
- [ ] **A nested split loses the outer split warning** (S)
  - When a split half is split again, the leaves come from fresh `_analyze` output and the parent's "group of N" hint is dropped (`BatchProcessor` split recursion).
  - Fix: pass the inherited warnings down the recursion. Do not bring hints back on the false-alarm path.
  - Add a test: a double split of 4 photos should leave both hints on each leaf.
- [ ] **Sibling bottles of the same product are merged with a review note** (M)
  - Fields that are missing count as wildcards (`image_processor.bottle_identity`). Name-subset matching links products such as "Stagg" and "Stagg Jr".
  - Ideas:
    - A targeted second vision prompt on one photo from each half (fill level, label wear, store-pick sticker, barrel number), charged to the split budget.
    - A short list of distinct sub-brands.
    - Splitting a group by photo in interactive review or `feedback`.
  - Re-score any change on the labelled 98-photo fixture. The tolerant bias was chosen because it measured better.
- [ ] **A split group's hint note reaches both halves** (S)
  - `BatchProcessor` passes the group's `--hints` note to both halves, so a note about bottle A also reaches bottle B's vision prompt (the prompt forbids copying values from it). `Image.bottle_ref` is the same for every image of a group, so the halves cannot be told apart for the note lookup.
- [ ] **Ctrl-C waits for a running vision request** (S)
  - On SIGINT/SIGTERM the run sets cancel and shuts the executor down without waiting, but the interpreter still joins worker threads at exit. Consider `os._exit` after logging shutdown, once a test covers it.
- [ ] **Small code-health items** (S)
  - `--approve` help says "applied by the fact-check stage"; it is applied by `feedback` itself.
  - The Web Research Log renders each query inside double quotes, but real queries start with a quoted name, so the page shows `""Example Creek Single Barrel" ..."` (`preview_generator`, see `examples/processed_sample/`). Render queries as code spans instead.
  - Log lines in `lib.model_server` and `cli.commands` print the raw `--ollama-url` (credentials, if a user puts them there); use `settings.redact_url`.
  - `model_server._check_capability` treats an empty capability list (an old Ollama) as "not supported".
  - Settings error messages use the short key name (`enabled=...`); the source names the variable or section, but the full key would be clearer.
  - `PageRecord` is frozen but `cited_for` is a list (hashing raises); a `cited` page with empty `cited_for` is allowed and should be flagged.
  - The SSRF guard resolves a host and `requests` resolves it again, so DNS rebinding is not fully excluded (documented in `web_fetch`). Pinning the IP needs a transport adapter.
  - Provider `requests_made` counters are incremented without a lock.
- [ ] **Fold the `BaseModelServer` ABC into `OllamaServer`** (M)
  - It has a single subclass. A pure move; gate it on the live spawn test (`BOURBON_LIVE_TESTS=1 BOURBON_LIVE_SPAWN=1`) on macOS and Linux.

## 5. Dependencies and install

- [ ] **HEIC: declare `pillow-heif` as an optional extra** (S)
  - Today `.heic` is accepted (`image_organizer`), but `pillow-heif` is not in `pyproject` or `uv.lock`.
  - The README's `uv pip install pillow-heif` is undone by the next `uv sync`. HEIC registration is then skipped silently (`text_extractor`, `ollama_client`).
  - Fix:
    - Add `heic = ["pillow-heif>=0.16"]` (wheels exist for macOS and manylinux), re-lock, and document `uv sync --extra heic`.
    - In pre-flight, warn once and skip `.heic` files when the plugin is missing, instead of failing each bottle at the vision step.
- [ ] **Delete `requirements.txt` and `dev-requirements.txt`** (S)
  - `pyproject` and `uv.lock` are the source of truth, and these two have already drifted.
  - `install.sh` is their only user. If a pip export is ever needed, generate it with `uv export`.
- [ ] **Switch `opencv-python` to `opencv-python-headless`** (S)
  - `cv2` is used only for thresholds and contours (`text_extractor`), with no GUI calls.
  - This removes the `libgl1`/`mesa-libGL` system dependency on Linux (README install sections, `install.sh`).
  - pip users must uninstall `opencv-python` first, because both packages provide `cv2`.
- [ ] **Dev-tooling drift** (S, in its own commit)
  - Remove:
    - `pytest-asyncio`, which no tests use;
    - `black` and `[tool.black]`; ruff formats.
  - Add `pre-commit` to the dev extra and run `pre-commit install`.
  - Bump the ruff-pre-commit rev from `v0.1.0` to the locked 0.17.x. Move `select`/`ignore` under `[tool.ruff.lint]`; ruff already warns about this.
  - `ruff check src` reports about 1050 findings from the broad rule set (mostly pre-existing style: UP045/UP006, TRY003, EM10x, PLR2004, T201...). Decide which rules to keep, then fix or ignore by rule. Also pre-existing B905 (`zip()` without `strict=`) in `commands.py` and `preview_generator.py`.
- [ ] **Fix `install.sh` and delete its duplicate, `scripts/install.sh`** (S)
  - Problems:
    - the shebang `/usr/bin/bash` doesn't exist on macOS; use `#!/usr/bin/env bash`;
    - `libgl1-mesa-glx` and `tesseract-langpacks` are package names that no longer exist;
    - it suggests pulling `qwen3-vl:32b`; change it to `qwen3.6:35b-a3b` and `gpt-oss:120b` (or `gpt-oss:20b`);
    - its dev dependencies are wiped by the next `uv sync`; use `uv sync --extra dev`.
  - Test on Debian or Ubuntu, Fedora, and macOS.

## 6. Local cleanup: untracked files on this machine (owner's call)

- [ ] `rm -rf MagicMock/`: empty folders leaked by an earlier test (no current test creates it).
- [ ] `rmdir bottle_metadata/`: empty, created by the deleted `scraper.py`.
- [ ] `rm -rf tests/fixtures/images/`: three empty folders, untracked and unused.
- [ ] Remove the old test debris in `processed/` (git-ignored): `error_report_20261113_*.json` (3) and 4 empty `Unknown_Bottle_20261122_*` folders, left by tests from before the repo-root guard. They are schema v1, so `retry` refuses them anyway.
- [ ] Delete the root `hints.md`: an untracked template no code reads (hints are TOML now; see `examples/hints.example.toml`). Then drop `hints.md` from `.gitignore`.
- [ ] Move or delete `.claude/settings.local.json.bak.20261122183030`. It may contain machine-local settings, so never commit it.
- [ ] Add to `.gitignore`: `.claude/settings.local.json*`, `.ruff_cache/`, and `MagicMock/` as a fallback (the repo-root test guard is the real fix).
- [ ] 🔶 `IMG_4643.jpeg` (repo root) and `dupes/IMG_4638.jpeg` are tracked.
  - The grouping fixture expects them together, as one bottle. Their numbers are missing from `HighValueRoom/`.
  - Confirm they were set aside on purpose, then move both into the photo set as part of the §1 photo decision.

## Checked and deliberately left alone

- **The `key: value` fallback in `vision_prompt.py` (`_parse_lines`)** is labelled "legacy" but still runs whenever the model returns no JSON. Keep it.
- **`BatchProcessor.split_calls_used`** is a simple read-only accessor next to `split_calls_remaining`. Keep it.
- **`angle_base` and `dhash64`** are still used for pairing photos that have no capture time. Keep them.
- **Moving `test_cli_contracts.py`'s subprocess tests in-process** would be a speed refactor, not cleanup. It isn't tracked here.
- **The test fixture name `highvalueroom_capture_times.json`** matches the old package name but is a data file name; keep it.
