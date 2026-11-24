# Feature Specification: Checkpoint / Stage-Resumable Runs (`--from-stage`)

**Feature Branch**: `003-stage-resumable-runs`

**Created**: 2026-11-24

**Status**: Draft

**Input**: User description: "Checkpoint / Stage-Resumable Runs (--from-stage): Persist raw Vision (L) and OCR (O) outputs to disk immediately after Stage 1. Allow reruns starting at research, factcheck, or render so adjusting search queries or verification rules never requires re-running the 2-hour vision pass."

## Overview

A full `process` run over the reference photo set (98 photos, 55 bottle
groups) spends about two hours in the vision pass before any web research or
fact check happens. Today nothing from that pass reaches disk until each
batch has also been researched, fact-checked, reviewed and rendered. So:

- tuning a search query, a verification rule or the output layout means
  re-reading every label with the vision model;
- a crash, a Ctrl-C or a model failure during research, fact check or review
  of a batch throws away that batch's vision readings.

This feature splits a run into four named stages. It saves each bottle's
label reading (the fact check's `L1` source) and OCR text (`O1..On`) as soon
as that bottle's first stage finishes. A later run can then start at
`research`, `factcheck` or `render` from those saved results.

| Stage | What it covers today | Needs a model |
|-------|----------------------|---------------|
| 1. `vision` | Scan the photos, group them by capture time, assign bottle IDs, run OCR on every photo, read the label with the vision model (with the same-bottle split check), count quantity | Vision model |
| 2. `research` | Web search, polite fetches, page excerpts, the outcome of every page consulted, the vendor reference image (`-v`) | None (vision model only for the `-v` image comparison) |
| 3. `factcheck` | Evidence pack, text-model claim extraction, deterministic quote and value verification, reconciliation, confidence, publish gate, description | Text model (none when the fact check is off) |
| 4. `render` | Interactive review (`-i`), the bottle folders (Markdown, JSON, OCR text, evidence files, image copies), batch preview and batch data, run summary | None |

Hint field values from `--hints` are applied between stages 1 and 2, as they
are today. They also shape the web search: the search identity is the label
reading with hint values applied, so a hint value can change which pages
research finds.

Human review decisions (interactive review in the source run, and later
`feedback` corrections and approvals on its batches) happen after the fact
check. A resumed run must never drop one silently (FR-030, FR-035).

## Clarifications

### Session 2026-11-24

- Q: Should an interrupted vision pass itself be resumable (FR-006)? → A:
  No (option C). There is no `vision` start point. The run index lists the
  groups stage 1 never reached, and a resumed run or a resume dry run
  reports them as "not read yet". Reading them needs a new full run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-run research and fact check without re-reading labels (Priority: P1)

The owner has finished a full run. They change how search queries are built
or which search provider is used, then start a new run at the `research`
stage. The saved label readings and OCR text of every bottle are reused
as-is. Only research, fact check and render run again, so the results come
back in research-plus-fact-check time instead of two hours plus that.

**Why this priority**: This is the owner's stated goal. The vision pass costs
the most by far, and query tuning is the most frequent reason to re-run.

**Independent Test**: Run `process` on a small fixture set with fake vision,
search and text models. Then run again with `--from-stage research` and a
changed search setting. Check that the second run makes no vision requests
and no OCR calls, sends the changed queries, and saves complete schema-v2
bottle outputs whose label and OCR sources match the first run exactly.

**Acceptance Scenarios**:

1. **Given** a completed run whose stage-1 results are saved, **When** the
   owner runs `--from-stage research`, **Then** no photo is grouped, read by
   OCR or sent to the vision model, and every bottle goes through hint
   application, research, fact check and render.
2. **Given** the same run, **When** the owner changes the search provider or
   query settings and runs `--from-stage research`, **Then** the new research
   uses the current settings and the saved outputs record the new queries and
   pages.
3. **Given** `--from-stage research` without `-v`, **When** the run starts,
   **Then** the vision model is never loaded.
4. **Given** `--from-stage research` with `-v`, **When** the run reaches the
   vendor-image comparison, **Then** the vision model is used only for that
   comparison, as today.

---

### User Story 2 - Stage-1 results survive a failure in a later stage (Priority: P1)

During a long run, the fact check of batch 7 crashes, the text model runs
out of memory, or the owner presses Ctrl-C during interactive review. The
label readings and OCR text of every bottle whose stage 1 had finished,
including all of batch 7, are already on disk. The owner resumes from
`research` without losing any vision work.

**Why this priority**: Without durable stage-1 output, the other stories
can only resume runs that finished cleanly. A two-hour pass is exactly the
kind of work that gets interrupted.

**Independent Test**: Use fakes to make the research or fact-check stage
raise for one batch, or simulate an interrupt. Check that a stage-1 record
exists on disk for every bottle whose vision reading had finished, and that
`--from-stage research` completes those bottles with no vision requests.

**Acceptance Scenarios**:

1. **Given** a run where a bottle's vision reading (with its split check) has
   just finished, **When** the run moves on, **Then** that bottle's stage-1
   record is already on disk before any later stage of its batch starts.
2. **Given** a run interrupted during research, fact check or review of
   batch N, **When** the owner runs `--from-stage research`, **Then** every
   bottle of batches 1..N whose stage 1 finished is processed without a new
   vision reading.
3. **Given** a bottle whose vision reading failed, **When** stage 1 ends,
   **Then** the failure is recorded with its reason, the existing
   `error_report_<bottle_id>.json` is still written for `retry`, and a
   resumed run reports the bottle as not resumable instead of guessing a
   reading.

---

### User Story 3 - Re-run only the fact check after changing verification rules (Priority: P2)

The owner changes how claims are verified: a new quote-matching rule
(sibling 004), new validators (sibling 008), a different text model or a
different reasoning effort. They run `--from-stage factcheck`. The saved
label reading, OCR text and the web excerpts exactly as they were shown
before are reused. No web request is made and the vision model is not
loaded.

**Why this priority**: Verification rules change often while accuracy is
being tuned (see the accuracy evaluation harness in ToDo.md §2). Re-fetching
pages would also change the evidence and make before/after comparisons
meaningless.

**Independent Test**: After a full run with fakes, run `--from-stage
factcheck` with network access blocked and a different fake text model.
Check that the run succeeds, makes zero web requests, cites the same
evidence excerpts (same content hashes), and records the new text model.

**Acceptance Scenarios**:

1. **Given** a run whose research results are saved, **When** the owner runs
   `--from-stage factcheck`, **Then** no search or page request is made and
   the evidence excerpts the fact check sees are identical to the saved ones.
2. **Given** `--from-stage factcheck`, **When** the run starts, **Then** only
   the text model is loaded (none with `--no-verify`), and the vision model
   is not loaded.
3. **Given** `--from-stage factcheck` with `--no-verify`, **When** the run
   finishes, **Then** every bottle has the deterministic fact sheet from
   label, OCR and user facts only, as a full run with `--no-verify` would.
4. **Given** a hints file whose value for a bottle's name or release changed
   since the source run's research, **When** the owner runs
   `--from-stage factcheck`, **Then** that bottle is refused with the changed
   fields named and `--from-stage research` suggested, and the other bottles
   proceed.

---

### User Story 4 - Re-render outputs with no models and no network (Priority: P2)

The owner changes how the Markdown page, the batch preview or the output
files look (for example the reviewer contact sheet from sibling 009, or the
folder layout from sibling 006). They run `--from-stage render`. Every
bottle's saved fact sheet is rendered again with no model server, no model
and no network.

**Why this priority**: Rendering changes are cheap to make and should be
cheap to check. Today they need a full run.

**Independent Test**: With no model server running and network blocked, run
`--from-stage render` on a saved run. Check that it succeeds and that the
rendered facts, sources, citations and publish gate match the source run.

**Acceptance Scenarios**:

1. **Given** a run whose fact-check results are saved, **When** the owner
   runs `--from-stage render` with no model server reachable, **Then** the
   run succeeds without trying to start or contact one.
2. **Given** `--from-stage render`, **When** the run finishes, **Then** the
   field values, statuses, confidence, citations, conflicts, prices,
   verification and publish gate of each bottle equal the source run's. Only
   run identifiers, timestamps and lineage information differ.
3. **Given** `--from-stage render -i`, **When** the owner reviews a bottle,
   **Then** interactive review behaves as in a full run (edits re-reconcile
   with no model call, approvals are recorded).
4. **Given** a source run whose bottles were corrected and approved in
   interactive review and later with `feedback`, **When** the owner runs
   `--from-stage render`, **Then** every one of those corrections and
   approvals appears in the rendered outputs.

---

### User Story 5 - Know exactly where every resumed result came from (Priority: P2)

The output feeds a fact-based social account, so a reader of any bottle
output must be able to tell which stages ran in this run, which were loaded
from an earlier run, and which run and saved records they came from.

**Why this priority**: Provenance beats speed in this project. A resumed
output that looks like a fresh full run would hide what was actually
re-checked.

**Independent Test**: Resume from each stage and inspect the bottle JSON and
the run settings file. Each must name the stage resumed from, the source
run, the stages re-executed and the digests of the records loaded.

**Acceptance Scenarios**:

1. **Given** any resumed run, **When** a bottle is saved, **Then** its record
   names the source run, the stage resumed from, the stages loaded, the
   stages run again, and the digest of each saved record it used.
2. **Given** a saved record that was changed after it was written, **When** a
   resumed run loads it, **Then** that bottle is refused with a message that
   names the record, and the record's contents are never used as label or
   OCR evidence.

---

### User Story 6 - Preview a resume before running it (Priority: P3)

The owner runs `--from-stage research --dry-run` to see which run would be
resumed, how many bottles are usable, which are not and why, which stages
would run and which models would be needed. Nothing is written, no model is
loaded and no web request is made. As with today's `--dry-run`, the model
server is only probed, and only when a remaining stage needs a model.

**Why this priority**: It is a convenience. The same checks also run at the
start of a real resumed run.

**Independent Test**: Run `--dry-run` with each `--from-stage` value against
a saved run that has one changed photo and one failed bottle. Check the
report and that no model, network or file write happened.

**Acceptance Scenarios**:

1. **Given** a saved run, **When** the owner adds `--dry-run` to a resume,
   **Then** the report lists the source run, the counts of usable and
   unusable bottles with reasons, the stages to run and the models needed.
   No model is loaded or pulled, no web request is made and no file is
   written.
2. **Given** `--from-stage render --dry-run`, **When** it runs with no model
   server reachable, **Then** it succeeds without probing or starting one.

---

### Edge Cases

- **No saved results**: `--from-stage` with no source run that has the needed
  records fails before any work. The message names the output folder
  searched and the stage needed.
- **Missing later-stage records**: `--from-stage factcheck` on a run that
  never finished research (for example, it was interrupted during research)
  refuses those bottles. It suggests `--from-stage research` instead of
  running research quietly.
- **Research or fact check was off in the source run**: a source run made
  with `--no-web` saves a research record marked "skipped (web lookups
  disabled)" with no evidence (FR-005). `--from-stage factcheck` reuses it
  as-is, so the fact check sees label and OCR only, as in a `--no-web` full
  run. The run logs that the source had web lookups off and that adding web
  evidence needs `--from-stage research`. A source run made with
  `--no-verify` can be resumed at `factcheck` with the text model on.
- **Options for a stage that is loaded, not re-run**: for example
  `--search-provider` with `--from-stage factcheck`, or `--text-model` with
  `--from-stage render`. Such an option cannot take effect, so it is handled
  like the stage-1 options (FR-019). A different explicit command-line value
  is an error. A different value from a config file or environment variable
  gives one warning.
- **Hint value for an identity field changed since research**: the research
  record was gathered for the old search identity, for example a different
  name or release. `--from-stage factcheck` refuses that bottle, names the
  changed fields and suggests `--from-stage research` (FR-034). The fact
  check never judges evidence that was searched for a different product.
- **Photos moved, renamed or edited after stage 1**: each photo is checked
  against the content hash saved at stage 1. A photo that is missing or
  different makes that bottle fail with the reason "photo changed or missing
  since stage 1"; its old reading is never reused. This applies at every
  `--from-stage` value, including `render`, because the outputs copy the
  photos and must show the photos that were read. If the folder only moved,
  `-f` points to the new location and photos are matched by their path
  relative to the photo folder plus their hash. Without `-f` on the command
  line, the photo folder recorded by the source run is used.
- **Saved record edited by hand, truncated or unreadable**: the bottle is
  refused, and the message names the file. Corrections made by a person go
  through hints, `feedback` or interactive review, never by editing saved
  records, so a human edit is never passed off as a label reading.
- **Saved record from an unknown or newer record version**: refused with
  the version found and the versions supported.
- **Vision model or grouping options given together with `--from-stage`**:
  stage 1 is not re-run, so these values cannot take effect. An explicit
  command-line value that differs from the saved one is an error. A
  different value from a config file or environment variable gives one
  warning, and the saved value is used and recorded.
- **Hints file changed since stage 1**: hint field values (sourced
  corrections) are applied from the current hints file. A per-bottle
  reading note that differs from the one sent to the vision model at
  stage 1 cannot take effect without a new vision reading, so it gives a
  per-bottle warning saying so.
- **Hints file given at stage 1 but not now**: no hint values are applied,
  and the run logs that the source run used a hints file.
- **Bottle that the split check split into several bottles at stage 1**:
  each resulting bottle has its own stage-1 record. The split decision and
  its warnings are part of the record and are not recomputed.
- **Bottle whose stage 1 failed**: it is listed as not resumable with its
  saved failure reason. It stays available to `retry`.
- **Stage 1 interrupted before every group was read**: a resume processes
  the bottles that have a stage-1 record and reports the unreached groups,
  with their photos, as "not read yet". They are not failures and are not
  read by the resume. Reading them needs a new full run (FR-006).
- **Two runs both claim to be the most recent**: the run whose stage-1
  start time is latest wins. The time is recorded with its time zone, so
  runs from machines in different zones, or from both sides of a
  daylight-saving change, compare correctly. The chosen run is always
  logged. An explicit run identifier removes any ambiguity.
- **Resuming a resumed run**: a run that was itself resumed can be the
  source of a later resume (FR-036). For example, research is redone in
  run B from run A's stage-1 records, then run C resumes B at `factcheck`.
  C's lineage names the run that produced each record it used: A for
  stage 1 and B for research.
- **Bottles the source run already rendered**: by default a resumed run
  processes every usable bottle of the source run again, including bottles
  that were fully rendered before an interruption. The bottle list
  (FR-013) limits it to the bottles still needed.
- **Reviewer decisions in the source run**: corrections, approvals and
  rejections made in the source run's interactive review or with a later
  `feedback` are not re-applied when the fact check runs again (FR-030).
  The resumed run lists them per bottle so they are not lost silently.
  Bottles the source run rejected are skipped unless they are named in the
  bottle list (FR-035).
- **Run identifier that matches no saved run**: the run fails before any
  work and lists the run identifiers that are available.
- **Output folder that holds several runs**: only the chosen source run's
  records are read. Bottle records from other runs are never mixed in.
- **Interrupted resumed run**: the resumed run's own completed stage records
  are durable too, so it can itself be resumed.
- **Records written on macOS and resumed on Linux, or the reverse**:
  supported. Text is UTF-8, and photo locations do not depend on the
  platform's path conventions.
- **Web cache expired or cleared between runs**: `--from-stage research`
  fetches again under the current cache rules. `--from-stage factcheck`
  never needs the cache, because it uses the saved excerpts.

## Requirements *(mandatory)*

### Functional Requirements

**Stage-1 checkpoint (vision and OCR persisted immediately)**

- **FR-001**: The system MUST write each bottle's stage-1 record to durable
  storage as soon as that bottle's stage 1 (OCR, vision reading and
  same-bottle split check) has finished. This happens before any later
  stage of that bottle's batch starts, and the record does not depend on
  later stages succeeding.
- **FR-002**: A stage-1 record MUST hold:
  - the bottle ID and its photos (primary first), each with its location
    relative to the photo folder and its content hash;
  - the OCR text of every photo, exactly as the fact check would receive it
    as `O` sources, including any passage or segment boundaries the fact
    check relies on (see sibling 004);
  - any other value read from the photos themselves at stage 1, for
    example a barcode decoded by sibling 007, so that a resumed run never
    has to open the photos again for it;
  - the vision reading of every label field as the fact check receives it
    as the `L1` source: parsed from the model's reply, before any hint,
    correction or review;
  - the raw reply text of the vision request whose reading was used,
    unparsed. For a split half, this is that half's own reply. Today the
    reply is parsed and then discarded, so keeping it is new;
  - the vision model tag, the reported vision confidence, the same-bottle
    verdict, the number of vision attempts, and the warnings (including the
    split-check outcome);
  - the quantity count;
  - the hint reading note sent to the vision model, if any;
  - the settings that affect stage 1 (photo folder, grouping gap, maximum
    group size, split budget ratio, vision model);
  - the tool version, the run identifier and the time it was written.
- **FR-003**: A bottle whose stage 1 failed MUST get a stage-1 record that
  marks it as failed, with the failure reason and no reading. The existing
  failure report for `retry` MUST still be written.
- **FR-004**: Each run MUST keep an index of its bottle groups. The index
  says which groups finished stage 1, which failed and which were never
  reached, and records whether stage 1 finished for the whole run.
- **FR-005**: The stage-1 records MUST be written even when later stages are
  off (`--no-web`, `--no-verify`), so that a later run can add them. With
  `--no-web`, a research record marked as skipped (with the reason and no
  evidence) MUST still be written, so that `--from-stage factcheck` can
  re-run a label-and-OCR-only fact check.
- **FR-006**: An interrupted vision pass is not itself resumable. There is
  no `vision` start point, and no mechanism reads only the groups that
  stage 1 never reached. Instead:
  - the run index (FR-004) MUST list every group that stage 1 never
    reached, with its photos;
  - a run resumed from any stage MUST process only the bottles whose
    stage-1 record exists, and MUST report each unreached group as
    "not read yet" (with its photos) separately from failed bottles. These
    groups are not counted as failed (FR-023);
  - `--dry-run` with `--from-stage` MUST include the unreached groups in
    its report (FR-025).

  Reading the unreached groups needs a new full run.

**Later-stage checkpoints**

- **FR-007**: The research stage MUST save its result for each bottle before
  that bottle's fact check starts. The result holds the queries, every page
  consulted with its outcome, the evidence excerpts exactly as they will be
  shown to the fact check (with their content hashes), the search identity
  the queries were built from (label fields with hint values applied), the
  research settings in effect (provider and its address with secrets
  removed, query and page limits), the result of any other lookup that
  runs in this stage (for example sibling 007's registry lookup), and the
  vendor reference image decision when `-v` is on.
- **FR-008**: The fact-check stage MUST save each bottle's fact sheet
  (fields, sources, citations, conflicts, prices, verification, description,
  publish gate, review state) before that bottle's interactive review or
  rendering starts.
- **FR-037**: When interactive review in the same run, or a later `feedback`
  on one of its batches, changes a bottle's fact sheet (correction, conflict
  choice, approval, rejection), the latest reviewed fact sheet MUST become
  available to `--from-stage render` as a new saved version with its own
  digest. The version saved by the fact check MUST be kept, never
  overwritten.
- **FR-009**: Every saved stage record MUST carry a record version and a
  content digest, so a later run can detect any change, truncation or
  unsupported version.

**Resuming**

- **FR-010**: The `process` command MUST accept `--from-stage` with the
  values `research`, `factcheck` and `render`. Any other value MUST be
  rejected before any work starts, with a message that lists the valid
  values. Leaving the option out MUST behave exactly as today: a full run
  from stage 1.
- **FR-011**: `--from-stage` MUST be a command-line option only. It MUST NOT
  be settable in a config file or an environment variable, because a stored
  setting would quietly skip the vision pass on every later run.
- **FR-012**: The owner MUST be able to name the source run by its run
  identifier. When none is named, the most recent run in the output folder
  that has the records the chosen stage needs for at least one bottle MUST
  be used. "Most recent" means the latest stage-1 start time, compared with
  time zones. Its identifier and its usable-bottle count MUST be logged at
  the start of the run.
- **FR-013**: The owner MUST be able to limit a resumed run to a list of
  bottle IDs. If any listed ID is not in the source run, the run MUST stop
  before any work and list the unknown IDs, as a configuration error does.
  Bottles not listed are skipped and are not counted as failed.
- **FR-014**: `--from-stage research` MUST load the stage-1 records and then
  run hint application, research, fact check and render. It MUST NOT scan
  for or group photos, run OCR, make a vision reading or run the split
  check.
- **FR-015**: `--from-stage factcheck` MUST load the stage-1 and research
  records and then run hint application, fact check and render. It MUST
  make no search, page, robots or image request. The evidence the fact check
  sees MUST be byte-identical to the saved excerpts.
- **FR-034**: Before a `--from-stage factcheck` run uses a bottle's research
  record, it MUST compare the search identity in that record with the
  bottle's current search identity (label reading plus the current hint
  values). If they differ, the bottle MUST be refused with a reason that
  names the changed fields and suggests `--from-stage research`.
- **FR-016**: `--from-stage render` MUST load each bottle's latest saved fact
  sheet (FR-037, so including the source run's review and `feedback`
  outcomes) and render it, with interactive review first when `-i` is
  given. It MUST
  call no model, make no network request, and neither need nor start a
  model server.
- **FR-017**: A resumed run MUST load only the models its remaining stages
  call:
  - `research`: no model, except the vision model for the `-v` vendor-image
    comparison, and the text model for the fact check that follows;
  - `factcheck`: the text model only, or none with `--no-verify`;
  - `render`: none.

  The disk-space, memory and model-availability checks MUST cover only
  those models. The vision model MUST NOT be warm-loaded at start-up unless
  `-v` needs it. The OCR engine check MUST be skipped, because no OCR runs.
- **FR-018**: The stages that run again MUST use the current settings and
  the current tool version (search provider, web options, text model,
  reasoning effort, verification rules, output format). The stage-1 results
  MUST be used as saved.
- **FR-019**: Some options only affect a stage that the resumed run loads
  instead of re-running:
  - the vision model, the photo folder and the grouping options for stage 1;
  - the web options for research, under `factcheck` or `render`;
  - the text-model options for the fact check, under `render`.

  An explicit command-line value for such an option that differs from the
  value recorded for the loaded stage MUST be an error. The one exception is
  `-f` pointing to a moved photo folder (FR-021). A different value from a
  config file or environment variable MUST give a single warning, and the
  recorded value is used and recorded again in the lineage.
- **FR-020**: Hint field values MUST be applied from the hints file given to
  the resumed run, in the same way and with the same warnings as in a full
  run. When a bottle's hint reading note differs from the note saved in its
  stage-1 record, the run MUST warn for that bottle that the note cannot
  take effect without a new vision reading.
- **FR-021**: At every `--from-stage` value, before a bottle is processed,
  the system MUST check that each photo listed in its stage-1 record exists
  and still has the saved content hash. A bottle with a missing or changed
  photo MUST fail with a reason that names the photo, and none of its saved
  records may be used. `-f` on the command line MUST let the owner point to
  a moved photo folder. Photos are then matched by relative location and
  content hash. Without it, the photo folder recorded in the stage-1 record
  is used.
- **FR-022**: A saved record whose digest does not match, that cannot be
  read, or whose record version is not supported MUST cause that bottle to
  be refused, with a message that names the record. A resumed run MUST
  never treat the contents of such a record as evidence.
- **FR-023**: A bottle that cannot be resumed (stage 1 failed, a record
  missing for the chosen stage, a photo changed, a record invalid) MUST be
  counted as failed in the run summary, with its reason. It MUST NOT be
  processed from an earlier stage without the owner asking for that.
- **FR-024**: Batching, `--concurrency` for research, interactive review,
  failure reports, the batch preview and batch data read by `feedback`, the
  run summary and the existing exit codes MUST behave as in a full run. A
  resume that finds no usable source run MUST exit with code 3 (no input)
  and a message.
- **FR-025**: `--dry-run` together with `--from-stage` MUST report the source
  run, the counts of usable and unusable bottles with reasons, the stages
  to run and the models needed. It MUST NOT load, pull or warm a model,
  make a web request, or write any file. As with today's `--dry-run`, it
  MAY probe the model server, without starting it, to check the models
  needed. It MUST NOT probe at all when no remaining stage needs a model
  (`render`, or `factcheck` with `--no-verify`).

**Provenance and safety**

- **FR-026**: Every bottle output and the run settings file of a resumed run
  MUST record:
  - the stage resumed from;
  - the source run identifier;
  - which stages were loaded and which were run again;
  - the digest of each saved record that was used, and the run that
    originally produced it. This can be an earlier run than the source
    when the source was itself resumed.
- **FR-036**: A resumed run MUST keep its own run index. The index lists the
  records the run produced and the records it loaded, by producing run and
  digest. A resumed run can therefore itself be the source of a later
  resume, including after it was interrupted, without copying or changing
  the earlier runs' records.
- **FR-027**: In a resumed run, the label source (`L1`) and the OCR sources
  (`O1..On`) of each bottle's fact sheet MUST be identical to those in its
  stage-1 record.
- **FR-028**: A resumed run MUST NOT change or delete any saved stage
  record, run index, run settings file or batch data of its source run, or
  of any earlier run it loads from. Where its bottle outputs are written is
  governed by FR-029.
- **FR-029**: A resumed run MUST be a new run with its own run identifier
  and its own records, linked to its source run by its lineage (FR-026).
  Its bottle outputs MUST be written wherever the current output layout
  puts a re-saved bottle. Today that means new folders beside the old ones
  (never overwritten, `_2`/`_3` suffixes). Once sibling 006 lands, it means
  006's resolved bottle paths and its update rule. This feature defines no
  output layout of its own.
- **FR-030**: When a run resumes from `research` or `factcheck`, the fact
  sheets are rebuilt, so the source run's reviewer decisions (corrections,
  conflict choices, approvals, rejections from interactive review or
  `feedback`) MUST NOT be carried over by this feature. Every bottle needs a
  new approval to become publish-ready. Hint values are re-applied (FR-020).
  When a run resumes from `render`, the latest saved fact sheets (FR-037),
  including approvals and review state, MUST be rendered unchanged.
  Replaying reviewer decisions across runs is sibling 005's job. Once 005
  lands, a resumed run replays the reviewer ledger like any other run.
- **FR-035**: Until sibling 005 lands, a run resumed from `research` or
  `factcheck` MUST list, per bottle, every reviewer correction, approval or
  rejection recorded for that bottle in the source run that it is not
  re-applying, with the field and value, so that none is lost silently.
  Bottles the source run rejected MUST be skipped, and reported as skipped,
  unless they are named in the bottle list (FR-013).
- **FR-031**: Saved records MUST NOT contain API keys, passwords or other
  credentials. URLs are stored with secrets removed, as the run settings
  file does today.
- **FR-032**: Saved records MUST be readable on macOS and Linux no matter
  which platform wrote them.
- **FR-033**: The README, the quick-start and detailed docs, and
  `--help` MUST describe the stages, what each `--from-stage` value
  re-runs and loads, the records kept and where they are, and the
  provenance fields.

### Key Entities

- **Stage**: one of `vision`, `research`, `factcheck`, `render`, in that
  fixed order. A run executes a contiguous suffix of this order.
- **Stage-1 record (per bottle)**: the durable label reading (`L1`), the OCR
  text (`O` sources), the photo list with hashes, the vision metadata
  (model, confidence, same-bottle verdict, attempts, warnings, split
  outcome), the quantity, the reading note used, the stage-1 settings, the
  record version and digest. A failed bottle's record holds only the failure
  reason.
- **Research record (per bottle)**: the search identity used, the queries,
  every page consulted with its outcome, the evidence excerpts exactly as
  shown to the fact check, with their hashes, the vendor-image decision, the
  record version and digest. When web lookups were off, it records only
  that research was skipped and why.
- **Fact record (per bottle)**: the complete fact sheet that the render
  stage reads, with its record version and digest. The fact check saves the
  first version. Interactive review and `feedback` changes add later
  versions, and no version is overwritten.
- **Run index**: per run, the bottle groups and the latest stage each one
  completed, whether stage 1 finished for the whole run, the run's stage-1
  settings and start time (with time zone), and, for a resumed run, the
  records it loaded with their producing run.
- **Run lineage**: on a resumed run, the source run identifier, the stage
  resumed from, the stages loaded and the stages run again, and the record
  digests used with the run that produced each one. It is stored in the run
  settings file and in each bottle's output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After a full run on the reference photo set (98 photos, 55
  groups), a `--from-stage research` run makes 0 vision requests and 0 OCR
  operations. Its wall-clock time, without interactive review, is at most
  the source run's research, fact-check and render time plus 2 minutes of
  loading and checking overhead. That is about two hours less than a full
  run.
- **SC-002**: A `--from-stage factcheck` run makes 0 network requests and
  loads 0 vision models, which is verified with network access blocked.
- **SC-003**: A `--from-stage render` run of 55 bottles, without `-i`,
  finishes in under 1 minute with no model server reachable and no network
  access. This includes the photo checks and the image copies.
- **SC-004**: When a run is interrupted at any point after a bottle's stage 1
  finished, 100% of the finished stage-1 readings can be recovered by a
  resumed run, and none of them needs a new vision reading.
- **SC-005**: For 100% of bottles in a resumed run, the label and OCR source
  text in the fact sheet is identical to the saved stage-1 record.
- **SC-006**: A `--from-stage render` run on unchanged saved fact sheets
  produces fact data identical to the source run's, except for run
  identifiers, timestamps and lineage.
- **SC-007**: In tests, 100% of changed photos and changed, truncated or
  unsupported saved records are detected, and the affected bottles are
  refused with a message that names the file.
- **SC-008**: The owner can try a change to the search queries or the
  verification rules and see the new results for the whole reference set
  without waiting for a vision pass. The edit-to-result loop shrinks from
  more than 2 hours to the research and fact-check time alone.
- **SC-009**: In tests where the source run holds reviewer corrections,
  approvals and rejections, 0 of them are lost silently. A `render` resume
  shows 100% of them. A `research` or `factcheck` resume lists 100% of them
  in its report, and no bottle the source run rejected is output unless it
  is named in the bottle list.

## Assumptions

- **Existing behaviour this builds on (not re-specified):**
  - Capture-time grouping, bottle IDs, OCR of every photo, the vision
    reading and the same-bottle split check already happen in this order
    per batch. On the coordinator thread, the split check runs after the
    parallel first reading (`image_processor.batch_processor`). Stage 1 ends
    after the split check, which is the earliest point where a bottle's
    reading is final.
  - Research and fact check already run per batch after stage 1
    (`image_processor.enrichment.Enricher.enrich_batch`). The vision model
    is unloaded before the text model loads unless `--keep-models-loaded`
    is set.
  - Each bottle's label reading and OCR text are already kept in its
    metadata. This includes `ocr_by_image` and `full_text` in
    `models.bottle.ExtractedMetadata`. They already reach the fact check as
    the `L1` source (structured, not quotable) and the `O1..On` sources
    (`image_processor.fact_prompt`).
  - Today all of this reaches disk only when the batch is rendered
    (`PreviewGenerator.save_bottle_files`, `save_batch_preview`), after
    research, fact check and review. Nothing is saved between stages. That
    is the gap this feature closes.
  - Saved bottle outputs (schema version 2) already hold `metadata`,
    `web_research`, `facts` and `evidence/W<n>.txt`, the excerpts exactly
    as the fact check saw them, with their `content_sha256`. The research
    and fact records may reuse these outputs, as long as they hold
    everything the later stages need and meet FR-009.
  - The web cache keeps pages for 7 days and search results for 24 hours,
    and `--refresh-web` bypasses it. `--from-stage research` follows these
    rules unchanged.
  - Hint field values are applied after stage 1 as sourced corrections
    (`lib.hints.apply_hint_overrides`). Hint reading notes go into the
    vision prompt, so they are part of stage 1. Hint values also feed the
    search identity (`enrichment._hinted_identity`). Search queries are
    built deterministically from it (`web_research.build_queries`), and no
    text model is used in research. This is why FR-034 compares
    identities.
  - The vision model's reply text is parsed and then dropped
    (`BottleAnalyzer._extract_from_result`). The save path already writes
    a `vision_raw_response` key when a bottle has one, but nothing sets it.
    Keeping the raw reply (FR-002) is new capture, not existing behaviour.
  - Quantity is counted from byte-identical photos with no model
    (`BottleAnalyzer.detect_quantity`), right after each batch's vision
    pass. It belongs to stage 1.
  - Interactive review runs after the fact check, and `feedback` later
    rewrites a batch's data (`batch_<id>_preview.json`) and saves new
    bottle folders. Reviewer corrections become user facts on the fact
    sheet (`enrichment.user_facts_from_corrections`). This is why FR-037
    and FR-035 exist.
  - Today's `--dry-run` probes the model server, without starting it, to
    check the required models. FR-025 keeps that for the models a resume
    still needs.
  - `retry` re-runs one failed bottle from its error report, and `feedback`
    applies corrections and approvals to a saved batch without any model
    call. Both stay as they are.
  - The run settings file `run_<run_id>_settings.json` already records the
    effective settings with no API keys. FR-026's lineage is added to it.
- The two-hour figure comes from the owner's runs on the reference set. The
  exact time varies with hardware and model.
- Saved records hold text only, with no copies of images, so their size is
  small next to the photos. They are kept until the owner deletes them.
  There is no automatic pruning.
- Re-running stages after a resumed start is never fully deterministic,
  because web pages change and the text model samples. Only
  `--from-stage render` promises identical fact data (SC-006).
- Exit code 3 is reused for "no usable source run", because it already
  means "no input to process".
- **Where resumed outputs go (FR-029)** is resolved as a default rather
  than asked. A resumed run is a new run, and its bottle outputs follow
  whatever output layout is current. Today that is the never-overwrite
  rule. Later it is sibling 006's resolved paths. Siblings 005 and 006
  adopted the same rule. An earlier option, replacing the source run's
  folders in place, would have contradicted FR-028 and would decide 006's
  layout from inside this spec, so it was dropped.
- An unknown bottle ID in the bottle list stops the run before any work
  (FR-013). A typo should not quietly shrink an evaluation run.

## Dependencies

- **Sibling 004 (contiguous quote verification)**, **sibling 008 (spirits
  validators)**: these change the fact-check stage. With this feature they
  can be evaluated on a saved run through `--from-stage factcheck`. Neither
  depends on this feature to be built. 004 records per-photo OCR passage
  boundaries, so the stage-1 record must keep them (FR-002). 008's
  "reconcile stage" is part of this spec's `factcheck` stage. Validators
  run again on `research` and `factcheck` resumes, and not on a `render`
  resume, which renders the saved fact sheet unchanged.
- **Sibling 005 (reviewer override ledger)**: owns carrying reviewer
  corrections, approvals and rejections across runs. 005 requires resumed
  and render-only runs to replay the ledger. FR-030 defers to it. Until it
  lands, FR-035 reports what is not re-applied, and resumed runs from
  `research` or `factcheck` start unapproved. Both specs change the
  `process`/`feedback` orchestration in `cli/commands.py`. FR-037 (review
  and `feedback` add fact-record versions) touches the same `feedback`
  path that 005 changes.
- **Sibling 006 (bottle identity layout)**: owns the output folder layout,
  resolved paths and the update-in-place rule (its FR-024). FR-029 follows
  006 and defines no layout. FR-028 protects only stage records, indexes,
  settings files and batch data, so it does not conflict with 006 updating
  a bottle's folder in place. **Boundary gap to resolve**: 006 lists
  "skipping already-completed bottles on re-run, `--force`/`--reprocess`
  and checkpointing" as owned by 003. This spec covers checkpointing and
  stage resume only. Skipping completed bottles in a fresh full run
  (ToDo.md §4) is out of scope here (see Out of Scope), so 006 or a new
  item needs to own it. Expect a merge conflict in `preview_generator` and
  in the README's output section.
- **Sibling 007 (barcode and TTB COLA lookup)**: 007 leaves stage names and
  checkpoint contents to this spec, and no new stage names are added. The
  four stages stay fixed. Decoding a barcode from the photos is part of
  `vision` (stage 1), and the decoded value goes in the stage-1 record
  (FR-002). The registry lookup is network work in `research`, and its
  result goes in the research record (FR-007). So `--from-stage research`
  repeats the lookup but never re-reads the photos, and `--from-stage
  factcheck` repeats neither.
- **Sibling 009 (review contact sheet)**: a render-stage output. It is
  rebuilt from the saved photos by `--from-stage render`. 009 makes the
  rebuild deterministic.
- **Sibling 010 (social export)**: 010 treats the export as a separate
  read-only step over saved outputs, run only with `--export social`. It
  is not part of the render stage. `--from-stage render` refreshes the
  outputs the export reads. Whether `--export social` may be combined with
  `--from-stage render` in one command is 010's decision.
- **Sibling 011 (SearXNG local search)**: a research-stage provider.
  Switching to it is the main use of `--from-stage research`. Its
  run-settings fields (address source, redacted address) are part of the
  research settings recorded in FR-007 and compared in FR-019.
- **ToDo.md §2 "Recorded web-evidence replay"**: the research record and
  `--from-stage factcheck` cover the replay part of the accuracy harness.
  They re-run the fact check offline over exactly the evidence of one live
  run. The text model still samples, so only the evidence is fixed, not
  the model's answer.
- **ToDo.md §3 "`verify` subcommand"**: this feature covers the core of that
  item. `--from-stage factcheck` re-runs the fact check without the vision
  step, and `--from-stage research` stands in for `--relookup`. The ToDo
  entry should be updated or closed when this feature lands.
- **ToDo.md §3 "LLM response cache"**: complementary. A cached fact-check
  reply would make `--from-stage factcheck` cheaper for unchanged evidence.
  It is not required here.
- **ToDo.md §4 "A re-run skips nothing"**: related but separate. That item
  skips bottles that are already done in a new full run. This feature
  resumes a known run from a chosen stage.

## Out of Scope

- Stopping a run after a chosen stage (for example, a vision-only overnight
  run). Stage-1 records are written anyway, so an interrupted or abandoned
  run can still be resumed.
- Re-running stage 1 for only some bottles of a saved run, including the
  groups an interrupted stage 1 never reached. `retry` already covers
  failed bottles; unreached groups are reported, not read (FR-006).
- Editing saved records by hand as a way to correct values. Corrections go
  through hints, `feedback` or interactive review.
- Automatic pruning or garbage collection of saved records.
- Skipping bottles that are already done in a fresh full run, and the
  `--force`/`--reprocess` flags (ToDo.md §4). A full run without
  `--from-stage` behaves as today (FR-010). The bottle list (FR-013) is the
  only way to narrow a resumed run.
- Choosing an output layout or a folder update rule. That belongs to
  sibling 006 (FR-029).
- Re-grouping photos, or changing the split decision, from saved records.
- A separate subcommand. Resuming is an option of `process`.
