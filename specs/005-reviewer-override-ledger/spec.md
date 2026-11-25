# Feature Specification: Append-Only Reviewer Ledger

**Feature Branch**: `005-reviewer-override-ledger`

**Created**: 2026-11-24

**Status**: Draft

**Input**: User description: "Append-Only Reviewer Ledger (overrides.jsonl / overrides.toml): Store all human review actions (RETRACT, SET, APPROVE) as an immutable log keyed by bottle_id rather than mutating generated JSON files directly. Replay the ledger during reconciliation and rendering so reruns never overwrite manual edits."

## Context: what exists today

This feature changes *where human review decisions live*, not how a person makes them. Grounding (as of `feat/web-lookups-config` @ 9002167):

- **Review entry points already exist.** `process -i` and `feedback -i` offer `[A]ccept`, `[R]eject`, `[E]dit`, `[C]onflicts`, `[N]ext`, `[Q]uit` per bottle (`PreviewGenerator.interactive_prompt`). `feedback --corrections fix.json` applies a JSON list of corrections; `feedback --approve <bottle_id>` (repeatable) approves a bottle. `retry --report <error report>` applies the `suggested_fix` field values a person typed into a generated error report (`commands._retry_bottle`): this is a fourth review entry point, and it is itself a hand edit of a generated JSON file.
- **Interactive review is all-or-nothing today.** `interactive_prompt` collects every decision and returns them only when the review ends; if the review fails (for example end of input), `_interactive_review` applies nothing and approves no bottle, and `feedback -i` exits 1.
- **Approvals record an entry point, not a person.** `facts.review` holds `{approved, by, at}`, where `by` is `interactive` or `feedback`. An approval is withdrawn when any of the 12 fact field values (supported or user-stated) changes afterwards (`enrichment.refresh_facts` compares `FactSheet.flat_values`); the generated description is not part of that comparison, but the publish gate separately requires a verified, non-stale description.
- **How edits persist today.** An edit or conflict choice becomes a `Correction` (bottle_id, field, the value the person saw, the new value, time, source) that is applied in memory (`image_processor.corrections.apply_corrections`), appended to the bottle's `corrections` list, and folded into the fact sheet as a `user` fact (`enrichment.refresh_facts`). An approval is written into the fact sheet's `review` block (`enrichment.approve_facts`) and the publish gate is recomputed. `feedback` then **rewrites the batch JSON** (`batch_<id>.json`) in place and saves every changed bottle to a **new `_N` output folder**. Rejections are listed in the batch JSON's `rejected` list.
- **Consequence (the problem).** Human decisions exist only inside generated output files. A fresh `process` run over the same photos starts from the model's readings again: every edit, conflict choice and approval is lost, and nothing records that a person ever made them. There is no way to withdraw a value without typing a replacement, and no history of who changed what. (A rerun never overwrites an old output folder, because every save picks a new `_N` folder, but the new folder is built without the human decisions.)
- **Related existing input.** The `--hints` TOML file (`lib.hints`) already lets the owner pre-declare per-bottle field values with a required `source` (10 fields: the correctable fields except `mash_bill` and `tasting_notes`); they are applied after the vision step as `user` facts, a disagreement with the label reading is logged as a review warning naming both values, and the fact sheet keeps it as a resolved conflict. The hints file is read-only input written by the owner; it is not a log of review actions.
- **Existing safeguards to keep.** Corrections are applied by bottle ID only, never by matching a value; only the correctable fields (`name`, `distillery`, `bottler`, `specific_release`, `batch_number`, `proof`, `abv`, `age`, `release_year`, `mash_bill`, `msrp`, `tasting_notes`) may be changed; an ambiguous (duplicate) bottle ID changes nothing; a value changed after approval withdraws the approval; an approval never clears other publish blockers.

What this feature adds: a single append-only ledger of reviewer actions, keyed by bottle ID, that is the system of record for human decisions, plus replay of that ledger wherever facts are reconciled and outputs are rendered, so that any rerun reproduces every human decision without anyone editing generated files.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Manual edits survive a full rerun (Priority: P1)

The owner reviews a batch, corrects a batch number, settles a proof conflict and approves the bottle. Weeks later they re-run `process` on the same photos (new model, better OCR, fresh web research). The corrected values and the conflict choice are applied again automatically, with their original reviewer, time and reason, and the Markdown/JSON outputs show them as reviewer-stated (`user`) values.

**Why this priority**: This is the core promise in the owner's text ("reruns never overwrite manual edits"). Without it every rerun silently discards human work, and a later post on the social account could repeat a value a person had already corrected.

**Independent Test**: Record a SET for one field of one bottle, delete every generated output (keeping the ledger), run `process` again on the same photos with all models faked to return a different value; the rendered output must show the reviewer's value, cite the ledger entry, and keep the machine's differing value as a resolved conflict.

**Acceptance Scenarios**:

1. **Given** a ledger SET of `batch_number = "C923"` for bottle `IMG_4643`, **When** `process` runs again and the vision model reads `C92`, **Then** the bottle's `batch_number` is `C923` with status `user`, the citation names the ledger entry (reviewer, time, reason), and the disagreement (`C92` from the label reading) is kept as a resolved conflict.
2. **Given** a reviewer edits a field in `process -i`, **When** the reviewer finishes that bottle, **Then** a SET entry is appended to the ledger before the next bottle is shown, and the generated JSON is produced from the replayed state rather than being the only record of the edit.
3. **Given** a `feedback --corrections fix.json` run, **When** the corrections are accepted, **Then** each becomes a SET entry whose source names the corrections file, and no generated batch or bottle JSON is edited in place as the record of the change.
4. **Given** a SET entry for a bottle ID that the current run never produced, **When** the run finishes, **Then** the entry is reported as unmatched (bottle ID and entry ID) and stays in the ledger untouched.

---

### User Story 2 - Approvals are replayed but never outlive the values they approved (Priority: P1)

The owner approves a bottle for publishing. On a rerun, if every field value the approval covered is the same, the bottle is approved again automatically. If any approved value changed (new web evidence, a different reading), the approval is shown as stale and the bottle is not publish-ready until a person approves it again.

**Why this priority**: The social account must only receive values a person actually approved. Replaying an approval onto different values would publish something no one checked; dropping all approvals on every rerun would force the owner to re-approve the whole collection each time.

**Independent Test**: Approve a bottle, rerun with identical faked model output (approval re-applies, bottle publish-ready if no other blockers), then rerun with one field changed (approval reported stale, bottle not publish-ready, blocker names the changed field).

**Acceptance Scenarios**:

1. **Given** an APPROVE entry recorded when the bottle's approved content (its 12 fact field values, Unknown included, and its verified description text) was V, **When** a rerun produces exactly V, **Then** the approval is applied with its original reviewer and time and the publish gate is recomputed (other blockers still apply).
2. **Given** the same APPROVE entry, **When** a rerun produces a fact field value that differs from V, **Then** the approval is not applied, the bottle's blockers include "approval stale: <field(s)> changed since approval on <date>", and the bottle is not publish-ready.
3. **Given** the same APPROVE entry, **When** a rerun keeps every field value but the fact check writes a different description text, **Then** the approval is stale with "description changed since approval on <date>", because the reviewer never saw the new text.
4. **Given** an APPROVE entry followed later by a SET on the same bottle, **When** the ledger is replayed, **Then** the approval is stale (the reviewer approved different values), matching today's rule that a change after approval withdraws the approval.
5. **Given** `feedback --approve <bottle_id>` or `[A]ccept`, **When** the action is taken, **Then** an APPROVE entry is appended that records the values being approved, the reviewer and the time.
6. **Given** an applied APPROVE that the reviewer now regrets, **When** they withdraw that entry by its entry ID (a RETRACT that names the entry), **Then** on replay the bottle is not approved and both entries remain in the history.

---

### User Story 3 - Retract a value without inventing a replacement (Priority: P2)

The owner sees a value that is wrong or unverifiable (for example an MSRP taken from a page that turned out to be a different release) but has no correct value to type in. They retract it. On this and every later run the field is shown as Unknown with the reviewer's reason, even if the machine finds the same value again, until a person sets a value or withdraws the retraction.

**Why this priority**: "Unknown + reason beats a guess" is a project rule, but today the only way to remove a bad value is to type a different one. RETRACT closes that gap. It ranks below P1 because SET and APPROVE cover the most common review actions.

**Independent Test**: Retract `msrp` on a bottle, rerun with the fake web research again returning the same MSRP; the output shows MSRP Unknown with the retraction reason, and the bottle's total value is unknown.

**Acceptance Scenarios**:

1. **Given** a RETRACT of `msrp` for bottle B with reason "W2 is the 2022 release", **When** the ledger is replayed, **Then** B's `msrp` is Unknown, the reason names the reviewer's retraction, and the machine-found value is kept only in the bottle's audit record, never as the field's value.
2. **Given** a RETRACT followed later by a SET on the same field, **When** the ledger is replayed, **Then** the SET value applies (the later entry wins) and the retraction remains visible in the history.
3. **Given** a RETRACT that targets an earlier ledger entry by its entry ID (for example a mistaken SET), **When** the ledger is replayed, **Then** the targeted entry's effect is no longer applied (the field goes back to what the machine sources and any earlier still-applied entries say, rather than becoming Unknown), and both entries remain in the ledger and in the history.
4. **Given** a reviewer in interactive review, or a corrections file, **When** they withdraw a field value, **Then** they must give a reason, and an entry with no reason is refused and nothing is appended.

---

### User Story 4 - See and audit every human decision for a bottle (Priority: P2)

Before posting, the owner (or a reviewer helping them) wants to know who changed what on a bottle, when, why, and whether each action is still in effect. The bottle's Markdown page and JSON show a "Reviewer actions" history, and a command lists the ledger entries for one bottle or for the whole collection.

**Why this priority**: Provenance beats speed for this project; a fact-based account must be able to answer "where did this value come from?" for reviewer-stated values as well as for web and label values.

**Independent Test**: Append SET, RETRACT and APPROVE entries for one bottle; render it; the page lists all three in order with reviewer, time, reason, and status (applied / superseded / withdrawn / stale / unmatched).

**Acceptance Scenarios**:

1. **Given** a bottle with ledger entries, **When** its outputs are rendered, **Then** each reviewer-stated field cites the ledger entry that set it, and the page lists every entry for that bottle with its current status.
2. **Given** the ledger, **When** the owner asks for the history of one bottle ID, **Then** every entry for that ID is shown in append order with its status, without running any model.

---

### User Story 5 - Detect a damaged or hand-edited ledger (Priority: P3)

The ledger is the record of human decisions behind public posts, so an accidental edit, deletion, reordering or truncated write must be noticed instead of silently changing what was approved.

**Why this priority**: Integrity protection matters, but it guards against rare events; the feature delivers value without it.

**Independent Test**: Write a valid ledger, then alter one entry's value, delete one entry, and truncate the last line in three copies; each copy is refused before any model work, with the line number and the kind of damage named.

**Acceptance Scenarios**:

1. **Given** a ledger in which an earlier entry was edited, removed or reordered, **When** any command loads it, **Then** the command stops before model work (exit code 1), naming the first damaged entry.
2. **Given** a ledger whose final line was cut off by a crash, **When** it is loaded, **Then** the partial line is reported and the command stops; the tool never silently drops or rewrites entries.
3. **Given** a ledger whose last whole entry (or entries) was deleted, **When** a command loads it and the most recent run record for that ledger names a longer ledger, **Then** the command stops and reports the missing tail; with no run record to compare against, this case cannot be detected and the spec does not claim it.

---

### Edge Cases

- **Bottle ID no longer exists or changed** (photo renamed, regrouped, or a duplicate-stem hash changed because photos moved): entries stay in the ledger and are reported as unmatched; they are never applied to a different bottle by value matching. Stable IDs across reruns depend on sibling spec 006.
- **Duplicate bottle ID in one run** (two groups share an ID): no ledger entry for that ID is applied (same rule as today's corrections); the run reports it.
- **SET whose "value seen" no longer matches** the machine value on a rerun: the reviewer's value still wins (the person established it), and the machine's different value is kept as a resolved conflict naming both, as hints do today. This intentionally differs from today's one-shot corrections, which refuse a stale `original_value`.
- **SET whose "value seen" does not match at the time it is recorded** (for example a corrections file written against an older batch): the action is refused and nothing is appended, as today's corrections refuse a stale `original_value`; this catches a file aimed at the wrong batch before it becomes a standing decision.
- **SET to a field that is not correctable**, or a value that fails the field's validity rules (e.g. proof not above 0 or above 200, a non-4-digit release year, empty text): the entry is refused at the time it is recorded and nothing is appended.
- **Hand-added ledger lines**: the ledger is written only by the tool. A line added or changed by hand fails the integrity check (FR-021) and stops the run before model work; standing hand-written overrides belong in the hints file (or `overrides.toml`, see Q2), not in the ledger.
- **REJECT** (`[R]eject` today): recorded in the ledger as a bottle-level action so a rerun does not quietly bring back a bottle the owner rejected; the rejected bottle still gets a failure report and is not rendered as an output bottle. A later APPROVE of the same bottle, or a RETRACT naming the REJECT entry, lifts the rejection.
- **RETRACT of a core field** (name, distillery/bottler, proof/ABV): the field becomes Unknown, so the existing core-field blocker makes the bottle not publish-ready; the retraction never bypasses the gate.
- **Error-report `suggested_fix` used by `retry`**: the values are recorded as SET entries (origin: the report path) when `retry` applies them, so a later `process` run keeps them.
- **Two commands append at once** (for example `process -i` in one terminal and `feedback --approve` in another): both entries are recorded whole and in a single order; neither is lost or interleaved.
- **Interactive review aborted** (end of input, Ctrl-C): actions for bottles the reviewer finished are already in the ledger; actions on the bottle in progress are discarded, and no approval is ever recorded for an unfinished bottle. This deliberately changes today's all-or-nothing behaviour, where an aborted review applies nothing at all. A bottle counts as finished when the reviewer chooses `[A]`, `[R]`, `[N]` or `[Q]`; edits made with `[E]`/`[C]` before `[N]` or `[Q]` are recorded, as they are applied today.
- **Generated outputs deleted before a rerun**: the ledger is not a generated output. No tool command deletes, moves or rewrites it, and its default location is the top of the output folder, never inside a bottle or batch folder, so deleting bottle folders and batch files keeps it.
- **No ledger exists yet**: behaviour is identical to today for a run with no review actions; the ledger is created on the first recorded action.
- **Hints file and ledger both set the same field**: [NEEDS CLARIFICATION (Q2): How does `overrides.toml` relate to the existing `--hints` TOML file and to the append-only `overrides.jsonl`? Options: (A) an owner-written file of standing overrides read beside the ledger, with hint `[fields]` unchanged; (B) it replaces the hints file's `[fields]` part, with a migration; (C) a generated, read-only view of the effective overrides; (D) the owner meant one log in either format, so only `overrides.jsonl` exists.] Until decided, the default precedence is ledger over hint over machine sources, with every disagreement kept as a resolved conflict.
- **Same SET recorded twice** (the reviewer re-enters a value already in effect): the second entry is still appended (it is a real action) and the first is shown as superseded; the value does not change, so an approval given between them is not made stale.
- **Existing schema-v2 outputs** already holding corrections and approvals made before this feature: they can be imported into the ledger once (see FR-019); importing twice adds nothing.
- **Very long ledger** (years of reviews): replay cost grows with the number of entries, not with model work; no model is called to replay.

## Requirements *(mandatory)*

### Functional Requirements

**Recording actions**

- **FR-001**: The system MUST record every human review action as one entry appended to a single append-only ledger; recorded entries MUST never be modified or removed by the tool.
- **FR-002**: The ledger MUST support these actions: **SET** (a field of a bottle takes a reviewer-stated value), **RETRACT** (either a field's value is withdrawn and shown as Unknown with the reviewer's reason, or an earlier ledger entry named by its entry ID is withdrawn so its effect is no longer applied), **APPROVE** (the bottle is approved for publishing), and **REJECT** (the bottle is excluded from output, as `[R]eject` does today).
- **FR-003**: Every entry MUST carry: a unique entry ID, the bottle ID it applies to, the action, the field (for SET and a field RETRACT), the targeted entry ID (for an entry RETRACT), the new value (for SET), the value the reviewer saw (for SET and a field RETRACT), the reviewer identity, the time (UTC), the origin (interactive review, corrections file path, `--approve` flag, retry error report path, import), and a free-text reason (required for RETRACT and REJECT, optional otherwise).
- **FR-004**: An APPROVE entry MUST record the content the reviewer approved: the value of every fact field (all 12, with Unknown recorded as Unknown) and the verified description text, as they stood at that moment, so that a later replay can tell whether the approved content is still what would be published.
- **FR-005**: Every existing review entry point MUST write to the ledger instead of making generated files the record of the decision: `process -i` and `feedback -i` (`[A]`, `[R]`, `[E]`, `[C]`), `feedback --corrections FILE`, `feedback --approve BOTTLE_ID`, and the `suggested_fix` values that `retry` applies from an error report.
- **FR-005a**: The reviewer MUST be able to record a field RETRACT from interactive review and from a corrections file, and to withdraw an earlier entry by its entry ID without interactive review. A RETRACT without a reason, or one naming an entry ID that does not exist or belongs to another bottle, MUST be refused and MUST NOT be appended.
- **FR-006**: In interactive review, the actions for a bottle MUST be appended when the reviewer finishes that bottle (chooses `[A]`, `[R]`, `[N]` or `[Q]`), so an aborted review keeps every finished bottle's decisions and records nothing for the bottle in progress. This replaces today's all-or-nothing behaviour.
- **FR-007**: SET and RETRACT MUST be limited to the existing correctable fields, and SET values MUST pass the same validity rules the hints file applies today (proof above 0 and at most 200, ABV above 0 and at most 100, MSRP above 0 and at most 100,000; a four-digit release year; non-empty text for every other field, including `mash_bill` and `tasting_notes`, which hints cannot set). A SET or field RETRACT whose value seen differs from the bottle's current value at the moment it is recorded MUST be refused, as stale corrections are today. An invalid action MUST be refused with a message and MUST NOT be appended.
- **FR-008**: Concurrent appends from separate commands MUST each be recorded completely and in a single, well-defined order.

**Replaying actions**

- **FR-009**: Every run that reconciles facts or renders outputs (`process`, `retry`, `feedback`, and any later re-render or export stage) MUST load the ledger and replay its entries for the bottles in that run, with no model call needed for the replay.
- **FR-010**: Replay MUST match entries to bottles by exact bottle ID only. Entries for IDs not present in the run, or for IDs that are ambiguous in the run, MUST NOT be applied and MUST be reported (entry ID, bottle ID, reason).
- **FR-011**: Entries MUST be replayed in append order; for the same bottle and field the latest SET or RETRACT determines the outcome, and earlier entries remain in the ledger and in the rendered history as "superseded".
- **FR-012**: A replayed SET MUST become a reviewer-stated (`user`) fact that wins over label, OCR, web and hint values; whenever any other source holds a different value, that disagreement MUST be kept as a resolved conflict naming both values.
- **FR-013**: A replayed field RETRACT MUST make the field Unknown with a reason naming the reviewer, time and stated reason, MUST prevent any machine source from re-populating it on later runs, and MUST keep any machine value found for it in the bottle's audit record, never as the field's value. A replayed entry RETRACT MUST stop the named entry from being applied; the bottle is then replayed as if that entry had never been recorded, and the named entry is shown as "withdrawn".
- **FR-014**: A replayed APPROVE MUST be applied only when, after replaying every SET and RETRACT, the bottle's 12 fact field values and its verified description text equal those recorded in the APPROVE entry; otherwise the bottle MUST carry a blocker "approval stale" naming the changed fields (or "description") and the approval date. An applied approval MUST NOT clear any other publish blocker.
- **FR-015**: A replayed REJECT MUST exclude the bottle from rendered output bottles and write the same failure report a rejection writes today, unless a later APPROVE of the same bottle, or an entry RETRACT naming the REJECT, supersedes it.
- **FR-016**: Ledger replay MUST happen before the publish gate is computed and before any output (bottle Markdown and JSON, batch preview, and any downstream view such as a contact sheet or social export) is rendered, so every rendered artifact reflects the replayed state.
- **FR-017**: `feedback` MUST NOT rewrite the batch JSON or any bottle JSON in place as the way it records a review decision; it MUST append to the ledger and then re-render the affected bottles from the replayed state. Re-rendered bottles are written where the output layout puts them: until sibling 006 defines resolved output paths, today's rule applies (a new folder, never overwriting an existing one). Any reader of a batch (a later `feedback`, the contact sheet, the social export) MUST take approvals and rejections from replay, not from the batch JSON's `rejected` list or a saved `facts.review` block alone.

**Visibility and integrity**

- **FR-018**: Each rendered bottle (Markdown and JSON) MUST list its ledger entries in append order with reviewer, time, action, field, value, reason and status (applied, superseded, withdrawn, stale, unmatched), and every reviewer-stated field MUST cite the ledger entry that set it.
- **FR-019**: The system MUST provide a one-time, repeatable-without-duplicates import that turns reviewer decisions already stored in existing schema-version-2 outputs into ledger entries with origin "import" and their original times: interactive and corrections-file corrections become SETs, `facts.review` approvals become APPROVEs (snapshot taken from the saved values), and each batch's `rejected` list becomes REJECTs. Corrections whose source is a hints file MUST NOT be imported (the hints file stays their record). Because today's outputs record only the entry point (`interactive`, `feedback`), not a person, imported entries MUST say the reviewer is not recorded rather than guess one.
- **FR-020**: The owner MUST be able to list the ledger entries for one bottle ID, or for all bottles, with each entry's current status, without running any model.
- **FR-021**: The ledger MUST be tamper-evident: each entry MUST be linked to the entry before it so that an edited, removed (other than the last), reordered or partly written entry is detected on load. Each run MUST also record the ledger's entry count and last entry ID in its effective settings record (FR-022), and a load MUST compare against the most recent such record it finds for that ledger, so that removal of the final entries is detected whenever a run record exists. A damaged ledger MUST stop the command before any model work with exit code 1, naming the first damaged entry; the tool MUST NOT repair or rewrite it automatically.
- **FR-022**: The ledger's location MUST be configurable through the existing settings layers (settings file, environment variable, command-line flag), defaulting to `overrides.jsonl` at the top of the output folder, and the run's effective settings record MUST name the ledger used. No command may delete, move or rewrite the ledger.
- **FR-023**: The run summary MUST report how many ledger entries were applied, superseded, withdrawn, stale and unmatched.
- **FR-024**: The ledger MUST be plain UTF-8 text with one entry per line, readable and diff-able on macOS and Linux without this tool.

### Key Entities

- **Ledger**: The ordered, append-only record of all reviewer actions for a collection. Attributes: location, entries in append order, integrity state (intact / damaged at entry N).
- **Ledger Entry**: One reviewer action. Attributes: entry ID, bottle ID, action (SET, RETRACT, APPROVE, REJECT), field, new value, value seen, approved-values snapshot (APPROVE only), reviewer, time, origin, reason, link to the previous entry. For an entry RETRACT, the ID of the earlier entry it withdraws.
- **Bottle ID**: The identifier the ledger is keyed by: today the run-assigned ID (first photo's file stem, with a short path hash when two groups share a stem). Its long-term stability is owned by sibling spec 006.
- **Replay Result**: Per bottle, the effect of the ledger on one run: applied SETs and RETRACTs, approval state (applied / stale with changed fields or description / withdrawn / none), rejection, unmatched and superseded entries. It feeds the fact sheet, the publish gate and every rendered view.
- **Overrides file (`overrides.toml`)**: A TOML form named by the owner; its exact role (hand-authored standing overrides, generated readable view, or just an alternative format) is pending clarification Q2.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a rerun test over at least 10 bottles carrying SET, RETRACT, APPROVE and REJECT actions, with every generated output (everything except the ledger) deleted beforehand, 100% of reviewer actions are reflected in the new outputs and 0 are lost.
- **SC-002**: 0 generated output files need to be edited by hand (or by the tool in place) to record a review decision; every decision can be traced to exactly one ledger entry.
- **SC-003**: 0 approvals are applied to a bottle whose field values or description differ from what the reviewer approved; in a test that changes one approved field or the description text, 100% of affected bottles are reported "approval stale" and are not publish-ready.
- **SC-004**: For every reviewer-stated or retracted value on a rendered bottle, the owner can identify who made the decision, when, and why from the bottle's page alone, in under 1 minute.
- **SC-005**: In a damage test (one edited entry, one deleted middle entry, one reordered pair, one truncated final line, and one deleted final entry with a run record present), 5 of 5 damaged ledgers are detected and refused before any model work.
- **SC-006**: Re-applying all reviewer decisions to the owner's full collection (about 55 bottles) requires no model calls and adds less than 5 seconds to a run.
- **SC-007**: Running the import twice on the same existing outputs adds entries only on the first run (0 duplicates on the second).

## Assumptions

- **Single reviewer, local machine.** The owner is the main reviewer; reviewer identity defaults to the operating-system user name and can be overridden by a setting. No authentication or multi-user permissions are in scope. Reviewer identity and reasons are private to the owner's files: they appear in the ledger and on local bottle pages, never in public exports (sibling 010 already excludes reviewer names).
- **Ledger file name and default location.** The append-only log is `overrides.jsonl` (owner's name), stored by default in the output folder next to the batch files; it is configurable (FR-022) so it can live outside a disposable output folder.
- **REJECT is included** although the owner's text lists only RETRACT, SET and APPROVE: `[R]eject` is an existing human review action, and leaving it out would let a rerun silently resurrect a rejected bottle, contradicting "reruns never overwrite manual edits".
- **A reviewer SET wins over a stale "value seen".** Today's corrections refuse to apply when the bottle no longer holds the value the person saw. For replay, the person's stated value is the established fact, so it wins, and the disagreement is surfaced as a resolved conflict rather than silently dropped. The "value seen" is kept for audit.
- **Approval binds to what was approved, not to the run.** An approval is re-applied on a rerun only when the approved field values and description text are unchanged (FR-014). For field values this preserves today's rule (`enrichment.refresh_facts` withdraws an approval when a value changes after approval). Adding the description text is stricter than today: a full rerun regenerates the description with the text model, so a reworded description makes the approval stale and the owner re-approves it. This follows the accuracy-first rule (the social account must only carry text a person saw); a render-only rerun (sibling 003) that keeps the description keeps the approval.
- **RETRACT covers both targets** (resolves the earlier open question): a field RETRACT gives the owner's "Unknown + reason" outcome, and an entry RETRACT is the only way to undo a mistaken SET, APPROVE or REJECT in an append-only log (a later SET cannot bring back the machine value, and there is no separate un-approve action). Supporting both adds no new action name.
- **Where re-rendered bottles are written** (resolves the earlier open question): the ledger never depends on the output location. Until sibling 006 defines resolved output paths, re-rendering follows today's never-overwrite rule (a new folder); afterwards it follows 006.
- **`[N]ext` and `[Q]uit` are not recorded as actions** (they are navigation, not decisions); they only end the bottle, so its `[E]`/`[C]` edits are appended.
- **Hints stay as they are** (`--hints` notes for the vision prompt and sourced `[fields]`), pending Q2; they remain a read-only owner input and are not written to by the tool.
- **The existing `corrections` list and `facts.review` block** in the bottle JSON remain in schema v2 as rendered results of replay; adding ledger references to them is a planning decision. If the schema must change, the schema version and the `feedback`/`retry` readers change with it.
- **Domain rules are untouched.** This feature records and replays human decisions; it does not change any spirits rule, confidence rule or publish-gate blocker other than adding "approval stale".
- **Platform.** Works the same on macOS and Linux; no network, model or container is involved in recording or replaying.

## Dependencies

- **006-bottle-identity-layout (depends on)**: The ledger is keyed by bottle ID. Today's ID is the first photo's file stem (plus a path hash on collision), which can change if photos are renamed, moved or regrouped. 006 owns stable bottle identity and resolved output paths; this spec only requires that an ID which cannot be matched is reported, never guessed. 006 also decides where a re-rendered bottle is written (FR-017 uses today's never-overwrite rule until then).
- **003-stage-resumable-runs (coordinates with)**: A resumed or render-only run (`--from-stage`) MUST replay the ledger (FR-009, FR-016). Both specs change `process`/`feedback` orchestration in `cli/commands.py`; expect merge conflicts there.
- **008-spirits-validators (coordinates with)**: Deterministic validators should also check reviewer SET values (for example a proof that contradicts a "Bottled-in-Bond" release); a reviewer value still wins, but a validator warning must remain visible. Ordering of replay vs. validation is a planning decision shared with 008.
- **009-review-contact-sheet (feeds)**: The contact sheet must show the replayed state (approvals, retractions, stale approvals).
- **010-social-export (feeds)**: Social export must treat a bottle as publishable only when its approval was applied by replay (not stale, not withdrawn) and it is not rejected by replay. 010's FR-005(c) checks "the batch's rejected list"; once this spec lands, that list is no longer rewritten by `feedback` (FR-017), so 010 must read rejections from replay. Exports must not carry reviewer identity or ledger reasons (already 010's FR-011).
- **Shared code hot spots**: `cli/commands.py` (`feedback_command`, `_interactive_review`, `_apply_approvals`, `_resave_bottles`, `retry_command`/`_retry_bottle`), `image_processor/preview_generator.py` (`interactive_prompt`, rendering), `image_processor/enrichment.py` (`approve_facts`, `refresh_facts`), `image_processor/corrections.py`, `lib/settings.py`.

## Out of Scope

- Any graphical or web review interface; review stays in the terminal (`-i`) and via files and flags.
- Multi-user accounts, permissions, signatures or remote/shared ledger storage and syncing.
- Re-grouping bottles (merge/split) as a review action; it was removed and is tracked separately in ToDo.md.
- Changing how hints guide the vision model (`note`).
- Rewriting or compacting the ledger (deleting history); any future compaction would be a separate, explicitly approved feature.
- Publishing or posting to social media (sibling 010 exports drafts; posting remains manual).
