# Specification Quality Checklist: Checkpoint / Stage-Resumable Runs (`--from-stage`)

**Purpose**: Check that the specification is complete and of good quality before planning starts.

**Created**: 2026-11-24

**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain (FR-006 resolved by the owner on 2026-11-24; FR-029 was resolved as a default; see Iteration 3)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

Validation ran twice.

**Iteration 1**

- Every item passed except the open clarification markers.
- Every point in the owner's text is covered:
  - the vision (L) and OCR (O) results are saved as soon as stage 1 ends: FR-001 to FR-005, User Story 2;
  - a re-run can start at `research`, `factcheck` or `render`: FR-010 and FR-014 to FR-016, User Stories 1, 3 and 4;
  - changing search queries or verification rules never needs the vision pass again: FR-014, FR-015, FR-018, SC-001 and SC-008.
- Names of modules and functions appear only under Assumptions and Dependencies. They are there to record existing behaviour, as the workflow asked, and they are not requirements.
- The CLI flag `--from-stage`, its values and the exit code are part of the user-facing interface the owner asked for. They are not implementation choices.

**Iteration 2**

- Re-checked the spec against the code in this worktree, as of `9002167`.
- Today the batch JSON and the bottle folders are written only after enrichment and review (`_process_batches`). This confirms the gap that FR-001 closes.
- Hint reading notes go into the vision prompt. That is the reason for FR-020's warning.
- No changes were needed.

**Iteration 3 (independent review)**

The spec was checked against the code at `9002167` and against the sibling
specs 004 to 011. Issues found and fixed:

- **Grounding**:
  - The raw vision reply is parsed and then discarded today. A
    `vision_raw_response` key is honoured on save, but nothing sets it. So
    "raw reply when available" was not testable. FR-002 now requires the
    raw reply and records that keeping it is new.
  - Quantity counting needs no model. Hint values feed the search identity
    as well as the fact check. Assumptions now say both.
- **Dry run contradicted existing behaviour**: today's `--dry-run` probes
  the model server. FR-025 and User Story 6 now keep that probe for the
  models a resume still needs, and skip it for `render`.
- **Reviewer decisions were lost**:
  - FR-008 saved the fact record before review, yet FR-030 promised that
    `render` keeps approvals. Interactive review and `feedback` happen after
    the fact check, so a `render` resume would have dropped corrections and
    approvals.
  - Added FR-037, which makes later fact-record versions from review and
    `feedback`. FR-016 now renders the latest version.
  - Added FR-035, which reports the decisions a `research` or `factcheck`
    resume does not re-apply and skips rejected bottles.
  - Added SC-009 and acceptance scenario US4-4.
- **Stale research after a hint change**: a changed identity hint would have
  let the fact check judge evidence that was searched for another product.
  Added FR-034, the search identity in FR-007, and acceptance scenario
  US3-4.
- **`--no-web` source runs**: these were refused at `factcheck`. They now
  get a "skipped" research record (FR-005) and can be re-checked on label
  and OCR only.
- **Chained resumes**: resuming a resumed run had no defined record
  ownership. Added FR-036, and FR-026 now names the run that produced each
  record.
- **Options for stages that are loaded, not re-run**: research and text
  options under a later stage were silently ignored. FR-019 now covers
  them, using the same rules as the stage-1 options.
- **Photo checks**: these were missing for `render`, and `-f` had no
  default. FR-021 now applies at every stage and falls back to the recorded
  photo folder.
- **FR-013**: the handling of an unknown bottle ID was ambiguous. It now
  stops the run before any work.
- **FR-012**: "most recent" was ambiguous across time zones. It is now
  defined by the stage-1 start time, compared with time zones.
- **FR-017**: pre-flight no longer warm-loads the vision model or checks
  for Tesseract when no OCR runs.
- **FR-029 resolved as a default**:
  - Option B, replacing the source run's folders, contradicted FR-028.
  - Option C is a coordination matter. Siblings 005 and 006 already adopt
    "today's never-overwrite rule until 006, then 006's layout".
  - FR-028 is narrowed to records, indexes, settings and batch data, so it
    no longer conflicts with 006's update-in-place rule.
- **Sibling boundaries**:
  - 004's OCR passage boundaries and 007's decoded barcode are now in the
    stage-1 record. 007's registry lookup and 011's provider settings are
    now in the research record.
  - 008's reconcile step is mapped to `factcheck`.
  - 010 is a separate read-only export, not a render-stage output.
  - 006 assigns "skip completed bottles on re-run" to 003. That item is now
    explicitly out of scope here and flagged under Dependencies as a gap to
    resolve.
- **Success criteria**: SC-001 and SC-003 now exclude interactive review
  time.

Re-validation:

- All items pass except the single open clarification marker, FR-006.
  It is genuinely the owner's call, because it adds a fourth start point
  that the owner's text does not list, against the owner's stated goal.
- The new requirements are numbered FR-034 to FR-037 and SC-009, appended
  so that existing references stay valid.

**Iteration 4 (owner clarification, 2026-11-24)**

- FR-006 resolved as option C: no `vision` start point; the run index lists
  groups stage 1 never reached, and resumes and resume dry runs report them
  as "not read yet". Added a matching edge case and updated Out of Scope.
- No open items remain.
