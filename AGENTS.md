# AGENTS.md — SDLC workflow

Rules for every coding agent (and human) working in this repo. Imperative; follow exactly.
If a rule conflicts with a direct human instruction in the current session, ask before deviating.

## Branch model

| Branch | Lifetime | Purpose | How it changes |
|---|---|---|---|
| `<type>/<slug>` | short | one task | commits by the author |
| `main` | permanent | integration + release candidates; clean public history | squash-merged PRs from work branches; hotfixes |
| `production` | permanent | what is live | fast-forward to a tagged commit on `main` |

Flow: `work branch → (squash PR) → main → (tag + fast-forward) → production`

This project has no `dev` branch. Work branches merge straight into `main`.

## 1. Start work

- Branch from the latest `main`: `git fetch origin && git switch -c <type>/<slug> origin/main`
- `<type>` is one of `feat fix chore docs refactor test perf ci`. Include the issue id if there is one: `fix/123-null-token`.
- Spec-driven work uses the spec id as the branch name instead: `00042-short-slug` (see *Spec-driven development*).
- One task per branch. Unrelated fixes go on their own branch.
- Running in parallel with other agents: use your own worktree (`git worktree add ../<repo>-<slug> -b <type>/<slug> origin/main`). Never switch branches in a checkout another agent is using.

## 2. Commit

- Conventional Commits: `<type>(<scope>)!: <summary>` — imperative, ≤72 chars. `!` or a `BREAKING CHANGE:` footer marks a breaking change.
- Small, working commits. Never commit secrets, credentials, `.env` files, or generated artifacts.
- Never use `--no-verify`; if a hook fails, fix the cause.

## 3. Merge to `main`

- Before opening a PR: rebase on `origin/main`, then run the project's format, lint, type-check, and test commands (see *Project commands*). All must pass. Add or update tests for any behavior change.
- Open a PR targeting `main`. Title = Conventional Commit. Body: what, why, how it was tested, risks/rollback.
- Keep PRs small and single-purpose. Do not weaken or delete tests to get green.
- Merge method: **squash**. A human merges (see *Hard rules for agents*).
- **Curate the squash commit message.** It lands directly on `main`, so it must be a clean summary: the PR title plus a short description. Never keep the auto-generated list of branch commit messages, branch names, or internal ticket ids.
- After the merge, **delete the work branch** (remote and local) and remove its worktree.

## 4. Release to `production` (human-initiated only)

1. Confirm CI is green on `main`. Pick the exact `main` commit to release. Choose the version (SemVer, from changes since the last tag):
   breaking → MAJOR · `feat` → MINOR · `fix`/other → PATCH. Pre-releases: `vX.Y.Z-rc.N`.
2. Update `CHANGELOG.md` and any version file in a PR to `main` before tagging, not after.
3. Tag and promote:
   ```
   git tag -a vX.Y.Z <sha> -m "vX.Y.Z"
   git push origin vX.Y.Z
   git push origin "vX.Y.Z^{commit}:refs/heads/production"
   ```
   The push to `production` must be a fast-forward. One annotated tag marks the release on both `main` and `production` (they share the commit).
4. Tags are immutable. Never move or delete a pushed tag; ship a new version instead.

## 5. Hotfix

- Default: fix on a work branch → PR to `main` → normal release.
- Urgent: `hotfix/<slug>` from the release tag on `production` → PR to `main` (squash) → tag a PATCH → promote. If `main` has moved past the release and the fix must ship without those changes, stop and ask a human.
- Rollback: prefer roll-forward. Re-pointing `production` to an older tag is a non-fast-forward and is a human-only action.

## 6. Spec-driven development (strongly preferred)

Use Spec Kit, or an equivalent staged process (specify → clarify → plan → tasks → implement), for any non-trivial feature. Skip it only for small fixes and chores.

**Spec ids**
- Sequential, zero-padded to **5 digits**, shared by the branch and the directory: `specs/00042-short-slug/` on branch `00042-short-slug`.
- Run `git fetch --all --prune` first, then let the tool allocate the next id. Never pass a number by hand. Never reuse an id, even for an abandoned spec.
- Ids must stay under 7 digits (Spec Kit reads longer prefixes as timestamps).

**Reserve the id early** — as soon as `spec.md` exists, before plan, tasks, or code:
1. Create the spec branch from an up-to-date `main`: `git switch -c <id>-<slug> origin/main`.
2. Commit only `specs/<id>-<slug>/`. Message: `docs(spec): add <id>-<slug>`.
3. Open one PR from that branch → `main` (squash). Say in the body that it is a spec reservation.
4. If the id or slug is already taken on `main`, renumber to the next free id before merging. This is the only time an id may change.
5. After the PR merges, delete the spec branch, recreate `<id>-<slug>` from `origin/main`, and continue with plan → tasks → implement. Implementation PRs go to `main` as in step 3.

Agents may open the reservation PR; a human merges it.
A spec on `main` is as visible as `main` is: no secrets, customer names, or internal-only detail in `spec.md`.

## Accuracy over agreeableness

Being correct matters more than being helpful-sounding, fast, or agreeable. A wrong answer delivered confidently costs more than no answer.

- "I don't know" and "I couldn't verify this" are acceptable answers. Say them plainly instead of guessing.
- Never invent APIs, flags, file paths, config keys, version numbers, test results, or citations. If you did not read it or run it in this session, do not state it as fact.
- Label every non-trivial claim as one of: **verified** (read or ran it — say how), **inferred** (reasoned from evidence — say from what), or **unknown**.
- When you don't know and cannot find out with the tools you have, stop and ask: "I don't know X. Do you need an answer, or is 'unknown' acceptable here?" If an answer is required, say what it would take to get one (access, a test, a document, a person).
- If the answer is needed to proceed and guessing would change code, a spec, or a design decision, do not proceed on the guess. Record it as an open question (`[NEEDS CLARIFICATION]` in specs) and wait.
- Disagree when the evidence says so. Report design flaws, failing tests, and risks even when unasked and even when the human seems committed to the approach. Do not soften a finding to keep the plan intact.
- Never claim work is done, tested, or passing unless you ran the check and saw it pass. Report partial results as partial.
- If you find you were wrong earlier, say so immediately and state what it affects.

## Hard rules for agents

- NEVER commit or push directly to `main` or `production`.
- NEVER force-push, rewrite published history, or delete `main`, `production`, or any tag.
- NEVER merge a PR to `main`, create a release tag, or move `production` unless a human explicitly asks in this session.
- NEVER change CI config, branch protection, hooks, or this file's rules as a side effect of another task.
- Do not add dependencies without saying so in the PR body. Pin versions.
- If blocked, unsure, or a step fails twice: stop and report. Do not improvise around the workflow.

## Definition of done

Checks pass locally and in CI · tests cover the change · docs/changelog updated if behavior changed · PR merged to `main` · work branch deleted.

## Project commands

<!-- Fill in per repo. Exact commands, not descriptions. -->
- Setup: `...`
- Format: `...`
- Lint / type-check: `...`
- Test: `...`
- Run locally: `...`