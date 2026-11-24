# AGENTS.md — SDLC workflow

Rules for every coding agent (and human) working in this repo. Imperative; follow exactly.
If a rule conflicts with a direct human instruction in the current session, ask before deviating.

## Branch model

| Branch | Lifetime | Purpose | How it changes |
|---|---|---|---|
| `<type>/<slug>` | short | one task | commits by the author |
| `dev` | permanent — never delete | integration + pre-release testing | squash-merged PRs from work branches |
| `main` | permanent | release candidates; clean public history | squash of `dev` (one commit per promotion); spec reservations; hotfixes |
| `production` | permanent | what is live | fast-forward to a tagged commit on `main` |

Flow: `work branch → dev → (squash) → main → (tag + fast-forward) → production`

## 1. Start work

- Branch from the latest `dev`: `git fetch origin && git switch -c <type>/<slug> origin/dev`
- `<type>` is one of `feat fix chore docs refactor test perf ci`. Include the issue id if there is one: `fix/123-null-token`.
- Spec-driven work uses the spec id as the branch name instead: `00042-short-slug` (see *Spec-driven development*).
- One task per branch. Unrelated fixes go on their own branch.
- Running in parallel with other agents: use your own worktree (`git worktree add ../<repo>-<slug> -b <type>/<slug> origin/dev`). Never switch branches in a checkout another agent is using.

## 2. Commit

- Conventional Commits: `<type>(<scope>)!: <summary>` — imperative, ≤72 chars. `!` or a `BREAKING CHANGE:` footer marks a breaking change.
- Small, working commits. Never commit secrets, credentials, `.env` files, or generated artifacts.
- Never use `--no-verify`; if a hook fails, fix the cause.

## 3. Merge to `dev`

- Before opening a PR: rebase on `origin/dev`, then run the project's format, lint, type-check, and test commands (see *Project commands*). All must pass. Add or update tests for any behavior change.
- Open a PR targeting `dev`. Title = Conventional Commit. Body: what, why, how it was tested, risks/rollback.
- Keep PRs small and single-purpose. Do not weaken or delete tests to get green.
- Merge method: **squash**. Then **delete the work branch** (remote and local) and remove its worktree.

## 4. Promote `dev` → `main` (human-initiated only)

1. Confirm pre-release testing on `dev` is complete and CI is green.
2. Squash `dev` into `main` as ONE commit:
   ```
   git switch main && git pull --ff-only
   git merge --squash origin/dev
   git commit -m "release: <summary>"
   git push origin main
   ```
3. **Write the commit message by hand.** It must be a curated, user-facing summary. Never paste the auto-generated list of `dev` commit messages, branch names, PR numbers, or internal ticket ids — that is the leak the squash exists to prevent.
4. **Immediately back-merge** so the next promotion diffs cleanly (skipping this causes repeat conflicts):
   ```
   git switch dev && git pull --ff-only
   git merge origin/main -m "chore: sync main into dev"
   git push origin dev
   ```
   This merge must change no files. If it does, stop and ask.
5. Never delete, rebase, or force-push `dev`.

## 5. Release to `production` (human-initiated only)

1. Pick the exact `main` commit to release. Choose the version (SemVer, from changes since the last tag):
   breaking → MAJOR · `feat` → MINOR · `fix`/other → PATCH. Pre-releases: `vX.Y.Z-rc.N`.
2. Update `CHANGELOG.md` and any version file as part of the promotion in step 4, not after.
3. Tag and promote:
   ```
   git tag -a vX.Y.Z <sha> -m "vX.Y.Z"
   git push origin vX.Y.Z
   git push origin "vX.Y.Z^{commit}:refs/heads/production"
   ```
   The push to `production` must be a fast-forward. One annotated tag marks the release on both `main` and `production` (they share the commit).
4. Tags are immutable. Never move or delete a pushed tag; ship a new version instead.

## 6. Hotfix

- Default: fix on a work branch → `dev` → normal promotion.
- Urgent: `hotfix/<slug>` from `main` → PR to `main` (squash) → back-merge `main` into `dev` (step 4.4) → tag a PATCH → promote.
- Rollback: prefer roll-forward. Re-pointing `production` to an older tag is a non-fast-forward and is a human-only action.

## 7. Spec-driven development (strongly preferred)

Use Spec Kit, or an equivalent staged process (specify → clarify → plan → tasks → implement), for any non-trivial feature. Skip it only for small fixes and chores.

**Spec ids**
- Sequential, zero-padded to **5 digits**, shared by the branch and the directory: `specs/00042-short-slug/` on branch `00042-short-slug`.
- Run `git fetch --all --prune` first, then let the tool allocate the next id. Never pass a number by hand. Never reuse an id, even for an abandoned spec.
- Ids must stay under 7 digits (Spec Kit reads longer prefixes as timestamps).

**Reserve the id early** — as soon as `spec.md` exists, before plan, tasks, or code:
1. Create the spec branch from an up-to-date `main`: `git switch -c <id>-<slug> origin/main`.
2. Commit only `specs/<id>-<slug>/`. Message: `docs(spec): add <id>-<slug>`.
3. Open two PRs from that branch, both squash: one → `dev`, one → `main`. Say in the body that it is a spec reservation.
4. If the id or slug is already taken on either target, renumber to the next free id before merging. This is the only time an id may change.
5. After the `main` PR merges, a human runs the sync from step 4.4 (`main` → `dev`, no file changes). Do not edit the spec on `dev` before this sync or the next promotion will conflict.
6. Delete the spec branch, recreate `<id>-<slug>` from `origin/dev`, and continue with plan → tasks → implement. Implementation PRs go to `dev` as in step 3 of the main workflow.

The spec reservation PR is the only change allowed on `main` besides the `dev` squash and hotfixes. Agents may open it; a human merges it.
A spec on `main` is as visible as `main` is: no secrets, customer names, or internal-only detail in `spec.md`.

## Hard rules for agents

- NEVER commit or push directly to `dev`, `main`, or `production`.
- NEVER force-push, rewrite published history, or delete `dev`, `main`, `production`, or any tag.
- NEVER merge to `main`, create a release tag, or move `production` unless a human explicitly asks in this session.
- NEVER change CI config, branch protection, hooks, or this file's rules as a side effect of another task.
- Do not add dependencies without saying so in the PR body. Pin versions.
- If blocked, unsure, or a step fails twice: stop and report. Do not improvise around the workflow.

## Definition of done

Checks pass locally and in CI · tests cover the change · docs/changelog updated if behavior changed · PR merged to `dev` · work branch deleted.

## Project commands

<!-- Fill in per repo. Exact commands, not descriptions. -->
- Setup: `...`
- Format: `...`
- Lint / type-check: `...`
- Test: `...`
- Run locally: `...`