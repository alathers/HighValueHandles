<!--
SYNC IMPACT REPORT (temporary - remove before committing)
========================================================
Version Change: (none) → 1.0.0
Rational: Initial constitution ratification - establishing core governance principles

Modified Principles: (none - new document)

Added Sections:
- Core Principles: I. Python + uv Environment Management
- Core Principles: II. Test-First (NON-NEGOTIABLE)
- Core Principles: III. Documentation Standards
- Core Principles: IV. Git Workflow Governance
- Core Principles: V. Quality Gates (NON-NEGOTIABLE)
- Additional Constraints: Tooling Requirements
- Development Workflow: Git Worktree Development

Removed Sections: (none - new document)

Follow-up TODOs: (none)
-->

# Bourbon Inventory Constitution

## Core Principles

### I. Python + uv Environment Management
All development uses Python with `uv sync` for environment management. Python is the implementation language for all project components. `uv` provides fast dependency resolution and reproducible virtual environments. Developers MUST use `uv sync` to synchronize dependencies and create consistent development environments.

### II. Test-First (NON-NEGOTIABLE)
All code MUST be tested. Tests are written alongside implementation, not as an afterthought. Code without tests is not considered complete. Test coverage must be meaningful and substantial - trivial tests that exercise nothing do not satisfy this principle. Implementation proceeds only with test coverage that validates the intended behavior.

### III. Documentation Standards
All code MUST include good documentation and comments. Documentation is recorded in both detailed and concise formats. Detailed documentation provides comprehensive explanations for deep understanding. A version distilled for quick reference ("ADHD version") provides essential points at a glance, helping developers rapidly grasp key concepts without wading through lengthy text. Both documentation styles are maintained and kept in sync as code evolves.

### IV. Git Workflow Governance
All development work is committed to git. Branching rules are strict: feature branches for ALL work, never direct commits to main. All changes are checked in - partial work and work-in-progress are still committed with appropriate messages. Git history is NEVER overwritten - no force-pushing to shared branches, no rebase rewriting history. Merges are performed carefully with conflict resolution when necessary, preserving the integrity of the git graph. Development occurs in git worktrees to ensure parallel feature development does not collide - each feature branch gets its own isolated worktree, preventing interference between concurrent development streams.

### V. Quality Gates (NON-NEGOTIABLE)
Pre-commit hooks enforce code quality standards. All code MUST pass linting before commits. Large files MUST NOT be committed - pre-commit hooks block commits exceeding size thresholds. Software engineering practices are solid and consistent: code review, style adherence, testing standards, and quality metrics are applied via automated gates. These gates are non-optional - they run on every commit attempt and block commits that fail. Quality is enforced automatically before code enters the repository, not manually after the fact.

## Tooling Requirements

A cross-platform `install.sh` script supports macOS X and Linux. The script installs all Python packages and supporting tools required for development. Developers MUST run `./install.sh` to bootstrap their development environment on supported platforms. The script is idempotent - running it multiple times safely restores the environment to the correct state.

## Development Workflow

### Git Worktree Development
Parallel feature development uses git worktrees for isolation. Each feature branch gets its own dedicated worktree, preventing file-level conflicts between concurrent development streams. Worktrees are created with `git worktree add` and removed with `git worktree remove` when work completes. This enables simultaneous development of multiple independent features without editing file contention or accidental cross-branch contamination. Developers MUST use worktrees for features, not switch branches in a single working tree.

## Governance

The constitution is the source of truth for development practices. All workflows, linting rules, code review checklists, and quality gates derive from these principles and constraints. Amendments to the constitution require documentation, version increment, and clear communication of the change rationale. Developers raising PRs must verify their changes comply with constitutional requirements. Complexity beyond these principles must be explicitly justified. For runtime development guidance, use feature specifications and implementation plans as detailed references, not casual heuristics.

**Version**: 1.0.0 | **Ratified**: 2026-11-09 | **Last Amended**: 2026-11-09
