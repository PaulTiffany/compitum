# CI/CD Workflows

This repository uses a small set of focused GitHub Actions workflows. Names and behavior are aligned for clarity and green, rigorous signal.

## CI
- Name: `CI`
- Triggers: push, pull_request (code-only by default; docs/assets/markdown ignored)
- What it does:
  - Lint (ruff), types (mypy), import smoke
  - Tests with CI profile; excludes routerbench and a few CI‑unfriendly strict cases
  - Matrix: ubuntu-latest, windows-latest
- Stability: concurrency enabled; job timeouts applied

## Docs
- Name: `Docs`
- Triggers: push, pull_request
- What it does:
  - Build Sphinx docs; linkcheck
  - Assemble site (hero index + docs under /docs); upload to Pages
- Stability: concurrency on Pages; least-privileged permissions

## Validation: Full
- Name: `Validation: Full`
- Triggers: workflow_dispatch, schedule (daily)
- What it does:
  - Installs dev deps; runs heavy benches (scheduled only)
  - Optional Cosmic Ray (strict) and mutmut full runs
  - RouterBench guarded tests if dataset present
  - Uploads artifacts
- Notes: Strict CR gating set to 1.0; heavy benches skipped on manual run

## Mutation: Reusable Shards
- Name: `Mutation: Reusable Shards` (reusable via `workflow_call`)
- Inputs:
  - `mutmut` (bool, default true)
  - `cr_quick` (bool, default false)
  - `mutmut_gate` (bool, default false)
  - `cr_score_threshold` (string, default '1.0')
  - `cr_timeout` (string, default '240')
  - `target_files` (string JSON array of filenames under `src/compitum/` to shard mutmut over)
- What it does:
  - Mutmut per-file shards (coverage-guided)
  - Cosmic Ray quick shards (gated by threshold)
  - Uploads shard artifacts (7‑day retention)

## Mutation: Dispatcher
- Name: `Mutation: Dispatcher`
- Triggers: workflow_dispatch, schedule (daily)
- What it does:
  - Calls `Mutation: Reusable Shards` with mutmut + CR quick
  - Use for nightly or manual mutation validation across the codebase

## Mutation: PR Label
- Name: `Mutation: PR Label`
- Triggers: pull_request (on open/reopen/label/sync)
- What it does:
  - If PR has label `mutation`, computes changed `src/compitum/*.py` files and passes them as shard targets
  - Runs mutmut over changed files and CR quick shards (threshold 0.99 for PRs)
  - Posts a summary comment with CR scores and mutmut survivors (if any)

## Release
- Name: `Release`
- Triggers: workflow_dispatch, tags `v*`
- What it does:
  - Quality gates (ruff, mypy, pytest)
  - Build sdist/wheel; twine check; smoke install test
  - Upload artifacts (dry-run by default)

## Rigor (Unified)
- Name: `Rigor`
- Triggers: push, pull_request, workflow_dispatch
- What it does:
  - Lint (ruff 100-col), types (mypy --strict), Bandit
  - Tests with 100% line+branch coverage and invariants suite
  - PyTest benchmarks with JSON artifact
  - RouterBench local evaluation (guarded) + Compitum eval + HTML analysis report
  - Light mutation (mutmut shards + Cosmic Ray quick shards; gated at 1.0)
  - Sphinx build (nitpicky) + linkcheck
  - Uploads artifacts for each phase

## Notes on CI‑only test selection
- A small number of strict tests are deselected in CI to keep runs stable on shared runners; local runs remain strict.
- Mutation workflows (mutmut + CR) provide strong guarantees; nightly runs remain strict (`cr_score_threshold` = 1.0).

## How to run mutation
- Manual: `Mutation: Dispatcher` → Run workflow
- Nightly: Runs automatically
- PRs: Add label `mutation` to run shards on changed modules

## CI Note

To enforce text hygiene in CI/CD, add a step to run `make check-mojibake`. This gate only scans `README.md` and `docs/**/*.md`, leaving `src/` and vendored code untouched.
