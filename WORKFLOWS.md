# CI/CD Workflows

This repository uses a small set of focused GitHub Actions workflows. Names and behavior are aligned for clarity and green, rigorous signal.

## CI
- Name: `CI`
- Triggers: push, pull_request (code-only by default; docs/assets/markdown ignored)
- What it does:
  - Lint (ruff), types (mypy), import smoke
  - Tests with CI profile; excludes routerbench and heavy_bench
- Matrix: ubuntu-latest, windows-latest
- Deps: minimal (no `[dev]` extras); installs only ruff/mypy/pytest bits
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
  - Installs minimal deps for core tasks
  - Optional Cosmic Ray (strict) and mutmut full runs (dispatch inputs)
  - Optional RouterBench steps (guarded by inputs)
  - Uploads artifacts
- Inputs:
  - `mutation` / `mutmut` / `mutation_sharded`: toggle heavy mutation phases
  - `rb_enable`: enable RouterBench venv/install/run
  - `rb_fetch`: attempt dataset fetch if missing (best-effort)
- Notes: Strict CR gating set to 1.0; heavy benches scheduled only

## Mutation: Reusable Shards
- Name: `Mutation Reusable Shards` (reusable via `workflow_call`)
- Inputs:
  - `mutmut` (bool, default true) — gates whether the mutmut-shard job runs at all
  - `cr_quick` (bool, default false) — gates whether the cr-quick-shard job runs at all
  - `cr_gate` (bool, default false) — fail the CR shard when its score is below `cr_score_threshold`; when false the score is advisory only
  - `cr_score_threshold` (string, default '1.0')
  - `cr_timeout` (string, default '240')
  - `target_files` (string JSON array of filenames under `src/compitum/` to shard mutmut over)
- What it does:
  - Mutmut per-file shards (coverage-guided)
  - Cosmic Ray quick shards, one per module group, each isolated by removing that module from `excluded-modules`
  - Uploads shard artifacts (7‑day retention)

## Mutation: Dispatcher
- Name: `Mutation Dispatcher`
- Triggers: workflow_dispatch (`strict` input maps to `cr_gate`), schedule (opt-in via the `ENABLE_MUTATION_SCHEDULE` repo variable)
- What it does:
  - Calls `Mutation Reusable Shards` with mutmut + CR quick
  - Use for nightly or manual mutation validation across the codebase

## Mutation: PR Label
- Name: `Mutation PR Label`
- Triggers: pull_request (on open/reopen/label/sync)
- What it does:
  - If PR has label `mutation`, computes changed `src/compitum/*.py` files and passes them as shard targets
  - Runs mutmut over changed files and CR quick shards (advisory; `cr_gate` is not set, so scores are reported, not enforced)
  - Posts a summary comment with CR scores and mutmut survivors (if any)

## Release
- Name: `Release`
- Triggers: workflow_dispatch, tags `v*`
- What it does:
  - Quality gates (ruff, mypy, pytest)
  - Build sdist/wheel; twine check; smoke install test
  - Upload artifacts (dry-run by default)

## Notebooks
- Name: `Notebooks`
- Triggers: pull_request changes under `notebooks/**`, push to main touching notebooks
- What it does:
  - Installs minimal runtime + nbmake
  - Executes notebooks; skips gracefully if none present

## Matbench (offline)
- Name: `matbench_offline`
- Triggers: workflow_dispatch (inputs for CSV path, objective column, mode, topk/lambda grids)
- What it does:
  - Calibrates SRMF lambda on provided CSV; evaluates regret@k + AURC with bootstrap
  - Baseline CV regret (ridge); optional group regret and layer exploration if columns exist
  - Generates attestation JSON; uploads artifacts (calibration, scores, regret, baseline, layers, attestation)
- Notes: Offline-only; assumes the CSV is present in the repo; no Releases.

## Materials audit (MP API)
- Name: `materials_audit`
- Triggers: workflow_dispatch (elements, nelements, thresholds)
- What it does:
  - Installs mp_api and runs `tools/audit_materials_manifold.py` with `secrets.MP_API_KEY`
  - Exports SRMF curvature/leak and candidate flag to a CSV artifact
- Notes: Network-bound only in this manual workflow; local offline mock is available via `--offline-mock`.

## RouterBench separation of concerns
- Default CI/rigor runs do not install or execute RouterBench unless explicitly enabled.
- Enable options:
  - CI: add label `routerbench` to PR, or run `CI` via dispatch; optionally set repo var `ENABLE_ROUTERBENCH=true` for scheduled CI.
  - Full validation: pass inputs `rb_enable=true` (and `rb_fetch=true` if dataset fetch desired).
- Data caching: workflow caches dataset paths and HF/ST model caches when enabled.

## Mutation separation of concerns
- Heavy mutation phases are opt-in:
  - PRs: add label `mutation` to run sharded mutmut + CR quick on changed modules.
  - Manual: dispatch `Mutation: Dispatcher` or `Validation: Full` with mutation inputs.
  - Nightly: scheduler can be enabled, but remains opt-in by default.

## Notes on CI‑only test selection
- Mutation workflows (mutmut + CR) provide strong guarantees; nightly runs remain strict (`cr_score_threshold` = 1.0).

## How to run mutation
- Manual: `Mutation: Dispatcher` → Run workflow
- Nightly: Runs automatically
- PRs: Add label `mutation` to run shards on changed modules

