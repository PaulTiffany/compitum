# Changelog

All notable changes to this project are documented here.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-07-23
- Pre-tag hardening (2026-07-22/23): a clean, from-scratch mutation re-verification against the
  frozen candidate found the prior certification was not a genuinely isolated single run, so it was
  redone: 20 files re-verified locally, then confirmed for real via `mutation_dispatch.yml` CI
  (previous local-only work in this project's history has been wrong 5+ times without that
  confirmation). Fixed real gaps in `constraints.py`, `integrations/materials_project_audit.py`, and
  `applications/fusion/eval_offline.py` via test-only changes; corrected a misclassified
  `security.py` survivor set after real CI disagreed with the local reasoning. Along the way, fixed
  a `heavy_bench`/`routerbench` marker inconsistency that let a stateful benchmark test spuriously
  fail `release.yml`; pinned ruff/mypy versions after CI and local dev silently diverged; and found
  and fixed a severe, previously-undiscovered bug in the Cosmic Ray CI shard
  (`.github/workflows/mutation.yml`): `excluded-modules` had used dotted Python-import names instead
  of the glob-path syntax Cosmic Ray actually expects, so every past `cr-quick-shard` run had been
  mutating the entire `src/` tree (including `routerbench`) regardless of its intended target —
  fixed, plus added coverage-scoped test selection to address the resulting per-mutant cost, fixed a
  `mutation_dispatch.yml` concurrency setting that could silently cancel an in-progress sweep, and
  expanded its matrix to the full 20-file set.
- Invariants: additional deep tests for score directionality, mixture discrimination, dual scaling, batch determinism
- Docs: Core Science 0.1.1 coverage mapped to tests; index strip and README guidance
- CI: invariants job (PR) + nightly deep; docs linkcheck
- MATBENCH (Release 3, in progress): offline evaluation stack (calibration, regret, ridge baseline,
  layer/quantile exploration, attestation), Materials Project integration (`--from-mp`, presets for
  kagome/nickelate/FeSe chemistries), and sweep/full/online CI workflows. Fixed a calibration-leakage
  bug where the "official" regret number was evaluated on rows that had already been used to select
  the calibration lambda, rather than on a genuinely held-out split; fixed `eval_baseline_regret.py`
  silently ignoring `--mode min` tasks; fixed a dangling `materials` extras key in `pyproject.toml`.
- Applications: simulated superconductivity (Supercon) wrapper and CLI; fusion/plasma monitor with
  Lp-norm sweep support; Materials Project manifold audit module.
- RouterBench: online workflow guarded behind a connection-string secret with local-cache fallback;
  isolated `.venv-routerbench` used consistently across CI, rigor, and full-validation workflows
  (RouterBench's frozen dependency snapshot conflicts with compitum's own floors, e.g. pydantic).
- Mutation CI: consolidated `mutation_dispatch.yml`/`mutation_on_label.yml` to call the reusable
  `mutation.yml` instead of duplicating it; fixed unexported shell variables that silently broke
  per-shard reporting; fixed the Cosmic Ray exclusion list so all 16 shards are actually isolated;
  removed two unimplemented, never-gated workflow inputs (`mutmut_gate`, `cr_full`); removed
  Workflow Lint's blanket exclusion on the mutation workflow files. Fixed the 3 tests that had been
  permanently deselected across CI/rigor/mutation baselines — they were genuine test bugs (wrong
  assertion index, a miscounted word length, and a monotonicity test confounded by predictors fit on
  noise), not flaky or environment-dependent, and are now part of the normal suite.
- Homepage: honest hero metrics, researcher identity, cross-links to the Verifiable Routing paper and
  certified hybrid figures.
- Docs/notebooks/wiki: notebook-to-wiki embedding pipeline, Binder links for example notebooks.
- Mutation hardening: real `mutmut` sweeps (CI-verified where noted) across the full `src/compitum`
  shard matrix, closing every genuinely-testable survivor with behavioral tests rather than
  line-coverage padding. `constraints.py` and `metric.py`'s previously-deferred survivors were
  resolved this pass: `constraints.py`'s shadow-price/relaxation/tie-boundary gaps (via call-count
  side-channel observability, since several mutations don't change the final returned value) plus a
  new Hypothesis property test sweeping `ReflectiveConstraintSolver.select()`; `metric.py`'s
  backtracking-loop `bt`-counter/boundary gaps (via batches engineered to need an exact, discrete
  number of halvings to converge, rather than fragile floating-point bisection), with one mutant
  proven genuinely equivalent by direct simulation. Extended mutation scope to 3 previously-untested
  behavior-bearing modules (`integrations/materials_project_audit.py`,
  `applications/fusion/diiid_adapter.py`, `applications/fusion/eval_offline.py`); see
  `MUTATION_HARDENING_STATUS.md` for the full per-file accounting, survivor dispositions, and known
  local-tooling caveats.
- Packaging: `compitum.__version__` and Sphinx's `version`/`release` now derive from installed
  package metadata (`importlib.metadata`) instead of duplicating the version string.

## [0.1.1] - 2025-10-27
### Added
- Core Science Package: expanded Hypothesis invariants across geometry (SPD/triangle/ray/descent),
  control (Lyapunov decay/saturation/recovery; ΔV sequences; combined boundedness), coherence
  (monotone/symmetry/score direction/mixtures), constraints (feasibility monotone; duals slack≈0,
  boundary≥0; monotone/scale sanity), determinism (batch/repeated; paraphrase flip budget + explainability), and pedagogy (practice and prepared environment).
- RouterBench UX: env/data fallback for 5‑shot PKL with helpful error text and fetch script
- Docs & CI: Invariants coverage; Sphinx linkcheck; GitHub Pages deploy
- Community health files (CONTRIBUTING, CODE_OF_CONDUCT, SUPPORT, SECURITY)
- CI: minimal lint/test workflow
- Website: improved Share to LLM behavior
- RouterBench fetch script and docs

## [0.1.0] - 2025-10-26
### Added
- Initial public release tag `v0.1.0` (clean history with code + docs)

### Restored/Repo Hygiene
- Restored `index.html`, configs, `benchmarks/`, and `examples/`
