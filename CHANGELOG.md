# Changelog

All notable changes to this project are documented here.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
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
