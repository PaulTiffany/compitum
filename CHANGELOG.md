# Changelog

All notable changes to this project are documented here.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
- Invariants: additional deep tests for score directionality, mixture discrimination, dual scaling, batch determinism
- Docs: Core Science 0.1.1 coverage mapped to tests; index strip and README guidance
- CI: invariants job (PR) + nightly deep; docs linkcheck

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
