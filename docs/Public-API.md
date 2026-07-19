---
title: Public API Stability
description: Stable modules, classes, and versioning policy for integrators.
---

# Public API Stability

This page lists the primary, stable modules and classes intended for integrators embedding Compitum as a library, and states our versioning policy.

Stable Modules and Classes

- Router and certificate
  - `compitum.router.CompitumRouter`
  - Certificate object returned by `router.route(prompt)`, with `.to_json()` for structured output

- Core components
  - `compitum.constraints.ReflectiveConstraintSolver`
  - `compitum.boundary.BoundaryAnalyzer`
  - `compitum.coherence.CoherenceFunctional`
  - `compitum.metric.SymbolicManifoldMetric`
  - `compitum.energy.SymbolicFreeEnergy`
  - `compitum.control.SRMFController`
  - `compitum.pgd.RegexPromptExtractor`
  - `compitum.predictors.CalibratedPredictor`
  - `compitum.models.Model`

Certificate Schema

- See {doc}`Certificate-Schema` for a minimal schema and download link for `assets/certificate.schema.json`.

Versioning Policy

- We follow SemVer in spirit for the public API listed above:
  - Backwards-compatible additions and internal changes: MINOR/PATCH increments.
  - Breaking changes to the public API: MAJOR increment and documented in `CHANGELOG.md`.
- Experimental or internal modules not listed here may change without notice.

Compatibility Expectations

- Constructors and method names for the listed classes are stable within a MAJOR version.
- The certificate JSON retains key fields (`model`, `utility`, `utility_components`, `constraints`, `boundary`, `drift`, `pgd_signature`, `timestamp`, `router_version`). Note the JSON keys are `boundary`/`drift` -- the `SwitchCertificate` dataclass's own Python attributes are named `boundary_analysis`/`drift_status`, but `.to_json()` renames them. New utility components may be added; consumers should treat unknown component keys as additive numeric terms.

Support and Feedback

- If you depend on a behavior not covered here, please open an issue to discuss formalizing it in the public API.

