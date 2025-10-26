---
title: Math Brief
description: Plain-language summary of the mathematical ingredients behind Compitum, with pointers to the project wiki for formal details.
---

# Mathematics: A Plain-Language Brief

:::{admonition} Quick Summary
:class: tip

Compitum routes requests to the model that maximizes a simple utility: performance minus lambda times total cost, under hard constraints. It learns a low-rank SPD metric to compare contexts, monitors boundary signals (gap, entropy, sigma), and emits a routing certificate for auditability.
:::

Purpose

- This page summarizes the mathematical ideas used by Compitum in reviewer-friendly language.
- It is a map, not a derivation. Formal details and proofs live in the project wiki.
- See the Pointers section below for derivations, references, and extended notes.

Core Objective

- Compitum selects a model that maximizes a utility U for a given willingness-to-pay (lambda):
  U = performance - lambda * total_cost.
- It respects hard constraints and emits a routing certificate with all components (utility parts,
  constraint status and approximate shadow prices, boundary diagnostics, drift monitors).

Constraints and Shadow Prices

- Constraints are linear inequalities A x <= b over process variables (e.g., rate limits, region
  availability, policy rules). The solver reports approximate local shadow prices (finite-difference
  viability diagnostics) indicating which constraints appear binding.
- Interpretation: If a constraint's shadow price is nonzero, relaxing it would increase U;
  large values signal active tradeoffs at the boundary.

SPD Metric Geometry (Intuition)

- Compitum maintains a symmetric positive definite (SPD) metric that shapes distances in a learned
  feature space. You can view this as a learned Mahalanobis distance with low-rank structure.
- Notation: M is SPD. Practical parameterization: a low-rank factor plus a small diagonal
  regularizer (rank, delta) for stability.
- Effect: The metric focuses sensitivity on directions that matter for routing outcomes and cost.

Coherence Functional (Metric-Aware KDE)

- Compitum tracks a notion of local "coherence" using kernel density estimation with the learned
  metric. High coherence implies similar contexts behaved predictably; low coherence invites caution
  (defer/route conservatively).

Boundary Diagnostics (Gap, Entropy, Sigma)

- gap: difference in utility between the chosen model and the runner-up; small gap -> ambiguity.
- entropy: softmax entropy over model scores; high entropy -> uncertainty.
- sigma: dispersion of per-model scores; large sigma -> spread in expected outcomes.
- Combined, these guide safe deferrals and help analyze near-frontier behavior.

Stable Online Updates (Trust Region / Lyapunov Intuition)

- Updates to predictors and the metric are constrained by a trust radius. This curbs step sizes,
  stabilizes learning, and avoids destabilizing swings.
- A Lyapunov-like energy idea motivates gentle change; we use a trust-region controller in practice.

Routing Certificate (Auditability)

- Every decision outputs:
  - utility and its components (quality, latency, cost)
  - constraints (feasible, shadow prices)
  - boundary diagnostics (gap, entropy, sigma)
  - drift status (instantaneous, EMA, integral trust radius)
- This provides a mechanistic, non-judge basis for evaluation and ablations.

Ethical Notes

- We use standard, well-established mathematics (Lagrangians, SPD metrics, KDE, trust regions).
- Compitum claims are empirical (utility/regret at fixed lambda, frontier gap), not theoretical
  guarantees beyond proper constraint handling and stable updates.

Pointers (for full details)

- Project wiki: Mathematics - https://github.com/PaulTiffany/compitum/wiki/Mathematics
- Wiki home: https://github.com/PaulTiffany/compitum/wiki
- Constraints: configs/constraints_us_default.yaml (default hard limits)
- Certificate anatomy: docs/PEER_REVIEW.md (section: Routing Certificate)

---

## Claims and Evidence (Code-Linked)

This section maps key scientific claims to the exact code and property-based tests that verify them.

- Learning descent (line search)
  - Claim: metric surrogate energy E(L; z) = 0.5 * beta_d * ||L^T z||^2 is non-increasing per update.
  - Code: `src/compitum/metric.py::batch_update_spd`
  - Test: `tests/invariants/test_invariants_lg.py::test_metric_update_line_search_non_increase`

- Zero-error Lyapunov decay (controller)
  - Claim: with d_star = 0 over steps, V_ctrl = (drift_integral)^2 is non-increasing; trust radius is bounded.
  - Code: `src/compitum/control.py`
  - Tests: `tests/invariants/test_invariants_control_sy.py::test_lyapunov_decays_under_zero_error`, `test_controller_integral_bounded_under_bounded_drift`, `test_trust_radius_moves_toward_bounds_under_persistent_signals`

- Two-timescale isolation (stride)
  - Claim: the metric low-rank factor L does not change before hitting `update_stride`.
  - Code: `src/compitum/router.py` (stride guards)
  - Test: `tests/invariants/test_invariants_control_sy.py::test_two_timescale_metric_update_stride`

- Routing-level distance descent (system proxy)
  - Claim: with fixed embedding and single model, the positive distance decreases overall across updates.
  - Test: `tests/invariants/test_invariants_srmf_lyapunov.py::test_distance_decreases_over_updates`

- Certificate auditability (constraints, boundary, drift)
  - Claim: every decision emits utility components, feasibility + approximate shadow prices, boundary diagnostics, and drift state.
  - Code: `src/compitum/router.py` (certificate), `src/compitum/constraints.py`, `src/compitum/boundary.py`, `src/compitum/energy.py`
  - Docs: `docs/Certificate-Schema.md`, `docs/PEER_REVIEW.md`

- Offline + redaction (privacy)
  - Claim: no judge models; offline flag; audit records redact prompt (hash only) and include digests.
  - Code: `src/compitum/cli.py` (offline, audit), `src/compitum/security.py` (hashing, audit record)
  - Docs: `SECURITY.md`

See also: `docs/SRMF-as-Lyapunov.md` for assumptions and lemmas with links to tests.

---

## Thanks

Thank you to reviewers, contributors, and the community for pushing us to make Compitum rigorous, reproducible, and accessible. Your feedback directly shaped the invariants, documentation, and the SRMF-to-Lyapunov framing that ties claims to falsifiable tests and concrete code.

