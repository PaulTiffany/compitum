---
title: Mathematical Bridge
description: Mapping Compitum’s terminology to standard optimization, geometry, and statistics for arXiv reviewers.
---

# Mathematical Bridge (PS → ML)

Purpose

- Provide a concise translation from project terminology to mainstream math/ML terms used in the Compitum paper and code.
- Clarify which pieces are claims (engineering guarantees), which are heuristics, and how they relate to standard references.

Term Mapping

- Bounded Observer → budgeted decision maker with finite evaluation resources; fixed willingness-to-pay (WTP) slices.
- Symbolic Manifold / Curvature → feature space R^D with a learned SPD Mahalanobis geometry M = L L^T + δI.
- Drift → exogenous variability of contexts; modeled via distances and predictive uncertainty.
- Reflection → regularization/selection: feasibility constraints, trust-region control, and density-based priors.
- Symbolic Free Energy → scalarized utility U combining quality, latency, cost, distance penalty, and a bounded coherence prior.
- Coherence Functional → KDE log-density prior in whitened coordinates under the learned metric.
- Arrow of Time → Lyapunov-inspired trust-region control (EMA + integral); directional update memory, ensuring stability.

Decision Rule

- Feasibility-first: filter by capabilities and linear constraints AxB ≤ b.
- Selection: choose argmax utility U(x, m) among feasible models.
- Certificate: emit utility components, feasibility and approximate local shadow prices (finite-difference), boundary diagnostics (gap/entropy/uncertainty), and drift/trust radius.

Guarantees vs. Heuristics

- Guarantees
  - PD metric via L L^T + δI; defensive Cholesky update.
  - Feasibility-first selection; constraint compliance by construction.
  - Bounded update magnitude via trust-region controller.
- Heuristics
  - Shadow prices as approximate local diagnostics (finite-difference viability); report-only.
  - KDE prior with clipping and small weight β_s; bounded influence and sensitivity-checked.
  - Per-sample batch updates are order-dependent; chosen for throughput.

References (Pointers)

- Mahalanobis metric learning (e.g., ITML/LMNN), kernel density estimation, and trust-region/step-size control in online optimization.
- See code anchors: energy (src/compitum/energy.py:33), metric (src/compitum/metric.py:23,39,106), constraints (src/compitum/constraints.py:36), boundary (src/compitum/boundary.py:19), coherence (src/compitum/coherence.py:41), lyapunov_control (src/compitum/control.py:15), router orchestration and certificate (src/compitum/router.py:25,80,147).

