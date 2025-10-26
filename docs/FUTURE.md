---
title: Future Directions
description: Promising next steps for research and engineering in Compitum.
---

# Future Directions

This page outlines promising, scoped next steps that extend Compitum's core: constraint‑aware routing, SPD metrics, and Lyapunov‑inspired controllers. These items are not required for launch.

## Dual Shadow Pricing (Online Primal–Dual)

- Goal: Introduce bona fide dual variables (shadow prices) λ for constraints that rise under sustained utilization pressure and fall when slack, enabling interpretable tradeoffs and budget compliance.
- Approach: Maintain λ ≥ 0 with a lightweight controller: observe usage u = A @ x_B, track EMA ũ, compute slack s = ũ − ρ·b, update λ ← clip(λ + η·s, 0, λ_max). Optionally price selection via Ũ = U − λ·u.
- Certificate: Add `constraints.dual_shadow_prices` and controller params (`eta`, `alpha`, `rho`).
- Scope: Standalone module; default off (observe‑only). Negligible overhead (vector ops in R^m).
- Status: Proposed; complements current finite‑difference boundary sensitivity.

## Robust Boundary Sensitivity

- Make epsilon configurable and scaled by constraint norm; optionally probe a small epsilon set and report a stabilized statistic.
- Emit the minimal Δb that flips argmax (critical relaxation) to aid audits.

## PyLantern Integration (Teaser)

- PyLantern (a.k.a. Fascia) explores observer‑bounded, curved computation in PyTorch. Integrating selective pieces can provide:
  - Observer‑bounded windows for controllers (trust region, duals) with typed, testable surfaces.
  - Geometry‑aware utilities where appropriate without sacrificing Torch ergonomics.
- This is exploratory and orthogonal; Compitum remains usable without PyLantern.

## Expanded Certificates

- Add optional provenance fields (constraint config hash, epsilon used, metric version) to strengthen audit trails.
- Provide compact “active constraints” summary (top‑k by magnitude) for quick dashboards.

## Evaluation Tracks

- Budget‑constrained routing tasks to quantify benefits of dual shadow pricing.
- Stress tests on boundary regimes (small utility gap, high entropy) to validate deferral behavior.

