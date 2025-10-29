---
title: Control Perspective (cs.SY)
description: Closed-loop view of Compitum's instantaneous, judge-free feedback and trust-region control for systems and control reviewers.
---

# Control Perspective

Related: [cs.LG](Learning-Perspective.md), [cs.CL](Language-Perspective.md), [stat.ML](Statistical-Notes.md), [SRMF & Lyapunov](SRMF-as-Lyapunov.md), [Peer Review Protocol](PEER_REVIEW.md), [Certificate Schema](Certificate-Schema.md)

This note frames Compitum as a closed-loop decision system with instantaneous, judge-free feedback and a trust-region controller that stabilizes online adaptation. It is intended for cs.SY reviewers.

## Closed-Loop Decomposition

Signals (per decision step t):

- Input context `x_t` (features) and pragmatic constraints `x_B,t` (Banach features).
- Routing policy `u_t = argmax_m U(x_t, m; θ_t)` over feasible models (`A x_B,t <= b`).
- Process measurements `y_t = {gap_t, entropy_t, uncertainty_t, feasibility_t, trust_radius r_t}` emitted in a routing certificate.
- Controller state `z_t = {L_t (metric factor), r_t (trust radius), EMA/integral accumulators}`.

Loop:

1) Measurement (instantaneous): compute U, boundary diagnostics (gap/entropy/uncertainty), feasibility; emit certificate.
2) Control law (SRMF): update trust radius `r_{t+1}` and cap the effective step size for metric update.
3) Adaptation: apply bounded update to `L` (SPD metric factor), then re-factor `M = L L^T + δ I` (Cholesky).

Design choices: instantaneous internal feedback (no judge model), feasibility-first selection, and a capped-step trust-region controller.

## Trust-Region Controller (SRMF)

- Update law (code anchor: `src/compitum/control.py:15`)
  - `r_{t+1} = clip(r_t + f(EMA(d_t), integral(d_t)), r_min, r_max)`
  - `η_cap = κ / (||∇|| + ε)`
  - Effective step for metric update: `η_eff = min(η_user, η_cap)`
- Anti-windup: integral term decays; `r` is clipped to `[r_min, r_max]`.
- Interpretation: when contradiction (distance or its proxy) is persistently high, the controller shrinks `r` and caps steps; when consistently low, it allows gentle increases in `r`.

## Metric Update (Bounded, PD by Construction)

- Geometry (anchors: `src/compitum/metric.py:23,39,106`)
  - `M_t = L_t L_t^T + δ I` (SPD); updated via surrogate gradient in `L`.
  - After each step, recompute Cholesky; if needed, increase `δ` defensively (ensures PD).
  - Frobenius-norm clip on `L` enforces an explicit bound on metric magnitude.

## Stability Indicators (Operational)

We do not claim a formal Lyapunov proof; instead, we provide Lyapunov-inspired operational indicators:

  - `I_cap = I / (||grad|| + I)`
- Feasibility by construction: constraints `A x_B <= b` ensure safe action space before optimization.
- Monotone corrections (empirical): trust-radius shrink events correlate with future regret reductions.
- Instantaneous feedback: zero-delay internal measurements (gap/entropy/uncertainty/feasibility) avoid judge-feedback latency.

## Delay and Noise (Why Instantaneous Feedback Helps)

- Judge-based designs introduce delay/noise in reward, complicating stability and slowing correction.
- Compitum measures endogenous signals at time `t` and adapts immediately with a capped step.

## Control KPIs (What to Inspect)

- Step-size capping: distribution of `η_eff` and its relation to `||∇||`.
- Trust-radius events: counts of shrink/expand, and their effect on regret trends.
- Metric health: PD checks (Cholesky success, `δ` adjustments), `||L_t||_F` statistics.
- Certificate coverage: fraction of decisions with boundary flags under which deferral would be suggested.

Helper script (from certificates + eval CSV):

```bat
python tools\analysis\control_kpis.py ^
  --certs reports\certificates.jsonl ^
  --eval data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\control_kpis.json ^
  --out-md reports\control_kpis.md
```

## Stability Evidence (0.1.1)

- Lyapunov proxy decay under zero drift; saturation under sustained drift; recovery when drift ceases.
  - `tests/invariants/test_invariants_control_lyapunov.py`
- ΔV proxy sequences: Lyapunov + small distance term is bounded over short sequences.
  - `tests/invariants/test_invariants_control_sequences.py`, `tests/invariants/test_invariants_control_deltaV_strong.py`
- Combined updates (metric+controller) keep a simple stability proxy finite and bounded.
  - `tests/invariants/test_invariants_control_combined_proxy.py`

## Decision Rule (for completeness)

- Feasibility-first: filter models by capabilities and `A x_B <= b` (`src/compitum/constraints.py:36`).
- Selection: choose `argmax U(x_t, m; θ_t)` among feasible.
- Certificate: emit utility components, feasibility and approximate local shadow prices (diagnostic), boundary diagnostics, and drift/trust-region status (`src/compitum/router.py:25,80`).

## References (Pointers)

- Trust-region / step-size control in online optimization.
- Mahalanobis (SPD) metric learning and low-rank parameterization.
- Kernel density estimation as a prior; shrinkage covariance.

