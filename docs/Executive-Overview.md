---
title: Executive Overview
description: What Compitum is, why it matters, where it wins, and how to evaluate it responsibly.
---

# Executive Overview

Why Compitum (in 60 seconds)

- Today, teams route across many models with different quality, latency, and price. Heuristics and
  ad‑hoc cascades leak money and create opaque failure modes; judge‑based feedback loops are slow,
  hard to audit, and risky in regulated settings.
- Compitum makes the tradeoff explicit and auditable: U = performance − lambda · cost, with hard
  constraints and a routing certificate that shows exactly why a choice was made, so you can fix the
  right thing fast.
- Outcome: near‑frontier behavior (efficient use of spend) with constraint compliance by design and
  immediate, mechanistic signals for operations, safety, and research.

What is Compitum

- A cost–quality aware routing engine that picks among models using a simple utility: U = performance − lambda · cost, subject to hard constraints. Every decision emits a mechanistic routing certificate for audit and ablations.

Why it’s different

- Mechanistic transparency: the certificate exposes utility components, constraint feasibility and shadow prices, boundary diagnostics (gap, entropy, sigma), and drift monitors.
- Constraint‑first: policy/compliance limits are built in; infeasible routes are rejected by construction.
- Responsible competitiveness: we report per‑baseline win rates at fixed WTP (lambda) and frontier gap with CIs, showing near‑frontier behavior even when “envelope wins” are rare.

Results (high level)

- Mean regret at fixed WTP slices (0.1, 1.0) on a bounded panel is the primary claim (see
  {doc}`PEER_REVIEW` for exact figures and per‑task breakdown); envelope win rate is reported as
  illustrative secondary context, not a pass/fail bar, and is expected to be low on this kind of
  flat, one‑shot benchmark data (see PEER_REVIEW's Methodological Limitation note).
- Constraint compliance ~100% by design.

Ethics and Reproducibility

- Offline, deterministic pipeline with fixed seeds; no judge‑based model calls.
- Licensed inputs only; we do not redistribute proprietary datasets. Artifacts are local with SHA‑256 manifest.
- 100% line+branch coverage; every release-critical module fully mutation-classified with only 2
  accepted, documented defensive survivors remaining (see `MUTATION_HARDENING_STATUS.md`); lint/type/security
  checks are clean. Docs build warning‑free.

How to try (Windows one‑shot)

```bat
make peer-review
python tools\generate_eval_tables.py
.\.venv\Scripts\python -m sphinx -b html docs docs\_build\html
```

What to read next

- Results Summary — {doc}`Results-Summary`
- Frontier Gap (with CIs) — {doc}`Frontier-Gap`
- Per‑Baseline Win Rate — {doc}`Per-Baseline-WinRate`
- Panel Summary — {doc}`Panel-Summary`
- Routing Certificate — {doc}`Certificate-Schema`
- Math Brief (plain language) — {doc}`Math-Brief`
- Peer Review (artifact guide) — {doc}`PEER_REVIEW`
- Artifact README (AE checklist) — {doc}`Artifact-README`
- RouterBench Fairness — {doc}`RouterBench-Fairness`

Where it fits vs. alternatives

- Heuristics/cascades: simple but brittle; Compitum gives you a single utility, constraint handling,
  and an audit trail for every decision.
- Judge‑based reward loops: flexible but opaque and risky; Compitum uses mechanistic, local signals
  you can inspect and test.
- Black‑box gates: may win panels but are hard to debug; Compitum favors near‑frontier efficiency with
  certificates that turn “why” into engineering actions.

When not to use

- If you have a single model and a fixed, non‑negotiable budget/latency, a simple static policy works.
- If you require external judges or human moderation in the loop, keep them outside the router and
  use the certificate to decide when to defer.

What to evaluate next (decision checklist)

- Does near‑frontier behavior hold on your tasks at your WTP slices?
- Do certificates help you resolve incidents faster (e.g., binding constraints, ambiguity signals)?
- Can you reduce spend or latency at parity quality for key workloads?
## Contributions (At a Glance)

- Control of Error for routing: instantaneous, judge‑free feedback signals (feasibility, boundary ambiguity, calibrated uncertainty) exposed per decision via a routing certificate.
- Stable online adaptation: Lyapunov‑inspired trust‑region controller caps update step sizes; SPD metric updates remain PD by construction.
- Constraint‑compliant decision rule: feasibility‑first argmax utility with approximate shadow prices reported for auditing.
- Evidence and tooling: fixed‑WTP regret/win‑rate with paired bootstrap CIs; Control‑of‑Error Index (CEI), reliability curve, and control KPIs helpers for calibration and stability.
