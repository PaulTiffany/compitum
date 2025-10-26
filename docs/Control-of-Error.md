---
title: Control of Error
description: Montessori’s “control of error” adapted to Compitum’s instantaneous, judge‑free feedback signals, with measurable indicators.
---

# Control of Error

Definition (Montessori → ML)

- In Montessori education, “control of error” means materials are designed so that learners can detect and correct their own mistakes without adult intervention (instant feedback, self‑correction, independence).
- In Compitum, we design the routing process with built‑in, mechanistic signals that expose errors and enable self‑correction without an external judge model.

Mechanisms (Implemented)

- Feasibility and Diagnostics
  - Hard constraints (capabilities + AxB ≤ b) enforce compliance by construction; approximate local shadow prices are reported for auditing (src/compitum/constraints.py:36).
- Ambiguity Detection
  - Boundary condition using utility gap, softmax entropy, and uncertainty flags close calls for optional deferral (src/compitum/boundary.py:19).
- Calibrated Uncertainty
  - Utility variance aggregates calibrated component quantiles and distance variance (src/compitum/energy.py:33).
- Stable Adaptation
  - Lyapunov‑inspired trust‑region control (EMA + integral) caps update sizes; SPD metric updates maintain positive definiteness (src/compitum/control.py:15; src/compitum/metric.py:23,39,106).
- Certification
  - Each decision emits a routing certificate with all signals for immediate inspection (src/compitum/router.py:25,80).

Formal Properties (Operational)

- Detect: boundary flag is predictive of higher regret than non‑boundary on average.
- Comply: constraint violation rate ≈ 0 by construction (report empirical rate).
- Correct: after drift/ambiguity spikes, trust‑region updates reduce a surrogate energy within K steps in practice.
- Certify: certificates contain sufficient fields to audit detection and correction per decision.

Control‑of‑Error Index (CEI)

Define CEI as a normalized summary of four measurable components (higher is better):

1) Deferral quality: PR/ROC of boundary flag predicting top‑q regret units (or regret > τ).
2) Calibration: monotone trend of uncertainty buckets vs. absolute regret (reliability curve score).
3) Stability: regret reduction (or bounded change) around trust‑radius shrink events (pre/post windows).
4) Compliance: 1 − violation rate.

Usage

- Compute from existing evaluation CSVs and certificate dumps; no judge model required.
- Report CEI alongside regret and win‑rate at fixed WTP to evidence instantaneous, judge‑free feedback quality.
- Helper script:

```bat
python tools\analysis\cei_report.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\cei_report.json ^
  --out-md reports\cei_report.md
```

Options: `--topq 0.1` for top‑quantile high‑regret labeling (default) or `--tau <value>` for an absolute threshold.

Notes

- Shadow prices are approximate local diagnostics via finite‑difference viability; they are reported only and do not influence selection.
- The coherence prior is KDE log‑density in whitened space; its influence is bounded by clipping and a small weight β_s.
