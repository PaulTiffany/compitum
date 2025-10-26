---
title: Instantaneous Feedback (cs.LG Deep Dive)
description: Why judge‑free, near‑zero‑latency signals change the learning dynamics relative to delayed reward.
---

# Instantaneous Feedback (Deep Dive)

We articulate why mechanistic, judge‑free signals at decision time (feasibility, boundary ambiguity, calibrated uncertainty, trust‑region state) can improve the learning dynamics relative to delayed reward.

## Setup

- At step t, Compitum observes S_t = {feasible_t, gap_t, entropy_t, uncertainty_t, r_t} and updates geometry parameters θ_t (e.g., L_t) with a capped step: θ_{t+1} = θ_t − η_eff ∇_θ U(x_t, m_t; θ_t), with η_eff ≤ κ/(‖∇‖+ε).
- No external judge is consulted; the signals are endogenous and instantaneous.
- In bandit/RL settings, feedback often arrives as a random reward R_{t+τ}, τ ≥ 1, requiring temporal credit assignment and introducing additional variance/noise.

## Intuition: Variance and Latency

- Immediate surrogates S_t reduce update delay and eliminate temporal credit assignment. Under bounded steps, this empirically accelerates regret reduction.
- Conceptually, S_t serve as full‑information control variates for ∇_θ U, whereas delayed rewards require estimating long‑horizon contributions.
- We do not claim new regret bounds; our claim is empirical and supported by CEI and control KPIs.

## What to Measure

- CEI components: deferral quality (AP/AUROC), calibration (Spearman ρ and reliability curve), stability under trust‑radius shrink, compliance.
- Control KPIs: shrink/expand counts, r statistics, Spearman ρ(shrink, future improvement).
- Fixed‑λ regret/win‑rate with paired bootstrap CIs across panels.

## Engineering Consequences

- No judge model is needed; evaluation remains deterministic and offline.
- Updates remain PD‑safe via M = L L^T + δI and Cholesky checks; step sizes are capped by SRMF.
- Bounded coherence prior ensures robustness to KDE bandwidth and β_s sweeps.

## Limits and Scope

- This is not a theoretical reduction of RL; it is a practical observation about estimator variance and latency under bounded online updates.
- Shadow prices are approximate diagnostics; selection is feasibility‑first argmax U.
- The approach complements, not replaces, delayed‑reward formulations where only bandit feedback is available.

