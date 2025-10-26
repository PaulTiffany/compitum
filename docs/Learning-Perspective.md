# Learning Perspective (cs.LG)

This note presents Compitum in standard ML terms: a constrained contextual routing problem with scalarized utility, calibrated uncertainty, a learned SPD geometry, and bounded online updates.

> Related: [cs.CL](Language-Perspective.md) · [cs.SY](Control-Perspective.md) · [stat.ML](Statistical-Notes.md) · [SRMF ⇄ Lyapunov](SRMF-as-Lyapunov.md) · [Peer Review Protocol](PEER_REVIEW.md) · [Certificate Schema](Certificate-Schema.md)

## Problem Setup

- Context x in R^D and pragmatic features x_B in R^4.
- Model set M. Each model m in M has base cost and a center mu_m in feature space.
- Utility at fixed willingness-to-pay lambda: U(x, m) = quality - lambda*(latency + cost) - beta_d*dist_M(x, mu_m) + beta_s*log p_M(x).
- Constraints: A x_B <= b and capabilities; enforced before selection.

Code anchors: src/compitum/energy.py:33 (utility), src/compitum/metric.py:23,39 (SPD metric and distance), src/compitum/coherence.py:41 (KDE prior), src/compitum/constraints.py:36 (feasibility).

## Decision Rule

- Feasibility-first: filter m by capabilities and A x_B <= b.
- Selection: choose argmax_m U(x, m) among feasible.
- Certificate: emit utility components, feasibility and approximate local shadow prices (finite-difference diagnostics), boundary diagnostics (gap/entropy/uncertainty), and trust-region state; see src/compitum/router.py:25,80.

## Learning Components

- Geometry: low-rank SPD Mahalanobis metric M = L L^T + delta*I; PD guaranteed by construction. Defensive Cholesky and delta adjustment; src/compitum/metric.py:23,39.
- Online adaptation: surrogate gradient in L with trust-region step cap (EMA + integral); src/compitum/metric.py:106, src/compitum/control.py:15. We say "Lyapunov-inspired"; no formal proof claimed.
- Predictors and calibration: component regressors with isotonic calibration and quantile bounds; src/compitum/predictors.py.
- Coherence prior: KDE log-density in whitened coordinates under M; clipped to bound influence; src/compitum/coherence.py:41. beta_s sweeps show robustness.

## Relation to ML Literature

- Constrained contextual routing: feasibility-first selection resembles constrained contextual bandits/knapsacks, with scalarized objectives at fixed lambda.
- Selective classification/deferral: boundary ambiguity (gap/entropy/uncertainty) aligns with abstention strategies to improve risk-sensitive utility.
- Metric learning: Mahalanobis (ITML/LMNN-style) geometry with low-rank structure and online updates.
- Calibration: isotonic regression and reliability curves for uncertainty evaluation.
- Density priors: KDE as an auxiliary regularizer; bounded effect via clipping and small beta_s.

## Evaluation Protocol

- Fixed-lambda slices: lambda in {0.1, 1.0} (headline) and a sensitivity grid for frontier plots.
- Pairing: compute deltas per evaluation unit against best baseline; aggregate via panel means; bootstrap CIs.
- Metrics: regret, win-rate, compliance rate; calibration (reliability curves, Spearman rho(uncertainty, |regret|)); deferral quality (boundary AUROC/AP); stability (rho(shrink in trust-radius, future regret decrease)).

Helpers: tools/analysis/cei_report.py, tools/analysis/reliability_curve.py, tools/analysis/control_kpis.py.
Fixed-lambda summary (paired bootstrap):

```bat
python tools\analysis\lg_summary.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --lambdas 0.1 1.0 ^
  --bootstrap 1000 ^
  --out-json reports\lg_summary.json ^
  --out-md reports\lg_summary.md
```

## Robustness and Limits

- Robustness: modest beta_s and rank sweeps do not flip headline conclusions; coherence is bounded by clipping; update stride affects throughput, not correctness.
- Limits: shadow prices are approximate local diagnostics (report-only; not KKT duals). Trust-region is Lyapunov-inspired without formal proof.

## Reproducibility

- Determinism: Hypothesis derandomized in CI; demo seeds fixed.
- Snapshot env with `pip freeze` and system info; attach CEI, reliability curve, control KPIs, and fixed-lambda summaries.

## Instantaneous Feedback vs. Delayed Reward (Significance)

- Mechanistic surrogates at decision time
  - At step t, Compitum observes judge-free, endogenous signals S_t = {feasibility, gap, entropy, uncertainty, trust-radius} and applies a bounded update. No temporal credit assignment is required; there is no reward delay.
  - In contrast, bandit/RL settings often observe a delayed, noisy reward R_{t+tau}, requiring credit assignment across tau and increasing estimator variance.

- Variance and latency intuition
  - Let theta be parameters of the geometry (L). Updates use grad_theta U(x_t, m; theta_t) with a capped step (trust-region). Using S_t as full-information surrogates reduces update latency and empirically reduces variance relative to delayed bandit feedback.
  - We do not claim new bounds; we make an empirical hypothesis: near-zero-latency internal signals lead to faster regret reduction under bounded steps than delayed judge signals.

- Empirical proxies (what to report)
  - CEI: deferral quality (boundary vs. high-regret), calibration (rho(uncertainty, |regret|)), stability (rho(shrink, future improvement)), compliance (~100%).
  - Control KPIs: shrink/expand events and the correlation between shrink events and future regret decrease.
  - Reliability curve: monotone increase of |regret| with uncertainty bins.

- Free-energy view (engineering)
  - The update loop minimizes a scalarized "free-energy" objective at decision time (quality - lambda*cost - beta_d*distance + beta_s*log-density). Instantaneous feedback reduces friction in that optimization by eliminating reward delay; the trust-region caps steps to maintain stability.

- Claim boundary (honest positioning)
  - Same data-generation process as RL, different estimator: we use full-information, judge-free surrogates available at decision time. We do not introduce a judge model or delayed reward; conclusions are empirical and robust across lambda, beta_s, and rank sweeps.
