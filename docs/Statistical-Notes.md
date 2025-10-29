---
title: Statistical Notes (stat.ML)
description: Statistical methodology and evaluation details for Compitum aimed at statistics/ML reviewers.
---

# Statistical Methodology

> Related: [cs.LG](Learning-Perspective.md) Â· [cs.CL](Language-Perspective.md) Â· [cs.SY](Control-Perspective.md) Â· [SRMF â‡„ Lyapunov](SRMF-as-Lyapunov.md) Â· [Peer Review Protocol](PEER_REVIEW.md) Â· [Certificate Schema](Certificate-Schema.md)

This page summarizes the statistical modeling choices and evaluation procedures behind Compitum in terms familiar to stat.ML.

## Utility, Regret, and Pairing

- Scalarization: we evaluate decisions via utility U = performance âˆ’ Î» Â· total_cost at fixed willingnessâ€‘toâ€‘pay Î». This induces a paired comparison per evaluation unit.
- Regret: r = U_best_baseline âˆ’ U_compitum; lower is better. All deltas are computed per unit (paired) before aggregation, reducing variance.
- Slices: headline Î» âˆˆ {0.1, 1.0}; sensitivity grid {1eâˆ’4, 1eâˆ’3, 1eâˆ’2, 0.1, 1.0, 10.0} for frontier plots.

## Uncertainty, Calibration, and Intervals

- Predictive uncertainty: utility variance aggregates calibrated component quantiles (p05, p95) and distance variance; see src/compitum/energy.py:33, src/compitum/metric.py:59.
- Confidence intervals: nonparametric bootstrap (B = 1000 resamples) over evaluation units; 95% percentile CIs. Paired bootstrap preserves dependency structure.
- Calibration diagnostics (recommended to report alongside CEI):
  - Reliability curve: bucket uncertainty into K bins and plot mean |regret| per bin.
  - Rank correlation: Spearman Ï(uncertainty, |regret|) (included in CEI helper).
  - ECEâ€‘style summary (optional): ECE = Î£_b w_b Â· | E[|regret||b] âˆ’ t_b | with t_b a monotone target (e.g., rankâ€‘normalized). We avoid overâ€‘interpreting ECE for nonâ€‘probabilistic scales; reliability curves are primary.

Helper commands:

```bat
python tools\analysis\reliability_curve.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --bins 10 ^
  --out-csv reports\reliability_curve.csv ^
  --out-md reports\reliability_curve.md ^
  --out-png reports\reliability_curve.png
```

## KDE Prior and Bandwidth

- Coherence prior: KDE logâ€‘density evaluated in whitened coordinates, with clipping to bound influence; see src/compitum/coherence.py:41.
- Bandwidth: Scottâ€™s rule under whitened features yields consistent scaling across dimensions; alternative selectors (Silverman, CV, plugâ€‘in) can be substituted. We report robustness to modest changes of Î²_s (prior weight) and clipping range.
- Subsampling: an approximate weighted buffer maintains recent whitened residuals; coherence is auxiliary to U and bounded. Sensitivity analysis ensures headline results are robust.

## Metric Learning and Regularization

- Geometry: a lowâ€‘rank SPD Mahalanobis metric M = L L^T + Î´I shapes distances; PD ensured by Î´ and Cholesky updates; see src/compitum/metric.py:23,39.
- Update: a surrogate gradient step on L with stepâ€‘size capped by a trustâ€‘region controller (SRMF), stabilizing online adaptation; see src/compitum/metric.py:106, src/compitum/control.py:15.
- Shrinkage: Ledoitâ€“Wolf shrinkage on whitened residuals reduces variance in dispersion estimates for distance uncertainty.

## Constraints and Shadow Prices

### Duals Evidence (0.1.1)

- Shadow price diagnostics behave as expected: slack~0 when clearly non-binding; boundary>=0; monotone near binding; scale with utility units.
  - Tests: `tests/invariants/test_invariants_constraints_duals.py`, `tests/invariants/test_invariants_duals_near_binding.py`, `tests/invariants/test_invariants_duals_monotone.py`, `tests/invariants/test_invariants_duals_scaling.py`
\r\n- Constraints: linear feasibility Ax_B â‰¤ b enforced before selection; selection is argmax_U among feasible models; see src/compitum/constraints.py:36.
- Shadow prices: approximate local signals computed via finiteâ€‘difference feasibility/viability; reported for auditing; not used to drive selection (i.e., not KKT duals).

## Multiple Comparisons and Preâ€‘registration

- Multiple baselines are compared via perâ€‘unit best baseline, keeping pairing tight. Frontier plots summarize across Î». We avoid hard nullâ€‘hypothesis testing across many tasks and instead report bootstrap CIs and panel averages.
- Evaluation protocol, Î» grid, and panel composition are predeclared in docs/PEER_REVIEW.md. Seeds and environment snapshots are saved with artifacts for replicability.

## Sensitivity and Robustness

- Î²_s (prior weight): conclusions robust to modest sweeps; coherence clipping bounds influence.
- Metric rank: varied within a small range to probe biasâ€“variance; headline conclusions unchanged.
- Update stride/trustâ€‘region: impacts adaptation cadence/throughput, not decision rule correctness.

## Reproducibility

- Determinism: fixed seeds for demos; Hypothesis derandomized in CI.
- Environment: include `pip freeze` and system info with reports.
- Artifacts: fixedâ€‘WTP tables, CEI report, routerbench summary, and manifest with SHAâ€‘256.


