---
title: Trust From Regret
description: How instantaneous, judge‑free feedback and bounded updates foster trust by minimizing regret.
---

# Trust From Regret

We frame “trust” as calibrated expectation of low future regret enabled by near‑zero‑latency internal feedback and bounded updates.

## Definition (Operational)

- Trust rises when recent decisions show: (i) low regret at fixed λ, (ii) good deferral quality on ambiguous cases, (iii) calibrated uncertainty, (iv) stable corrections after contradiction signals, and (v) constraint compliance.
- We measure these via the Control‑of‑Error Index (CEI) and Control KPIs.

## Instantaneous Feedback → Faster Correction

- Compitum emits endogenous, judge‑free signals at decision time: feasibility, boundary ambiguity (gap/entropy/uncertainty), and drift/trust‑radius state.
- A Lyapunov‑inspired trust‑region caps update step sizes, turning contradiction signals into gentle, stable adjustments of the learned SPD metric.
- This reduces correction latency and noise vs. delayed judge schemes, supporting lower future regret.

## Evidence (What to Report)

- Fixed‑WTP performance: regret and win‑rate vs. best baseline at λ ∈ {0.1, 1.0}; bootstrap CIs.
- CEI components (from existing CSVs/certificates):
  - Deferral quality (boundary vs. high‑regret): AP, AUROC.
  - Calibration: Spearman ρ(uncertainty, |regret|) and reliability curve.
  - Stability: Spearman ρ(shrink in trust‑radius vs. future regret decrease).
  - Compliance: feasible rate ≈ 1.
- Control KPIs: trust‑radius event counts (shrink/expand/steady), r summary stats, shrink→improve correlation.

## Helper Commands

- CEI report

```bat
python tools\analysis\cei_report.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\cei_report.json ^
  --out-md reports\cei_report.md
```

- Reliability curve

```bat
python tools\analysis\reliability_curve.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --bins 10 ^
  --out-csv reports\reliability_curve.csv ^
  --out-md reports\reliability_curve.md ^
  --out-png reports\reliability_curve.png
```

- Control KPIs

```bat
python tools\analysis\control_kpis.py ^
  --certs reports\certificates.jsonl ^
  --eval data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\control_kpis.json ^
  --out-md reports\control_kpis.md
```

## Notes

- “Lyapunov‑inspired” emphasizes bounded, stabilizing updates without claiming a formal proof.
- Approximate shadow prices are report‑only diagnostics; selection is feasibility‑first argmax U.
- Coherence prior is bounded (clipping, small β_s); conclusions robust to modest β_s changes.

