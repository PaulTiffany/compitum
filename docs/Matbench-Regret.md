# Matbench Regret (Offline)

This document defines the regret metrics we use for Matbench-style evaluations and how to
reproduce results offline with CSVs. The approach is conservative: it reports uncertainty,
uses fixed seeds, and avoids comparative claims by default.

Terminology
- Objective (`y_true`): numeric property to maximize or minimize (declared via `--mode`).
- Score: ranking function for selection. By default, we use SRMF proxies: `kappa − λ·leak`.
- Regret@k: Difference between oracle top‑k utility and model-selected top‑k utility, with
  normalized variant dividing by oracle utility.
- AURC: Area under the normalized regret curve over k (lower is better).

Offline tools
- Calibration: choose λ to minimize validation AURC, report held‑out test AURC with CIs.
  - `tools/calibrate_matbench_srmf.py`
- Evaluation: compute Regret@k and AURC using the chosen λ (or a provided score column).
  - `tools/eval_matbench_regret.py`

CSV schema
- Required features for SRMF: `band_gap, density, nsites, formation_energy_per_atom`.
- Required objective: e.g., `y_true` (set via `--objective-col`).
- Optional: `material_id` and `formula_pretty` for reporting.

Reproducible run
- Calibrate:
  - `python tools/calibrate_matbench_srmf.py --path data.csv --objective-col y_true --mode max --topk-grid 1,5,10 --lambda-grid 0.0,0.5,1.0 --bootstrap 1000 --seed 0 --out-json reports/matbench_calibration.json --scores-out reports/matbench_scores_test.csv`
- Evaluate with tuned λ:
  - `python tools/eval_matbench_regret.py --path data.csv --objective-col y_true --mode max --use-srmf --lambda-weight $(jq -r .best_lambda reports/matbench_calibration.json) --topk-grid 1,5,10 --out-csv reports/matbench_regret.csv --out-json reports/matbench_regret.json --bootstrap 1000 --seed 0`

Claims and limitations
- SRMF mapping is a proxy for manifold geometry and stability signals; it is not a surrogate for
  ab initio calculations. We report uncertainty, avoid comparative claims by default, and keep
  live integrations gated.



## Attestation and Groups
- Attestation: tools/generate_matbench_attestation.py
- Per-group regret: tools/eval_matbench_regret.py --group-col group --out-group-csv reports/matbench_regret_groups.csv



## Baselines and Layers
- Baseline CV regret CLI: tools/eval_baseline_regret.py
- Emergence exploration: tools/explore_matbench_layers.py
- Example: use quantile layers on band_gap and report AURC per-layer.

