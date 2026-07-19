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
- Calibrate (this also writes the held-out test split's scores to `--scores-out`,
  with `y_true`/`kappa`/`leak`/`score` per row — and `group` too, if `--group-col` is given):
  - `python tools/calibrate_matbench_srmf.py --path data.csv --objective-col y_true --mode max --topk-grid 1,5,10 --lambda-grid 0.0,0.5,1.0 --bootstrap 1000 --seed 0 --group-col group --out-json reports/matbench_calibration.json --scores-out reports/matbench_scores_test.csv`
- Evaluate on the held-out test split:
  - `python tools/eval_matbench_regret.py --path reports/matbench_scores_test.csv --objective-col y_true --mode max --score-col score --topk-grid 1,5,10 --group-col group --out-csv reports/matbench_regret.csv --out-json reports/matbench_regret.json --bootstrap 1000 --seed 0`
  - **Important:** evaluate against `--scores-out`'s file (the held-out test rows), not against the original `data.csv` with `--use-srmf --lambda-weight <best>`. Re-scoring the original CSV would re-include the train/val rows that were used to select λ, leaking calibration signal into the reported regret and making it incomparable to `eval_baseline_regret.py`'s honestly out-of-fold-evaluated baseline.

Claims and limitations
- SRMF mapping is a proxy for manifold geometry and stability signals; it is not a surrogate for
  ab initio calculations. We report uncertainty, avoid comparative claims by default, and keep
  live integrations gated.



## Attestation and Groups
- Attestation: tools/generate_matbench_attestation.py
- Per-group regret: pass `--group-col group --out-group-csv reports/matbench_regret_groups.csv` to the same `eval_matbench_regret.py` call above (no separate run needed) -- requires `--group-col group` to have also been passed to `calibrate_matbench_srmf.py` so the held-out scores file carries the group column.



## Baselines and Layers
- Baseline CV regret CLI: tools/eval_baseline_regret.py
- Emergence exploration: tools/explore_matbench_layers.py
- Example: use quantile layers on band_gap and report AURC per-layer.

