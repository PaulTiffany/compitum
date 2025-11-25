# Matbench (Offline Adapter)

This page documents the offline-first adapter and CLI for evaluating SRMF proxies on
Matbench-style CSV data. The goal is to provide reproducible, conservative analyses without
external dependencies or network calls.

Scope and assumptions
- Input: a CSV with columns `band_gap`, `density`, `nsites`, `formation_energy_per_atom`.
- Optional columns for reporting: `material_id` and `formula_pretty` (you may supply alternative
  column names via CLI flags).
- Optional label column (e.g., `label_candidate = 0/1`): enables simple metrics with optional
  bootstrap confidence intervals.

CLI usage
- Per-row SRMF evaluation and CSV export:
  - `python tools/eval_matbench_srmf.py --path data.csv --id-col mid --formula-col formula --out reports/matbench_srmf.csv`
- With labels and metrics (plus bootstrap CIs):
  - `python tools/eval_matbench_srmf.py --path data.csv --id-col mid --formula-col formula --label-col label_candidate --bootstrap 1000 --seed 0 --out reports/matbench_srmf.csv --metrics-out reports/matbench_srmf_metrics.json`

Outputs
- CSV: `material_id, formula, srmf_phase, curvature_kappa, stability_leak, prediction`.
- Optional JSON metrics: `precision, recall, accuracy` and bootstrap intervals when labels are present.

Claims & limitations
- The adapter maps material summaries to SRMF proxy coordinates for exploration and monitoring.
- Results are dataset-dependent; we report uncertainty where relevant and avoid comparative claims.
- Live loaders (e.g., matminer/matbench) are intentionally out-of-scope here and may be added later
  behind optional dependencies and explicit flags.

