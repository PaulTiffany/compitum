% How We Evaluate Matbench (Offline)

This page outlines our offline-first, reproducible evaluation path for Matbench-style tasks, and
the manual GitHub workflows that package artifacts for review.

Principles
- Offline-first: no network calls or external datasets required.
- Conservative reporting: uncertainty via bootstrap CIs; avoid comparative claims by default.
- Provenance: attestation with file hashes, environment info, and git commit when available.

CLI pipeline (local)
- Generate demo CSV:
  - `python examples/generate_matbench_demo.py --out data/matbench_demo.csv`
- Calibrate λ (kappa − λ·leak) on validation; report test AURC with CIs:
  - `python tools/calibrate_matbench_srmf.py --path data/matbench_demo.csv --objective-col y_true --mode max --topk-grid 1,5,10 --lambda-grid 0.0,0.5,1.0 --bootstrap 1000 --seed 0 --out-json reports/matbench_calibration.json --scores-out reports/matbench_scores_test.csv`
- Evaluate regret with tuned λ (including per-group curves if `group` exists):
  - `python tools/eval_matbench_regret.py --path data/matbench_demo.csv --objective-col y_true --mode max --use-srmf --lambda-weight $(jq -r .best_lambda reports/matbench_calibration.json) --topk-grid 1,5,10 --group-col group --out-csv reports/matbench_regret.csv --out-json reports/matbench_regret.json --out-group-csv reports/matbench_regret_groups.csv --bootstrap 1000 --seed 0`
- Attestation:
  - `python tools/generate_matbench_attestation.py --input-csv data/matbench_demo.csv --calibration-json reports/matbench_calibration.json --regret-json reports/matbench_regret.json --out reports/matbench_attestation.json`

Manual GitHub workflows (no releases)
- `matbench_offline` (workflow_dispatch): calibrates and evaluates on a committed CSV path and uploads artifacts:
  - Calibration JSON (tuned λ, AURC, CI, splits)
  - Test scores CSV (kappa, leak, score)
  - Regret@k CSV and summary JSON (AURC, optional CI)
  - Attestation JSON (hashes, env, commit)
- `materials_audit` (workflow_dispatch): runs live audit with `secrets.MP_API_KEY` and uploads CSV.

Artifacts and review
- Prefer linking the attestation JSON alongside CSV/JSON outputs.
- Provide the exact CLI commands used (with seeds) and the CSV schema.
- Keep claims conservative; regret is reported as opportunity loss vs. oracle.



## Baselines and Plots
- Baseline regret: tools/eval_baseline_regret.py --model ridge --folds 5 --topk-grid 1,5,10 --plot
- If matplotlib present, PNGs are written to reports/.

## Emergent Layers
- Explore SRMF layers via quantiles or k-means: tools/explore_matbench_layers.py
- Outputs CSV and JSON with per-layer AURC; use to guide ? tuning per-layer.



## Exporting a Task CSV
- Export via Materials Project (requires MP_API_KEY): 	ools/export_matbench_task_csv.py --from-mp --elements La Ni O --nelements 3 --objective band_gap --limit 500 --out data/mp_matbench_task.csv 
- Or generate a synthetic task: 	ools/export_matbench_task_csv.py --offline-mock --out data/matbench_task.csv 
- Then run the offline workflow or local pipeline on the exported CSV.

