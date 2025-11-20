# Superconductivity (Simulated) – Examples

Simulated-only demo for automated screening with Compitum's geometric controller.

## Data Schema (per file)
- Features: `x1..xD` (floats)
- Label: `label_sc` (0/1)

## Quick Start
1) Generate synthetic dataset:
```bash
python examples/supercon/make_synthetic_dataset.py --out data/samples/supercon
```
2) Evaluate:
```bash
python tools/eval_supercon_offline.py data/samples/supercon --state-dim 8 --rank 4 --alarm 0.5 --out reports/supercon_offline_metrics.csv
```
Notes: Simulated only; no real materials data is included.