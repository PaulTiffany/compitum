# Fusion Offline Evaluation (Examples)

This folder contains a minimal, practical setup to run offline evaluation and an Lp sweep over shot CSVs for early-warning analysis.

## Data Schema (per shot CSV)

Required columns:
- `time_ms` (float, milliseconds)
- `q_min` (float)

Optional columns (used if present):
- `Te_core`, `ne` (additional features can be added; missing dims are zero-filled)

State layout in the adapter:
- index 0 = Te_core
- index 1 = ne
- index 2 = q_min
- remaining entries are zeros up to `state_dim`

## Quick Start

1) Generate synthetic samples (optional):

```bash
python examples/fusion/make_synthetic_shots.py --out data/samples/fusion_shots
```

2) Offline evaluation (single threshold):

```bash
python tools/eval_fusion_offline.py data/samples/fusion_shots \
  --state-dim 8 --rank 4 --curvature-alarm 0.5 \
  --out reports/fusion_offline_metrics.csv
```

3) Lp sweep (p in [1,2]):

```bash
python tools/eval_lp_sweep.py data/samples/fusion_shots \
  --state-dim 8 --rank 4 --curvature-alarm 0.5 \
  --p-grid "1.0,1.25,1.5,1.75,2.0" \
  --lambda 1.0 \
  --out reports/fusion_lp_sweep.csv
```

Notes:
- For unit normalization, supply `scales` when constructing `PlasmaMonitor` programmatically; CLI-level support can be added if desired.
- The Lp sweep reports risk (miss rate), FAR, median lead-time, curvature, and Omega = risk + ?·FAR, plus p* picks.

