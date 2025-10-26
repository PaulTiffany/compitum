---
title: RouterBench Fairness
description: Evaluation fairness notes for RouterBench maintainers and contributors.
---

# RouterBench Fairness Notes

Purpose

- Document how we use RouterBench fairly: equal prompts, budgets, cost accounting, and seeds across baselines and Compitum; transparent panel composition and configs.

Panel and Configs

- Panel Summary: see {doc}`Panel-Summary` for tasks, models, WTP slices, and eval unit counts detected from the latest runs.
- Config files (examples):
  - `data/rb_clean/evaluate_routers.yaml`
  - `data/rb_clean/evaluate_routers_multitask.yaml`
- WTP grid: fixed slices used for headline tables are {0.1, 1.0}. Sensitivity grid for frontier plots includes {1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0}.

Fairness Principles

- Equal prompts: baselines and Compitum operate over the same prompt sets for each evaluation unit.
- Identical budgets and accounting: WTP (lambda) and token/cost accounting are identical across all systems for the unit being compared.
- Paired comparisons: “best baseline” is computed per evaluation unit at fixed WTP; regret and deltas are paired before aggregation.
- Seeds and grids: seeds are fixed; any limited hyperparameter sweep is preregistered, small, and frozen before final evaluation on held-out splits (see PEER_REVIEW).
- Oracle exclusion: oracle rows are excluded from baseline comparisons.

Reproducibility and Transparency

- Offline runs: evaluation uses locally cached, licensed inputs; no network fetching is performed by default scripts.
- Artifact integrity: we generate a manifest with SHA-256 checksums for key outputs.
- Evidence breakdown: in addition to panel averages, we provide per-baseline win rates, frontier gaps (with 95% bootstrap CIs), and per-task summaries.

Agreement and Diagnostics

- Per-baseline agreement tooling: see `tools/per_baseline_agreement.py` for measuring agreement rates across models/routers.
- Column inspection: see `tools/inspect_eval_cols.py` to verify expected columns and accounting fields in the evaluation CSVs.

Versioning and Compatibility

- RouterBench integration is kept in a separate folder and not modified by default CI or linters.
- We welcome PRs to update or extend fairness checks, panel definitions, or cost accounting notes. Please include YAML changes and a short justification of the impact on comparability.

