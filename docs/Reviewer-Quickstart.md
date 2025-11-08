---
title: Reviewer Quickstart
description: One-shot path, claims-to-evidence map, and sanity checks for artifact evaluation.
---

# Reviewer Quickstart

Purpose

- Provide an unimpeachable, offline path to reproduce our key results and to verify our claims with minimal friction.

Core Claims (What We Do)

- Deterministic router: decisions are deterministic given fixed inputs and seeds; no judge models used for selection.
- Geometry-aware updates: SPD metric learning + Lyapunov-inspired trust-region updates ensure stability.
- Constraint-aware selection: routing respects feasibility with explicit shadow prices (no hidden heuristics).
- Better regret: on bounded RouterBench panels, Compitum reduces regret vs. baselines under equal budgets.

Evidence Map (Where to Verify)

- Determinism & Invariants: `tests/invariants/` (router determinism, SPD geometry, coherence, constraints, control). Run: `pytest -q tests/invariants`.
- Certificates & Authenticity: `tests/certificates/` + `tools/verify_certificate.py` (jsonschema + canonical SHA-256). Run: `python tools/verify_certificate.py <cert.json>`.
- RouterBench Results (bounded): `reports/routerbench_report.md` and docs `RouterBench-Summary`. Generate: `python tools/generate_routerbench_report.py`.
- Fixed-WTP CIs: `reports/fixed_wtp_summary.{json,md}` → docs `Results-Fixed-WTP`. Generate: `python tools/generate_eval_tables.py`.
- Artifact Manifest: `reports/artifact_manifest.json` (SHA-256 of key outputs). Generate: `python tools/generate_artifact_manifest.py`.

One‑Shot (Offline) Reproduction

```bat
make peer-review
python tools\generate_eval_tables.py
```

This produces:

- `reports/report_release.html` (consolidated results)
- `reports/fixed_wtp_summary.{json,md}` (WTP=0.1, 1.0 with 95% CIs)
- `reports/routerbench_report.md` (bounded RouterBench comparison)
- `reports/artifact_manifest.json` (artifact list + SHA-256)

RouterBench (bounded, separate venv)

1) Create and activate: `.venv-routerbench`, install `src/routerbench/requirements.txt`.
2) Ensure dataset present at `data/routerbench_5shot.pkl` (we do not redistribute proprietary datasets).
3) Run: `scripts\run_routerbench.bat --config=data/routerbench/evaluate_routers.yaml --local`.

Sanity Checks

- No network: bounded evidence paths use local, licensed inputs; configs point to local resources.
- Pinned environment: requirements and environment snapshots are present; seeds fixed.
- Determinism: router and batch determinism tests pass.
- Fairness: equal prompts, identical budget/accounting; oracle excluded from baselines; regret computed per unit then aggregated.

Threats to Validity (Known Limits)

- Dataset redistribution: reviewers must fetch RouterBench from its source; we validate paths and hashes locally.
- Panel composition: evaluations are bounded; broader sweeps are optional and excluded from default gates.
- Environment drift: provided environment snapshots and pins; rebuild instructions included.

Pointers

- Full protocol: `docs/PEER_REVIEW.md`
- Artifact README: `docs/Artifact-README.md`
- Fairness notes: `docs/RouterBench-Fairness.md`

