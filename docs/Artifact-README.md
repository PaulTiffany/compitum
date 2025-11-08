---
title: Artifact README
description: Reproducibility checklist, environment, commands, and outputs for artifact evaluation.
---

# Artifact README (Reproducibility)

Purpose

- Provide a reviewer-first, offline, one-shot path to reproduce the key results and artifacts.
- Summarize environment, commands, expected outputs, and integrity checks.

Environment

- Python: 3.11+
- Platform: Windows (primary). POSIX equivalents included in {doc}`PEER_REVIEW`.
- No network access required; inputs are locally cached and licensed (RouterBench). We do not redistribute proprietary datasets.

One-Shot (Windows)

```bat
make peer-review
python tools\generate_eval_tables.py
.\.venv\Scripts\python -m sphinx -b html docs docs\_build\html
python tools\check_mojibake.py  # ensure no mojibake in README/docs
```

POSIX Equivalents

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
pytest -q && ruff check . && mypy src/compitum && bandit -q -r src/compitum -x src/routerbench
python tools/generate_eval_tables.py
python -m sphinx -b html docs docs/_build/html
```

Primary Outputs

- Reports (local):
  - `reports/report_release.html` (frontier plots, tables)
  - `reports/fixed_wtp_summary.{json,md}` (WTP = 0.1, 1.0 with 95% CIs)
  - `reports/mutation_summary.json` (Cosmic Ray summary)
  - `reports/artifact_manifest.json` (paths + SHA-256)

- Docs pages (local):
  - `docs/Per-Baseline-WinRate.md`
  - `docs/Frontier-Gap.md` (with 95% bootstrap CIs)
  - `docs/Results-By-Task.md`
  - `docs/Panel-Summary.md`

Integrity & Determinism

- Seeds are fixed for synthetic demo/predictors and evaluation scripts.
- A manifest with SHA-256 checksums is generated for key artifacts.
- Sphinx builds without warnings; evidence scripts are warning-free.
- Text hygiene: a normalization/check pair (`tools/normalize_mojibake.py`, `tools/check_mojibake.py`) keeps README/docs free of mojibake without touching src/third‑party files.

Runtime Notes (bounded panel)

- Quality gates (pytest/ruff/mypy/bandit): typically minutes on a workstation.
- Evidence generation and docs build: minutes; bounded panel keeps runtime modest.
- Full RouterBench sweeps are optional and excluded from default gates.

Badge Checklist (typical AE criteria)

- [x] No network access required for reproduction path
- [x] Exact environment and versions documented
- [x] One-shot scripts and explicit commands
- [x] Checksums/manifest for generated artifacts
- [x] Deterministic seeds and offline evaluation

For any questions or additional formats (e.g., SBOM, license summary), please open an issue. We aim to help without changing the scientific pipelines.

