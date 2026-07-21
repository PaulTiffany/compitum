# Reproducibility

Baseline

- Benchmark tool: RouterBench (vendored at `src/routerbench`) pinned to upstream commit `cc67d10` (origin/main).
- We do not modify RouterBench source for final runs. All behavior changes are applied via an external wrapper.

How We Run RouterBench (Upstream)

- Wrapper: `tools/run_routerbench_clean.py`, invoked directly with Python (also orchestrated via `tools/ci_orchestrator.py --bench-routerbench`).
  - Suppresses tokencost stdout warnings.
  - Standardizes token counts via tiktoken with `cl100k_base` fallback.
  - On Windows, fixes unsafe filename characters in per-eval CSV save.
  - Tokenizer backend can be selected with `--tokenizer-backend=tiktoken|tokencost|hf` (default: `tiktoken`).

Fast Config

- Config: `data/rb_clean/evaluate_routers.yaml`
- Run: `python tools\run_routerbench_clean.py --config=data/rb_clean/evaluate_routers.yaml --local`

Compitum Evaluation

- Our router remains in `src/compitum`. Adapter lives outside RouterBench at `tools/routerbench/routers/compitum_router.py`.
- Driver: `tools/evaluate_compitum.py`, invoked directly with Python (also orchestrated via `tools/ci_orchestrator.py --bench-compitum`).
- Run: `python tools\evaluate_compitum.py --config=data/rb_clean/evaluate_routers.yaml`

Outputs

- RouterBench upstream run: CSV/PKL under `data/rb_clean/eval_results`.
- Compitum run: CSV under `data/rb_clean/eval_results` (Compitum eval script writes per-eval CSVs).

Tokenization Policy

- Default: `tiktoken` with `encoding_for_model` and `cl100k_base` fallback.
- Alternatives: `--tokenizer-backend=tokencost` (use library defaults) or `--tokenizer-backend=hf` (experimental; falls back to tiktoken on errors).
- Rationale: consistency across routers is prioritized over absolute counts; OpenAI models receive the most accurate counts under `tiktoken`.

Diff Against Upstream

- `src/routerbench` is a git submodule pinned to the upstream commit above; we do not carry a local patch/fork against it. Any behavior changes are applied via the external wrapper scripts described above, not by editing the submodule in place.

One-Command Report

- End-to-end (tests + both benchmarks + HTML report):
- `scripts\run_peer_review.bat` (wraps `python tools/ci_orchestrator.py --all --config=data/rb_clean/evaluate_routers.yaml --report-out=reports/report_release.html`)
- Output report path is printed at the end, under `reports/`.

Individual Steps

- Unit tests only: `python tools/ci_orchestrator.py --tests`
- RouterBench only: `python tools/ci_orchestrator.py --bench-routerbench --config=data/rb_clean/evaluate_routers.yaml`
- Compitum only: `python tools/ci_orchestrator.py --bench-compitum --config=data/rb_clean/evaluate_routers.yaml`
- Build report from latest artifacts: `python tools/ci_orchestrator.py --report-out reports/report.html`

Quality Suite (Lint/Types/Sec/Tests/Mutation)

- One command: `scripts\run_quality.bat`
- Runs: ruff (style), mypy (types), bandit (security), pytest with coverage (line + branch), cosmic-ray (mutation). Results in `reports/quality_*.json`.

Licensing & Data Use

- RouterBench inputs follow their upstream licenses; we do not redistribute proprietary datasets in this repository or distributions.
- Evaluation runs operate offline on locally cached inputs; scripts do not auto-fetch datasets or call judge models.
- Generated artifacts (JSON/CSV/HTML) are local; a SHA-256 manifest is available in `reports/artifact_manifest.json`.
