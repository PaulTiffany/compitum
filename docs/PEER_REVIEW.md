---
title: Peer Review Package
description: Concise, reproducible evidence for Compitum's claims with evaluation protocol, metrics, baselines, and one-shot reproduction.
---

# Peer Review Package

This document is a reviewer-oriented guide to reproduce and scrutinize Compitum's results. It defines metrics, baselines, data, and statistical procedures, and provides one-shot commands to regenerate the exact artifacts used in our tables and report.

## Reviewer Quickstart

- Verify environment and tests
  - Python 3.11+ in a fresh venv
  - `pip install -e .[dev]`
  - `make check` (ruff + mypy + bandit + pytest)

- Regenerate end-to-end artifacts (Windows, one-shot)
  - `make peer-review` (invokes `scripts\run_peer_review.bat`)

- Inspect primary outputs
  - `reports/fixed_wtp_summary.md` (table with CIs for WTP = 0.1, 1.0)
  - `reports/report_release.html` (frontier plots, baselines, ablations)
  - `reports/mutation_summary.json` (Cosmic Ray summary; score and outcomes)

### Data Availability and Artifacts

- Inputs: RouterBench task inputs are locally cached and follow their upstream licenses. We do not
  redistribute proprietary datasets in this repository or packages.
- Outputs: Scripts write local JSON/CSV/HTML and a SHA-256 manifest under `reports/` and `docs/`.
- Network: The peer-review pipeline operates offline. No judge-based model calls are used.

- Optional: LLM briefing pack
  - Snapshot for context: `https://compitum.space/docs/repo_snapshot.jsonl`
  - Instruction: Use JSONL lines with `type`, `path`, `content`; cite `path:line`.

## Claims and Contributions (Auditable)

- Low regret at fixed budget: At given willingness-to-pay (WTP) budgets, Compitum achieves lower mean regret versus strong baselines, while respecting hard constraints.
- Deterministic and reproducible: 100% line+branch test coverage; Hypothesis derandomized; fixed seeds for synthetic demo fits; scripts produce immutable, checksummed artifacts.
- Mechanistic routing certificate: Each decision emits a structured certificate with utility components, constraint status and approximate local shadow prices, boundary diagnostics, and drift monitors.
- Stable online updates: A Lyapunov-inspired trust-region controller caps effective step sizes using EMA and integral signals; we do not claim a formal Lyapunov proof in this release.

### Near-Frontier Behavior (Interpretation)

- When Compitum is close to the cost-quality frontier, the average gap to frontier is small and the at-frontier percentage is high, even if envelope wins are rare.
- This indicates Compitum trades cost and performance comparably to the strongest baseline set at the given WTP (lambda), while respecting hard constraints and offering mechanistic diagnostics.
- We therefore report both per-baseline win rates at fixed WTP and the frontier gap summary to capture near-frontier behavior without over-claiming envelope dominance.

### ASCII Notation (Quick Reference)

- performance in [0, 1]; total_cost >= 0; lambda >= 0
- U = performance - lambda * total_cost
- regret = U_best_baseline - U_compitum
- WTP slices: lambda in {0.1, 1.0}; sensitivity grid {1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0}

## Constraints Summary

- Default constraints live at `configs/constraints_us_default.yaml` and are applied as linear
  inequalities A x <= b over process variables (e.g., usage, availability, policy limits).
- The routing certificate reports `feasible` and per-constraint approximate local shadow prices
  (finite-difference viability diagnostics) so you can see which limits are active and how they shape utility.
- We report a constraint compliance rate (target ~100%) alongside results.

## Fair Evaluation Notes

- Panel definition: see {doc}`Panel-Summary` for tasks, models, WTP slices, and eval unit counts.
- Seeds and determinism: seeds are fixed across baselines and Compitum; scripts and demo fits accept `--seed`.
- Cost accounting: identical token/cost accounting across baselines and Compitum for a given WTP (lambda).
- Best baseline: computed per evaluation unit at fixed WTP; comparisons are paired.
- Configs: evaluation YAMLs live under `data/rb_clean/` (e.g., `evaluate_routers.yaml`, `evaluate_routers_multitask.yaml`).

## Evaluation Protocol (Preregistered)

- Utility and regret
  - Let performance be task score in [0,1]; cost is token or dollar cost; WTP lambda >= 0.
  - Utility per evaluation unit: U = performance - lambda * total_cost.
  - Regret per unit: r = U_best_baseline - U_compitum (lower is better).

- Primary budgets (WTP)
  - Fixed slices: lambda in {0.1, 1.0} for headline table; sensitivity grid {1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0} for frontier plots.

- Primary metrics (per task and panel averages)
  - Mean regret and P95 regret (tail behavior)
  - Win rate: fraction of evaluation units where U_compitum >= U_best_baseline
  - Avg cost delta on wins: E[cost_compitum - cost_best_baseline | win]
  - Constraint compliance rate: fraction feasible (should be 100% by design)

- Statistical procedure
  - Evaluation unit: a single (prompt, task) item under a given lambda.
  - Nonparametric bootstrap (1,000 resamples) over evaluation units; report 95% percentile CIs.
  - Paired significance: all deltas computed per unit before aggregation.
  - Seeds fixed; any randomized baselines evaluated with the same seeds/grid.

### Acceptance criteria (example)

- Panel average mean regret lower than best baseline at lambda in {0.1, 1.0}
- Win rate > 50% with non-overlapping 95% CI vs. parity for at least one lambda slice
- Constraint compliance rate >= 99.9%

## Datasets, Tasks, and Baselines

- Tasks (bounded panel)
  - Grade-school math, HellaSWAG, MBPP, selected MMLU subtopics. The exact panel is configured and cached; see scripts for the manifest.
  - Licensing: RouterBench inputs follow their upstream licenses; we do not redistribute proprietary content.

- Baselines
  - KNN/MLP/cascade gates and common RouterBench routers (budget-aware and naive). "Best baseline" is defined per evaluation unit at fixed lambda.
  - Fairness: equal prompt sets, identical lambda, identical token accountings; report any hyperparameter deviations.

## One-Shot Reproduction

We provide Windows-first commands (cross-platform alternatives below). All scripts write JSON/HTML/CSV artifacts and a manifest with checksums and provenance.

1) Quality gates (lint, type, tests)

```bat
scripts\run_quality.bat
```

Artifacts: `reports\quality_*.json` (mypy/ruff/pytest logs).

### Optional: RouterBench Tests (separate venv)

RouterBench integration tests run in a separate virtual environment to avoid import conflicts. They are optional and excluded from the default PyTest profile.

Windows:

```bat
CALL .\.venv-routerbench\Scripts\activate.bat
set PYTHONPATH=%CD%;%CD%\src
python -m pytest -q -m routerbench
```

Linux/macOS:

```bash
source .venv-routerbench/bin/activate
export PYTHONPATH="$PWD:$PWD/src"
python -m pytest -q -m routerbench
```

Notes:

- Run without coverage enforcement for this subset (coverage fail-under=100 can trip when only routerbench tests run). If you do append coverage, pass `--cov=compitum --cov-branch --cov-append` and ensure fail-under is overridden.
- The peer-review scripts already handle RouterBench evaluations separately using an isolated environment and PYTHONPATH.

2) Evaluate baselines and Compitum on bounded panel

```bat
scripts\validate_compitum.bat
```

Artifacts: `reports\routerbench_report.md`, per-task CSVs under `data\rb_clean\eval_results\...`.

3) Fixed-WTP summary with 95% CIs (peer-review table)

```bat
set PYTHONPATH=%CD%;%CD%\src
.\.venv-routerbench\Scripts\python tools\analysis\fixed_wtp_ci.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --wtps 0.1 1.0 --bootstrap 1000 ^
  --out-json reports\fixed_wtp_summary.json --out-md reports\fixed_wtp_summary.md
```

Artifacts: machine-readable JSON and a human-readable Markdown table.

4) Consolidated HTML report (frontier plots, tables)

```bat
set PYTHONPATH=%CD%;%CD%\src
.\.venv-routerbench\Scripts\python tools\ci_orchestrator.py --all ^
  --config data\rb_clean\evaluate_routers.yaml --max-evals 0 ^
  --wtp-list "0.0001,0.001,0.01,0.1,1.0,10.0" --report-out reports\report_release.html
```

Artifacts: `reports\report_release.html`

Cross-platform shortcuts

```bash
make peer-review   # calls scripts\run_peer_review.bat
```

POSIX equivalents (community-maintained)

```bash
# Create and activate venv
python3 -m venv .venv
. .venv/bin/activate

# Install
pip install -e .[dev]

# Quality gates
ruff check .
mypy src/compitum
bandit -q -r src/compitum -x src/routerbench
pytest -q

# Fixed-WTP summary (adjust latest compitum CSV path as needed)
PYTHONPATH="$PWD:$PWD/src" python tools/analysis/fixed_wtp_ci.py \
  --input data/rb_clean/eval_results/<latest-compitum-csv>.csv \
  --wtps 0.1 1.0 --bootstrap 1000 \
  --out-json reports/fixed_wtp_summary.json --out-md reports/fixed_wtp_summary.md

# Consolidated HTML report
PYTHONPATH="$PWD:$PWD/src" python tools/ci_orchestrator.py --all \
  --config data/rb_clean/evaluate_routers.yaml --max-evals 0 \
  --wtp-list "0.0001,0.001,0.01,0.1,1.0,10.0" --report-out reports/report_release.html

# Generate docs tables and build
python tools/generate_eval_tables.py
python -m sphinx -b html docs docs/_build/html
```

## Environment, Seeds, and Determinism

- Python >= 3.11; create an isolated venv; install with `pip install -e .[dev]`.
- Determinism: Hypothesis tests set derandomize=true; demo fits accept `--seed`; orchestration scripts fix seeds.
- Provenance: a manifest records git SHA, tool versions, WTP grid, and SHA-256 of generated artifacts.

Engineering note: We follow the same Control‑of‑Error ethos in development. Invariants + Hypothesis, mutation testing, strict docs gates, and audience‑specific analyses (CEI, reliability, decision/control KPIs) provide immediate, judge‑free feedback for the codebase. During release preparation we briefly edited and promptly reverted one source file; all results and reports here were generated from the current tagged code with fixed seeds and a recorded environment.

## Mutation Testing (Cosmic Ray)

We run a fresh Cosmic Ray session as part of the quality pipeline and publish both the raw dump and a compact summary.

- Summary (JSON): `reports/mutation_summary.json`
- Raw dump (NDJSON): `reports/cr_report.json`

Current run (this branch):

- jobs: 5562; mutations_seen: 5562; outcomes: killed: 5562; mutation_score: 1.0000

The summary JSON aggregates Cosmic Ray's NDJSON dump and reports mutation score = killed / total. A score near 1.0 indicates tests reliably detect injected faults.

### Cross-platform notes

- Linux/macOS: replace Windows virtualenv paths with your venv's `python`; keep `PYTHONPATH` to include `src/`.
- Some RouterBench integrations are heavy; our scripts use a bounded panel and cached inputs to keep runtime manageable.

## Results (Fill-In Table)

Populate the following with the outputs from step (3). Report both per-task values and panel averages.

### WTP = 0.1

| Task | Best Baseline Utility | Compitum Utility | lambda Utility | Mean Regret | P95 Regret | Win Rate | Avg Cost delta (wins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| grade-school math | "" | "" | "" | "" | "" | "" | "" |
| hellaSWAG | "" | "" | "" | "" | "" | "" | "" |
| MBPP | "" | "" | "" | "" | "" | "" | "" |
| Panel Avg | "" | "" | "" | "" | "" | "" | "" |

95% CIs (per column) from bootstrap over evaluation units.

### WTP = 1.0

| Task | Best Baseline Utility | Compitum Utility | lambda Utility | Mean Regret | P95 Regret | Win Rate | Avg Cost delta (wins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| grade-school math | "" | "" | "" | "" | "" | "" | "" |
| hellaSWAG | "" | "" | "" | "" | "" | "" | "" |
| MBPP | "" | "" | "" | "" | "" | "" | "" |
| Panel Avg | "" | "" | "" | "" | "" | "" | "" |

## Results (Panel Averages - Auto-Included)

The table below is auto-generated from the latest `reports/fixed_wtp_summary.md` after running the peer-review script. If empty, run `make peer-review` to refresh.

```{include} Results-Fixed-WTP.md
```

## Routing Certificate: Anatomy and Example

Each routing decision emits a certificate used for mechanistic auditing and ablations:

```json
{
  "model": "auto",
  "utility": -0.617274,
  "utility_components": {"quality": 0.598477, "latency": -0.841229, "cost": -2.290030, "distance": -2.299428, "evidence": 0.0, "uncertainty": 0.117909},
  "constraints": {"status": "optimal", "violations": [], "feasible": true, "shadow_prices": {"lambda_0": 0.0, "lambda_1": 0.0, "lambda_2": 0.0, "lambda_3": 0.0}},
  "boundary": {"winner": "auto", "runner_up": "fast", "utility_gap": 0.020410, "entropy": 1.098564, "uncertainty": 0.117909, "is_boundary": false},
  "drift": {"trust_radius": 1.088503, "drift_ema": 0.229943, "drift_integral": 2.184457, "lyapunov_function": 4.771852}
}
```
See [Certificate Schema](Certificate-Schema.md) for the full field reference -- `shadow_prices` is a dict keyed by constraint row (`lambda_0`, `lambda_1`, ...), not a list, and the top-level JSON keys are `boundary`/`drift` (not the dataclass's internal `boundary_analysis`/`drift_status` attribute names).

Reproduce locally:

```bash
compitum route --prompt "Sketch a proof for AM-GM inequality." --trace
```

Interpretation guide

- utility_components: additive terms before constraints; sign matches contribution to U.
- constraints: `feasible` and Lagrange multipliers (shadow prices) for active limits.
- boundary_analysis: `gap` (utility gap to runner-up), `entropy` (softmax uncertainty), `sigma` (spread of model scores).
- drift_status: trust-region monitors (instantaneous, EMA, and integral).

## Reviewer Checklist

- [ ] Exact Python version, dependency pins, and seeds are documented.
- [ ] Scripts produce the same CSV/JSON/HTML artifacts with matching SHA-256.
- [ ] Primary results table contains per-task values and panel averages with 95% CIs.
- [ ] Baselines share prompts, budgets (lambda), and accounting; "best baseline" is computed per evaluation unit.
- [ ] Constraint compliance reported and >= 99.9%.
- [ ] Failure modes and ablations are presented (boundary ambiguity, constraint removal, metric rank/lambda sweeps).

## Ablations and Frontiers (Suggested)

- Remove constraints + measure violations and utility changes.
- Remove boundary diagnostics + measure missed deferrals/alerts under ambiguity.
- Metric rank/lambda sweeps + stability vs. fit capacity.
- Cost-quality frontier: utility vs. cost curves across the lambda grid; report area-under-frontier.

## Known Limitations and Threats to Validity

- Bounded panels may miss OOD behavior; include targeted shifts if feasible.
- Performance/cost proxies depend on upstream pricing and tokenization; results may vary under different cost models.
- Learned baselines can be sensitive to hyperparameters; we report configs and search ranges where applicable.

## Failure Modes and Guardrails

- Ambiguous boundary (high entropy, low gap): operator deferral recommended.
- Tight constraints with high lambda: can reduce apparent performance; report compliance rate and active constraints.
- Distribution shift: drift monitors trigger smaller trust radius; document policy for escalation.

## Limitations and Threats to Validity

- Panel boundedness may hide failure modes; include OOD tasks as stress tests.
- Baseline tuning: ensure equal budgets and fair hyperparameters; document any deviations.
- Cost model simplifications (token price proxies) affect lambda sweeps; report assumptions.

## Ethical and Responsible Use

- Constraint design should reflect compliance and safety requirements; routes that violate constraints are rejected by construction.
- No redistribution of proprietary datasets; scripts fetch or expect locally-available licensed inputs.

## How to Cite

If this work informs your research, please cite the repository and version number (see `compitum --version`).

---

For convenience, a condensed results table is also maintained in {doc}`Results-Summary`.

## Per-Baseline Win Rate

```{include} Per-Baseline-WinRate.md
```

## Frontier Gap

```{include} Frontier-Gap.md
```

## Decision Rule and Control of Error

- Decision rule: among feasible models (capabilities + AxB ≤ b), select the model with maximum utility U (src/compitum/router.py:80). Coherence contributes as a bounded prior term to U (src/compitum/energy.py:33; src/compitum/coherence.py:41).
- Detect: boundary ambiguity via gap/entropy/uncertainty (src/compitum/boundary.py:19); feasibility and approximate shadow prices for auditing (src/compitum/constraints.py:36); uncertainty from component quantiles and distance variance (src/compitum/energy.py:33).
- Correct: Lyapunov-inspired trust-region control for stable metric updates (src/compitum/control.py:15; src/compitum/metric.py:106).
- Certify: routing certificate includes utility components, feasibility, boundary diagnostics, and drift/trust radius (src/compitum/router.py:25).

## Mathematical Guarantees and Approximations

- Guarantees by construction
  - Positive definiteness: metric M = L L^T + δI; defensive Cholesky (src/compitum/metric.py:23, src/compitum/metric.py:39).
  - Feasibility-first selection: constraints enforced before argmax utility (src/compitum/constraints.py:36).
  - Bounded updates: trust-region controller caps effective step sizes (src/compitum/control.py:15).
- Documented heuristics
  - Shadow prices: approximate local diagnostics via finite-difference viability; report-only, not used for selection (src/compitum/constraints.py:36).
  - Coherence: KDE log-density prior in whitened space; influence bounded by clipping and small β_s (src/compitum/coherence.py:41).
  - Batch updates: per-sample adaptation in batch_route is order-dependent; acceptable throughput trade-off (src/compitum/router.py:147).

## Sensitivity and Robustness

- β_s (coherence weight): conclusions robust to modest sweeps; clipping bounds the influence.
- Metric rank: modest changes preserve headline results; rank controls bias–variance in metric learning.
- Update stride: affects throughput and adaptation cadence, not decision rule correctness.

## Calibration and Statistical Notes

- Bootstrap CIs: nonparametric bootstrap (B = 1000) over evaluation units with pairing preserved; report 95% percentile intervals.
- Calibration diagnostics: reliability curves (uncertainty buckets vs. |regret|), Spearman ρ(uncertainty, |regret|) (also in CEI). ECE‑style summaries can be added but are secondary for non‑probabilistic scales.
- KDE prior: Scott’s bandwidth in whitened features; clipping bounds influence; β_s sweeps reported for robustness.
- See {doc}`Statistical-Notes` for methodology details aimed at stat.ML reviewers.
- See {doc}`Learning-Perspective` for a cs.LG‑oriented framing (problem setup, decision rule, learning components, evaluation).

## Reproducibility Snapshot

Save environment with artifacts to aid replication:

```bat
python -m pip freeze > reports\env.txt
python - <<PY
import sys, platform
print(sys.version)
print(platform.platform())
PY
```

Include `env.txt` and system info with your report files.

## Reviewer FAQ

- Are shadow prices true dual variables? No. They are approximate local signals derived from finite-difference feasibility checks. They are reported for auditing and do not influence selection.
- What stabilizes updates without a Lyapunov proof? A Lyapunov-inspired trust-region controller caps effective step sizes using EMA and integral signals.
- Can the KDE prior dominate decisions? No. Its influence is bounded by clipping and a small β_s; conclusions are robust to modest β_s sweeps.
- Is batch metric update unbiased? The per-sample update in batch routing is order-dependent; it is a throughput trade-off that does not affect the feasibility-first decision rule.

## Data Lock (Optional, Security-Oriented)

To satisfy security-minded reviewers, you may lock and verify input data/configs by hashing them before evaluation and verifying after:

```bat
python tools\security\data_lock.py --write reports\data_manifest.json --paths data\rb_clean configs --exts .csv .json .yaml .yml
:: After evaluation
python tools\security\data_lock.py --verify reports\data_manifest.json --paths data\rb_clean configs --exts .csv .json .yaml .yml
```

We recommend sharing `reports/data_manifest.json` alongside evaluation outputs.

## Control-of-Error Index (CEI)

We summarize instantaneous, judge‑free feedback quality with a CEI composed of:

- Deferral quality (boundary vs. high‑regret): AP/AUROC
- Calibration (uncertainty vs. |regret|): Spearman ρ
- Stability (trust‑region shrink vs. future regret decrease): Spearman ρ
- Compliance: feasible rate

Helper script:

```bat
python tools\analysis\cei_report.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\cei_report.json ^
  --out-md reports\cei_report.md
```

Include the CEI report alongside fixed‑WTP summaries.

## cs.LG Fixed‑λ Summary (Paired Bootstrap)

Helper script for regret/win/compliance with 95% CIs per λ slice:

```bat
python tools\analysis\lg_summary.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --lambdas 0.1 1.0 ^
  --bootstrap 1000 ^
  --out-json reports\lg_summary.json ^
  --out-md reports\lg_summary.md
```

Attach `reports/lg_summary.md` to summarize per‑slice results for cs.LG reviewers.

## cs.CL Summary (Per-Task and Routing Mix)

Helper script for per-task win/boundary rates and routing distribution:

```bat
python tools\analysis\cl_summary.py ^
  --input data\rb_clean\eval_results\<latest-compitum-csv>.csv ^
  --out-json reports\cl_summary.json ^
  --out-md reports\cl_summary.md
```

Attach `reports/cl_summary.md` to summarize per-task behavior and selection distribution for cs.CL reviewers.

## Trust From Regret

- We frame trust as calibrated expectation of low future regret under bounded updates and instantaneous feedback.
- Report together:
  - Fixed‑WTP regret/win‑rate with paired bootstrap CIs.
  - CEI components: deferral quality (AP/AUROC), calibration (Spearman ρ; reliability curve), stability (ρ(shrink, future improvement)), compliance (~100%).
  - Control KPIs: trust‑radius shrink/expand/steady counts; r statistics; shrink→improve correlation.
- Helper scripts: see {doc}`Control-of-Error`, {doc}`Statistical-Notes`, and {doc}`Control-Perspective`.
