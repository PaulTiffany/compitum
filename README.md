# compitum

![CI](https://github.com/PaulTiffany/compitum/actions/workflows/ci.yml/badge.svg)
![Mutation (Sharded Dispatch)](https://github.com/PaulTiffany/compitum/actions/workflows/mutation_dispatch.yml/badge.svg)

Status quick links:
- CI runs: https://github.com/PaulTiffany/compitum/actions/workflows/ci.yml
- Mutation (nightly/manual): https://github.com/PaulTiffany/compitum/actions/workflows/mutation_dispatch.yml
- Full Validation (nightly/manual): https://github.com/PaulTiffany/compitum/actions/workflows/full.yml

A production-ready, geometrically-aware AI router with SPD metric learning, constraint-aware
selection (shadow prices), metric-aware KDE coherence, and Lyapunov-stable online updates.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick demo

```bash
compitum route --prompt "Prove the binomial identity using generating functions."
```

## Run tests

```bash
pytest
```

For local parity with CI Hypothesis settings, set the profile:

- PowerShell: `$env:HYPOTHESIS_PROFILE='ci'; pytest -q`
- Bash: `HYPOTHESIS_PROFILE=ci pytest -q`

See `configs/` and `examples/` for constraints and a synthetic benchmark.

## Community
- Philosophy: `PHILOSOPHY.md`
- Contributing: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security: `SECURITY.md`
- Support: `SUPPORT.md`

## Release Artifacts
- Consolidated report: `reports/report_release.html`
- Certificate schema (JSON): `docs/_extra/assets/certificate.schema.json`

## Examples

- Docs page: docs/Examples.md
- Folder on GitHub: https://github.com/PaulTiffany/compitum/tree/main/examples
- In-repo overview: examples/README.md

## Core Science 0.1.1

- Geometry: SPD bounds, triangle inequality, ray monotonicity, update descent.
- Stability: Lyapunov decay/saturation/recovery; ΔV proxy sequences; combined update boundedness.
- Coherence: monotone outward, ±v symmetry, inward score direction, mixture discrimination.
- Constraints: feasibility monotone; duals slack ≈ 0, boundary ≈ 0; monotone/scale sanity.
- Determinism: repeated/batch determinism; paraphrase flip budget + explainability.
- Pedagogy: practice raises evidence/utility (beta_s > 0); prepared environment fixes constraints.

Run invariants

```bash
pytest -q tests/invariants           # smoke
pytest -q -m lg                      # geometry/learning
pytest -q -m cl                      # explainability/determinism
pytest -q -m sy                      # control/stability
pytest -q -m stat                    # estimation/uncertainty
pytest -q -m pedagogy                # control of error
```

## RouterBench Data (5-shot pickle)

Some RouterBench‑based scripts expect a local copy of `routerbench_5shot.pkl` (not redistributed here).

- Download from: https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_5shot.pkl
- Or use the resolve URL in the fetch script (recommended):

```bat
python scripts\fetch_routerbench.py --also-copy-to-src
```

This places the file at `data/routerbench_5shot.pkl` and also copies it to
`src/routerbench/routerbench_5shot.pkl` for compatibility with existing defaults.
You can provide `--sha256 <HEX>` to verify integrity.

Security note: `.pkl` files can execute code when loaded; download only from trusted sources.

## Testing Strategy

The project maintains a rigorous, deterministic testing program.

*   **CI Profile (default):** `pytest` runs with `HYPOTHESIS_PROFILE=ci`. This uses a fixed random seed and a moderate number of examples (`max_examples=100`) for fast, repeatable builds.
*   **Mutation Profile:** For mutation testing with `cosmic-ray`, a dedicated `HYPOTHESIS_PROFILE=mutation` is used via a wrapper script. This allows for a different number of examples to balance thoroughness and speed.
*   **Invariants Suite:** A dedicated property-based test suite in `tests/invariants/` validates the core mathematical and operational invariants of the system. These tests are marked with `@pytest.mark.invariants`.

To run the full verification suite, including mutation testing:
```bat
set HYPOTHESIS_PROFILE=ci && ruff check . && pytest --cov=compitum --cov-branch && CALL .\.venv-routerbench\Scripts\activate.bat && python -m pytest --cov=compitum --cov-branch --cov-append src/routerbench && coverage report -m && del /q session.sqlite 2>nul && cosmic-ray init --force cosmic-ray.toml session.sqlite && cosmic-ray exec cosmic-ray.toml session.sqlite && cr-report session.sqlite
```
**Note:** The `mypy` check has been temporarily removed from this command due to a path resolution issue. See `plan3.txt` for details.

## Export Control

This project is open-source research code (MIT). Use is subject to U.S. export laws and sanctions compliance. Do not use if you are a sanctioned person/region.

## Benchmark Profiles

Three run profiles are provided for benchmarking. They are configured via environment variables.

*   **SMOKE:** A very fast run to ensure the benchmarks execute without error.
    ```bat
    set COMPITUM_SKIP_PERF_ASSERTIONS=1 && set COMPITUM_STEPS=400 && set COMPITUM_REFIT_POLICY=never && set COMPITUM_UPDATE_BATCH_SIZE=100000 && pytest -q -k "test_energy_drift or test_constraint_violation_rate or test_spd_det_and_trust_radius_bounds" --benchmark-min-time=0.01
    ```

*   **DEV:** A standard development run for quick performance checks.
    ```bat
    set COMPITUM_STEPS=800 && set COMPITUM_REFIT_POLICY=adaptive && set COMPITUM_UPDATE_BATCH_SIZE=4000 && pytest -q -k "test_energy_drift or test_constraint_violation_rate or test_spd_det_and_trust_radius_bounds or test_iso_utility_savings_vs_fixed_best or test_router_throughput_and_latency" --benchmark-autosave --benchmark-save=DEV
    ```

*   **FULL:** The complete diagnostic run, including the heavy benchmarks.
    ```bat
    set COMPITUM_STEPS=4000 && set COMPITUM_REFIT_POLICY=periodic && set COMPITUM_UPDATE_BATCH_SIZE=2000 && pytest -q -m heavy_bench --benchmark-autosave --benchmark-save=FULL
    ```

### A Note on Threading

For performance-critical benchmark runs, it is advisable to set the following environment variables to prevent thread over-subscription from libraries like NumPy, which can interfere with measurements:

```bat
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
```

## Running RouterBench Evaluation

To run the full `routerbench` evaluation, follow these steps:

### 1. Setup RouterBench Environment

Create a new virtual environment for `routerbench` and install the dependencies from `src/routerbench/requirements.txt`.

```bash
python -m venv .venv-routerbench
.venv-routerbench\Scripts\activate
pip install -r src/routerbench/requirements.txt
```

### 2. Pre-train Predictors

For fast evaluation, it's crucial to pre-train the `CalibratedPredictor` models to avoid lengthy fitting times during each run.

Run the `pretrain_predictors.py` script using the `routerbench` virtual environment:

```bash
set PYTHONPATH=C:\Users\paulc\projects\compitum\src && .\.venv-routerbench\Scripts\python.exe -m scripts.pretrain_predictors
```

This will save the pre-trained predictors to `data/pretrain_predictors/predictors_all-MiniLM-L12-v2_0.1.joblib`.

### 3. Run the RouterBench Evaluation

Execute the `evaluate_routers.py` script as a module within the `routerbench` package. This command will run the full evaluation and generate the results.

```bash
set PYTHONPATH=C:\Users\paulc\projects\compitum\src && .\.venv-routerbench\Scripts\python.exe -m routerbench.evaluate_routers --config data\routerbench\evaluate_routers.yaml --local --data-path routerbench_5shot.pkl
```

This command will generate CSV and PKL files in the `data/eval_results` directory, containing the evaluation metrics for various router models.

## Dev Tips

- Hypothesis CI parity locally:
  - PowerShell: `$env:HYPOTHESIS_PROFILE='ci'; pytest -q`
  - Bash: `HYPOTHESIS_PROFILE=ci pytest -q`

## Educator Pack

Generate a small, self-contained pack for workshops:

```bash
python scripts/generate_classroom_pack.py
```

This writes `artifacts/pedagogy_pack.zip` with the lab worksheet (`docs/Pedagogy-Lab.md`), a demo script, a sample certificate JSONL (if available), and a tiny prompt set.
