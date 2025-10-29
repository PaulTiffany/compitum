# Examples Overview

This folder contains small, runnable scripts that demonstrate core Compitum concepts. Each example is self‑contained and can be run directly after installing the package in a virtual environment.

## Categories

- Quickstart
  - `examples/demo_route.py` — one‑line CLI demo; routes a prompt and prints a JSON summary or full certificate via CLI flags.
- Certificate Card (Markdown)
  - `examples/certificate_card.py` — routes a prompt in‑process and prints a short Markdown card summarizing the certificate (model, utility, top components, boundary, feasibility, trust‑radius).
- Learning / Geometry
  - `examples/synth_bench.py` — builds two synthetic clusters and reports average SPD distances; a sanity check for the metric.
- Pedagogy / Control of Error
  - `examples/pedagogy_control_of_error.py` — demonstrates “practice improves evidence (and utility if beta_s > 0)” and “prepared environment fixes constraint loops.”
- Bridge Demo (internal tooling)
  - `examples/bridge_demo.py` — placeholder showing how BridgeBlocks map code spans to external artifacts (e.g., LaTeX).
 - Batch Routing
  - `examples/batch_route_demo.py` — routes a tiny batch of embeddings and prints a compact JSON summary per sample.
 - Certificate Explainers
  - `examples/certificate_card.py` — in‑process routing and Markdown summary.
  - `examples/explain_certificate_file.py` — read a certificate JSON/JSONL and print a Markdown summary.

## How to Run

Assumes you installed dev extras:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- Quickstart (CLI)
  - Bash
    ```bash
    compitum route --prompt "Prove the harmonic series diverges." --trace
    ```
  - PowerShell
    ```powershell
    compitum route --prompt "Prove the harmonic series diverges." --trace
    ```

- Geometry sanity
  ```bash
  python examples/synth_bench.py
  # Expected keys: {"avg_d_math": <float>, "avg_d_code": <float>}
  ```

- Pedagogy demo
  ```bash
  python examples/pedagogy_control_of_error.py
  # Observe: before/after evidence (and utility if beta_s>0); JP infeasible vs US feasible
  ```

- Certificate card
  ```bash
  python examples/certificate_card.py --prompt "Sketch a proof of AM-GM."
  ```
  
- Batch routing
  ```bash
  python examples/batch_route_demo.py --n 3 --D 35
  ```
  
- Explain a saved certificate
  ```bash
  python examples/explain_certificate_file.py --input reports/certificates_demo.jsonl
  ```

## Tips

- Determinism: set `HYPOTHESIS_PROFILE=ci` when running tests to match CI behavior.
- Performance: set `OMP_NUM_THREADS=1` (and similarly for MKL/OPENBLAS/NUMEXPR) for consistent timings.
- Windows shells: prefer PowerShell and use backticks to escape JSON if needed.

## Troubleshooting

- If CLI is not found, ensure the virtual environment is activated and `compitum` is installed with the console script entry point (provided by `pyproject.toml`).
- If imports fail, confirm `pip install -e .[dev]` succeeded and your Python is 3.9+.
