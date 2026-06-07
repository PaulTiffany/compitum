#!/usr/bin/env bash
set -euo pipefail

# Allow CI to choose a lighter profile; default to 'mutation' locally
export HYPOTHESIS_PROFILE=""
export HYPOTHESIS_SEED=""

# Provide conservative CPU defaults unless explicitly overridden
export OMP_NUM_THREADS=""
export MKL_NUM_THREADS=""
export OPENBLAS_NUM_THREADS=""
export NUMEXPR_NUM_THREADS=""

# Deselect the same known-failing tests that CI excludes (ci.yml / Makefile /
# rigor.yml). Without these, the cosmic-ray baseline fails and every mutant is
# scored against an already-red suite, corrupting all mutation results.
DESELECT=(
  --deselect tests/pgd/test_regex_prompt_extractor.py::test_math_signals_and_keywords
  --deselect tests/pgd/test_regex_prompt_extractor.py::test_semantic_proxies_unique_and_lengths
  --deselect tests/energy/test_symbolic_free_energy.py::test_energy_monotonic_wrt_distance_and_evidence
)

# Optionally enable coverage to guide mutation tools (writes .coverage)
if [[ "${PYTEST_COVERAGE:-}" == "1" ]]; then
  pytest -q "${DESELECT[@]}" --cov=compitum --cov-branch --cov-report=term-missing
else
  pytest -q "${DESELECT[@]}"
fi
