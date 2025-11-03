#!/usr/bin/env bash
set -euo pipefail

# Allow CI to choose a lighter profile; default to 'mutation' locally
export HYPOTHESIS_PROFILE="${HYPOTHESIS_PROFILE:-mutation}"
export HYPOTHESIS_SEED="${HYPOTHESIS_SEED:-1}"

# Provide conservative CPU defaults unless explicitly overridden
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Optionally enable coverage to guide mutation tools (writes .coverage)
if [[ "${PYTEST_COVERAGE:-0}" == "1" ]]; then
  pytest -q --cov=compitum --cov-branch --cov-report=term-missing
else
  pytest -q
fi
