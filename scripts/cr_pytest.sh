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

# Optionally enable coverage to guide mutation tools (writes .coverage)
if [[ "" == "1" ]]; then
  pytest -q --cov=compitum --cov-branch --cov-report=term-missing
else
  pytest -q
fi