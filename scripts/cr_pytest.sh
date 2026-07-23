#!/usr/bin/env bash
set -euo pipefail

# Allow the caller (Cosmic Ray's parent workflow) to choose a lighter profile;
# default to 'mutation' locally. Using ":=" instead of a plain assignment means
# a value already exported by the caller survives instead of being stomped.
: "${HYPOTHESIS_PROFILE:=mutation}"
: "${HYPOTHESIS_SEED:=}"
export HYPOTHESIS_PROFILE HYPOTHESIS_SEED

# Provide conservative CPU defaults unless explicitly overridden by the caller
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

# Coverage-scoped test selection: if the caller has already derived which
# test files actually exercise this shard's target module (see
# scripts/derive_test_scope.py), run only those instead of the full suite --
# every mutant otherwise pays a full-suite rerun, which is what made every
# cr-quick-shard job in .github/workflows/mutation.yml time out at its
# 60-minute cap regardless of the target file's size.
if [[ -n "${CR_TEST_SCOPE_FILE:-}" && -s "${CR_TEST_SCOPE_FILE}" ]]; then
  mapfile -t CR_SCOPED_TESTS < "${CR_TEST_SCOPE_FILE}"
  pytest -q "${CR_SCOPED_TESTS[@]}"
# Optionally enable coverage to guide mutation tools (writes .coverage)
elif [[ "${PYTEST_COVERAGE:-}" == "1" ]]; then
  pytest -q --cov=compitum --cov-branch --cov-report=term-missing
else
  pytest -q
fi
