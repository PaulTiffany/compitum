#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export HYPOTHESIS_PROFILE=ci

if [[ "${1:-}" == "deep" ]]; then
  pytest -q -m "invariants and deep"
else
  pytest -q tests/invariants
fi

