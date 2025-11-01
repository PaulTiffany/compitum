#!/usr/bin/env bash
set -euo pipefail

export HYPOTHESIS_PROFILE=mutation
export HYPOTHESIS_SEED=1

pytest -q

