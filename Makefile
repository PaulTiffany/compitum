.PHONY: setup test lint mypy mutate demo bench all

setup:
	python -m venv .venv
	.venv\Scripts\python -m pip install -U pip
	.venv\Scripts\python -m pip install -e ".[dev]"

test:
	PYTHONWARNINGS="ignore::RuntimeWarning" .venv\Scripts\python -m pytest

lint:
	.venv\Scripts\ruff check compitum

mypy:
	.venv\Scripts\mypy -p compitum --ignore-missing-imports

mutate:
	.venv\Scripts\cosmic-ray init cosmic-ray.toml session.sqlite
	.venv\Scripts\cosmic-ray exec cosmic-ray.toml session.sqlite
	.venv\Scripts\cr-report session.sqlite

demo:
	.venv\Scripts\python -m compitum.cli route --prompt "Sketch a proof for AM-GM inequality."

bench:
	.venv\Scripts\python examples/synth_bench.py

all: test lint mypy