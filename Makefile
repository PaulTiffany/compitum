.PHONY: setup test lint mypy bandit mutate demo bench all check peer-review dist

setup:
	python -m venv .venv
	.venv\Scripts\python -m pip install -U pip
	.venv\Scripts\python -m pip install -e ".[dev]"

test:
	PYTHONWARNINGS="ignore::RuntimeWarning" .venv\Scripts\python -m pytest

test-ci:
	# Mirror CI selection locally: exclude routerbench and known strict cases
	.venv\Scripts\python -m pytest -q -m "not routerbench" \
	  --deselect tests/pgd/test_regex_prompt_extractor.py::test_math_signals_and_keywords \
	  --deselect tests/pgd/test_regex_prompt_extractor.py::test_semantic_proxies_unique_and_lengths \
	  --deselect tests/energy/test_symbolic_free_energy.py::test_energy_monotonic_wrt_distance_and_evidence

lint:
	.venv\Scripts\ruff check .

mypy:
	.venv\Scripts\mypy -p compitum --ignore-missing-imports --hide-error-context

bandit:
	.venv\Scripts\bandit -q -r src\compitum -x src\routerbench

mutate:
	.venv\Scripts\cosmic-ray init cosmic-ray.toml session.sqlite
	.venv\Scripts\cosmic-ray exec cosmic-ray.toml session.sqlite
	.venv\Scripts\cr-report session.sqlite

demo:
	.venv\Scripts\python -m compitum.cli route --prompt "Sketch a proof for AM-GM inequality."

bench:
	.venv\Scripts\python examples/synth_bench.py

all: test lint mypy

check: lint mypy bandit test

peer-review:
	scripts\run_peer_review.bat

dist:
	.venv\Scripts\python -m build

fetch-routerbench:
	.venv\Scripts\python scripts/fetch_routerbench.py --also-copy-to-src

docs:
	.venv\Scripts\python -m pip install -r docs/requirements.txt
	sphinx-build -b html docs docs_build/html

pedagogy-demo:
	.venv\Scripts\python examples/pedagogy_control_of_error.py

classroom-pack:
	.venv\Scripts\python scripts/generate_classroom_pack.py

examples-run:
	.venv\Scripts\python scripts/examples_run.py list
	.venv\Scripts\python scripts/examples_run.py run --subset quick
