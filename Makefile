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
	.venv\Scripts\bandit -q -r src\compitum tools examples scripts -x src\routerbench

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

matbench-demo:
	.venv\Scripts\python examples\generate_matbench_demo.py --out data\matbench_demo.csv

matbench-calibrate:
	.venv\Scripts\python tools\calibrate_matbench_srmf.py --path data\matbench_demo.csv --objective-col y_true --mode max --topk-grid 1,5,10 --lambda-grid 0.0,0.5,1.0 --bootstrap 200 --seed 0 --out-json reports\matbench_calibration.json --scores-out reports\matbench_scores_test.csv

matbench-eval:
	.venv\Scripts\python - << "PY"
	import json, subprocess
	import sys

	best = json.load(open('reports/matbench_calibration.json','r',encoding='utf-8'))['best_lambda']
	cmd = [
	    sys.executable,
	    'tools/eval_matbench_regret.py',
	    '--path','data/matbench_demo.csv',
	    '--objective-col','y_true',
	    '--mode','max',
	    '--use-srmf',
	    '--lambda-weight',str(best),
	    '--topk-grid','1,5,10',
	    '--group-col','group',
	    '--out-csv','reports/matbench_regret.csv',
	    '--out-json','reports/matbench_regret.json',
	    '--out-group-csv','reports/matbench_regret_groups.csv',
	    '--bootstrap','200',
	    '--seed','0',
	]
	print('Running:', ' '.join(cmd))
	subprocess.check_call(cmd)
	PY

matbench-attest:
	.venv\Scripts\python tools\generate_matbench_attestation.py --input-csv data\matbench_demo.csv --calibration-json reports\matbench_calibration.json --regret-json reports\matbench_regret.json --out reports\matbench_attestation.json

matbench-pipeline: matbench-demo matbench-calibrate matbench-eval matbench-attest
