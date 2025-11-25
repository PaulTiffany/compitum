import json
import subprocess
import sys


def run_py(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable] + args, capture_output=True, text=True, check=False)


def test_examples_run_list_json() -> None:
    cp = run_py(["scripts/examples_run.py", "list", "--json"])
    assert cp.returncode == 0
    data = json.loads(cp.stdout)
    assert isinstance(data, list) and len(data) >= 1
    names = {row["name"] for row in data}
    assert "synth_bench" in names


def test_examples_run_dry_quick() -> None:
    cp = run_py(["scripts/examples_run.py", "run", "--subset", "quick", "--dry-run"])
    assert cp.returncode == 0
    # Expect OK lines for dry run
    assert "[OK]" in cp.stdout


import pytest

@pytest.mark.skipif(sys.platform == "win32", reason="Known OSError on Windows with subprocess.run and asyncio")
def test_examples_run_sets_pythonpath_for_quick_single() -> None:
    # Run a single quick example that imports compitum, with a minimal env
    env = {"HYPOTHESIS_PROFILE": "ci"}
    cp = subprocess.run(
        [sys.executable, "scripts/examples_run.py", "run", "--name", "synth_bench"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        shell=True, # Added to mitigate Windows subprocess issues
    )
    # Should succeed because examples_run injects PYTHONPATH=src
    assert cp.returncode == 0, cp.stderr
