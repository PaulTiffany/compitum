import os
import sys
import subprocess
from pathlib import Path
import pytest


@pytest.mark.benchmark
@pytest.mark.routerbench
def test_routerbench_integration() -> None:
    """
    Runs the routerbench benchmark script and checks for a successful exit code.
    """
    project_root = Path(__file__).resolve().parents[1]
    # Choose a cross-platform entry: use .bat on Windows, Python wrapper elsewhere
    if os.name == "nt":
        command = [
            str(project_root / "scripts" / "run_routerbench.bat"),
            "--config=data/routerbench/evaluate_routers.yaml",
            "--local",
        ]
    else:
        command = [
            sys.executable,
            str(project_root / "tools" / "run_routerbench_clean.py"),
            "--config=data/rb_clean/evaluate_routers.yaml",
            "--local",
            "--tokenizer-backend=tiktoken",
        ]

    env = dict(**os.environ)
    # Keep runs bounded/deterministic for CI unless the user overrides
    env.setdefault("RB_MAX_EVALS", "200")
    process = subprocess.run(command, capture_output=True, text=True, env=env)

    print(process.stdout)
    print(process.stderr)

    assert process.returncode == 0, "RouterBench script failed to run."
    assert (
        "Saved to:" in process.stdout or "Report written to:" in process.stdout
    ), "Expected results to be saved by RouterBench run."

