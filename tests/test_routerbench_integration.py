import os
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
    script_path = project_root / "scripts" / "run_routerbench.bat"

    # Use the maintained local config under data/routerbench.
    fast_config_path = "data/routerbench/evaluate_routers.yaml"

    command = [str(script_path), f"--config={fast_config_path}", "--local"]

    env = dict(**os.environ)
    # Keep runs bounded/deterministic for CI unless the user overrides
    env.setdefault("RB_MAX_EVALS", "200")
    process = subprocess.run(command, capture_output=True, text=True, env=env)

    print(process.stdout)
    print(process.stderr)

    assert process.returncode == 0, "RouterBench script failed to run."
    assert "Saved to:" in process.stdout, "Expected results to be saved by RouterBench run."
