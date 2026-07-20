import json
import subprocess
import sys
import pytest


pytestmark = pytest.mark.examples


def run_py(args: list[str]) -> str:
    proc = subprocess.run([sys.executable] + args, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def test_synth_bench_quiet() -> None:
    out = run_py(
        [
            "examples/synth_bench.py",
            "--quiet",
            "--seed",
            "0",
            "--D",
            "16",
            "--rank",
            "4",
            "--n",
            "50",
        ]
    )  # quick
    data = json.loads(out)
    assert "avg_d_math" in data and "avg_d_code" in data


def test_certificate_card() -> None:
    out = run_py(["examples/certificate_card.py", "--prompt", "AM-GM", "--seed", "1"])
    assert "Certificate Card" in out
    assert "Model:" in out and "Utility:" in out
