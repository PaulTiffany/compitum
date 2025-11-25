from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_export_matbench_offline_mock(tmp_path: Path) -> None:
    out = tmp_path / "task.csv"
    cp = subprocess.run(
        [sys.executable, "tools/export_matbench_task_csv.py", "--offline-mock", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists()
    df = pd.read_csv(out)
    required = [
        "band_gap",
        "density",
        "nsites",
        "formation_energy_per_atom",
        "y_true",
    ]
    assert set(required).issubset(df.columns)

