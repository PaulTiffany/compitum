from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(42)
    band_gap = rng.uniform(0.0, 3.0, size=n)
    density = rng.uniform(4.0, 9.0, size=n)
    nsites = rng.integers(2, 20, size=n)
    fe = rng.normal(-1.0, 0.5, size=n)
    y_true = 2.0 - band_gap + 0.1 * density + 0.01 * nsites - 0.1 * np.abs(fe)
    df = pd.DataFrame(
        {
            "band_gap": band_gap,
            "density": density,
            "nsites": nsites,
            "formation_energy_per_atom": fe,
            "y_true": y_true,
            "mid": [f"mp-{i}" for i in range(n)],
            "formula": [f"LaNiO{i}" for i in range(n)],
        }
    )
    df.to_csv(path, index=False)


def test_calibrate_matbench_srmf_cli(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out_json = tmp_path / "calib.json"
    scores_out = tmp_path / "scores.csv"
    _make_csv(csv)
    cp = subprocess.run(
        [
            sys.executable,
            "tools/calibrate_matbench_srmf.py",
            "--path",
            str(csv),
            "--objective-col",
            "y_true",
            "--mode",
            "max",
            "--lambda-grid",
            "0.0,0.5,1.0",
            "--topk-grid",
            "1,5,10",
            "--bootstrap",
            "10",
            "--seed",
            "0",
            "--out-json",
            str(out_json),
            "--scores-out",
            str(scores_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(out_json.read_text())
    assert "best_lambda" in payload and "val" in payload and "test" in payload
    assert scores_out.exists()

