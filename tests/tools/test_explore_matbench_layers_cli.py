from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(0)
    X1 = rng.uniform(0.0, 3.0, size=n)
    X2 = rng.uniform(4.0, 9.0, size=n)
    X3 = rng.integers(2, 20, size=n)
    X4 = rng.normal(-1.0, 0.5, size=n)
    y = 2.0 - X1 + 0.1 * X2 + 0.01 * X3 - 0.1 * np.abs(X4)
    df = pd.DataFrame(
        {
            "band_gap": X1,
            "density": X2,
            "nsites": X3,
            "formation_energy_per_atom": X4,
            "y_true": y,
        }
    )
    df.to_csv(path, index=False)


def test_explore_layers_cli(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out_csv = tmp_path / "layers.csv"
    out_json = tmp_path / "layers.json"
    _make_csv(csv)
    cp = subprocess.run(
        [
            sys.executable,
            "tools/explore_matbench_layers.py",
            "--path",
            str(csv),
            "--objective-col",
            "y_true",
            "--quantile-layer-on",
            "band_gap",
            "--quantiles",
            "0.0,0.5,1.0",
            "--topk-grid",
            "1,5,10",
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert out_csv.exists() and out_json.exists()
    payload = json.loads(out_json.read_text())
    assert "quantile_layers" in payload and isinstance(payload["quantile_layers"], dict)
