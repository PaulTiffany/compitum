from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 10) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "band_gap": rng.uniform(0.0, 3.0, size=n),
            "density": rng.uniform(4.0, 9.0, size=n),
            "nsites": rng.integers(2, 20, size=n),
            "formation_energy_per_atom": rng.normal(-1.0, 0.5, size=n),
            "y_true": rng.normal(0, 1, size=n),
        }
    )
    df.to_csv(path, index=False)


def test_generate_attestation(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out_json = tmp_path / "attest.json"
    _make_csv(csv)
    # Create dummy calibration and regret JSONs
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"best_lambda": 0.0}), encoding="utf-8")
    regret = tmp_path / "regret.json"
    regret.write_text(json.dumps({"AURC": 0.1}), encoding="utf-8")

    cp = subprocess.run(
        [
            sys.executable,
            "tools/generate_matbench_attestation.py",
            "--input-csv",
            str(csv),
            "--calibration-json",
            str(calib),
            "--regret-json",
            str(regret),
            "--out",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(out_json.read_text())
    assert "files" in payload and str(csv) in payload["files"]
    assert "calibration" in payload and "regret" in payload
