from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, with_labels: bool = True) -> None:
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(
        {
            "band_gap": rng.uniform(0.0, 3.0, size=n),
            "density": rng.uniform(4.0, 9.0, size=n),
            "nsites": rng.integers(2, 20, size=n),
            "formation_energy_per_atom": rng.normal(-1.0, 0.5, size=n),
            "mid": [f"mp-{i}" for i in range(n)],
            "formula": [f"LaNiO{i}" for i in range(n)],
        }
    )
    if with_labels:
        # Make a synthetic label roughly correlated with kappa threshold
        df["label_candidate"] = (df["band_gap"] < 1.0).astype(int)
    df.to_csv(path, index=False)


def test_eval_matbench_srmf_cli_basic(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out = tmp_path / "out.csv"
    _make_csv(csv, with_labels=False)
    cp = subprocess.run(
        [
            sys.executable,
            "tools/eval_matbench_srmf.py",
            "--adapter",
            "csv",
            "--path",
            str(csv),
            "--id-col",
            "mid",
            "--formula-col",
            "formula",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists()
    df = pd.read_csv(out)
    assert set(["material_id", "formula", "prediction"]).issubset(df.columns)


def test_eval_matbench_srmf_cli_with_metrics(tmp_path: Path) -> None:
    csv = tmp_path / "data2.csv"
    out = tmp_path / "out2.csv"
    mjson = tmp_path / "metrics.json"
    _make_csv(csv, with_labels=True)
    cp = subprocess.run(
        [
            sys.executable,
            "tools/eval_matbench_srmf.py",
            "--path",
            str(csv),
            "--id-col",
            "mid",
            "--formula-col",
            "formula",
            "--label-col",
            "label_candidate",
            "--bootstrap",
            "10",
            "--seed",
            "0",
            "--out",
            str(out),
            "--metrics-out",
            str(mjson),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert out.exists() and mjson.exists()
    payload = json.loads(mjson.read_text())
    assert "metrics" in payload
    assert set(["precision", "recall", "accuracy"]).issubset(payload["metrics"].keys())
