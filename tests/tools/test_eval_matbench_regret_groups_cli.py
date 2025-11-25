from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 60) -> None:
    rng = np.random.default_rng(123)
    band_gap = rng.uniform(0.0, 3.0, size=n)
    density = rng.uniform(4.0, 9.0, size=n)
    nsites = rng.integers(2, 20, size=n)
    fe = rng.normal(-1.0, 0.5, size=n)
    y_true = 2.0 - band_gap + 0.1 * density + 0.01 * nsites - 0.1 * np.abs(fe)
    group = np.where(band_gap < 1.5, "A", "B")
    df = pd.DataFrame(
        {
            "band_gap": band_gap,
            "density": density,
            "nsites": nsites,
            "formation_energy_per_atom": fe,
            "y_true": y_true,
            "group": group,
        }
    )
    df.to_csv(path, index=False)


def test_eval_matbench_regret_groups(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out_csv = tmp_path / "regret.csv"
    out_json = tmp_path / "regret.json"
    out_group = tmp_path / "groups.csv"
    _make_csv(csv)
    cp = subprocess.run(
        [
            sys.executable,
            "tools/eval_matbench_regret.py",
            "--path",
            str(csv),
            "--objective-col",
            "y_true",
            "--mode",
            "max",
            "--use-srmf",
            "--lambda-weight",
            "0.0",
            "--topk-grid",
            "1,5,10",
            "--group-col",
            "group",
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--out-group-csv",
            str(out_group),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert out_group.exists()
    gdf = pd.read_csv(out_group)
    assert set(["group", "k", "regret_norm"]).issubset(gdf.columns)
    payload = json.loads(out_json.read_text())
    assert "groups" in payload and isinstance(payload["groups"], dict)

