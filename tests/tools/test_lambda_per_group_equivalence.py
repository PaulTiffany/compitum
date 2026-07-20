from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv_with_groups(path: Path, n: int = 60) -> None:
    rng = np.random.default_rng(0)
    band_gap = rng.uniform(0.0, 3.0, size=n)
    density = rng.uniform(4.0, 9.0, size=n)
    nsites = rng.integers(2, 20, size=n)
    fe = rng.normal(-1.0, 0.5, size=n)
    y_true = 2.0 - band_gap + 0.1 * density + 0.01 * nsites - 0.1 * np.abs(fe)
    # Simple binary groups
    group = np.where(np.arange(n) % 2 == 0, "A", "B")
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


def test_lambda_per_group_equivalence(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _make_csv_with_groups(csv)
    lam = 0.5
    # Global lambda
    out_global = tmp_path / "global.json"
    cp1 = subprocess.run(
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
            str(lam),
            "--topk-grid",
            "1,5,10",
            "--out-json",
            str(out_global),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp1.returncode == 0, cp1.stderr
    jg = json.loads(out_global.read_text())
    # Per-group lambda mapping with the same lambda
    mapping = {"A": lam, "B": lam}
    mp = tmp_path / "lam.json"
    mp.write_text(json.dumps(mapping), encoding="utf-8")
    out_group = tmp_path / "group.json"
    cp2 = subprocess.run(
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
            str(0.0),  # overridden by lambda-per-group
            "--lambda-per-group",
            str(mp),
            "--group-col",
            "group",
            "--topk-grid",
            "1,5,10",
            "--out-json",
            str(out_group),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp2.returncode == 0, cp2.stderr
    jgg = json.loads(out_group.read_text())
    assert abs(jg.get("AURC", 0.0) - jgg.get("AURC", 0.0)) < 1e-12
