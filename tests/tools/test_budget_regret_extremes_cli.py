from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 12) -> None:
    rng = np.random.default_rng(0)
    y = rng.random(size=n)  # nonnegative utilities ensure selecting all is optimal
    c = rng.integers(1, 5, size=n)  # positive integer costs
    df = pd.DataFrame(
        {
            "band_gap": rng.uniform(0, 3, size=n),
            "density": rng.uniform(4, 9, size=n),
            "nsites": c,  # reuse as costs for simplicity
            "formation_energy_per_atom": rng.normal(-1.0, 0.5, size=n),
            "y_true": y,
        }
    )
    df.to_csv(path, index=False)


def test_budget_regret_extremes(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    out_budget = tmp_path / "budget.csv"
    _make_csv(csv)
    total_cost = pd.read_csv(csv)["nsites"].sum()
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
            "1,3,5",
            "--cost-col",
            "nsites",
            "--budget-grid",
            f"0,{total_cost}",
            "--out-budget-csv",
            str(out_budget),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    dfb = pd.read_csv(out_budget)
    # Budget=0 -> both oracle and model should be 0, hence regret 0
    row0 = dfb.iloc[(dfb["budget"] - 0).abs().argsort().iloc[0]]
    assert abs(row0["regret"]) < 1e-12
    # Budget >= total cost -> regret 0 (both can select all)
    rowT = dfb.iloc[(dfb["budget"] - total_cost).abs().argsort().iloc[0]]
    assert abs(rowT["regret"]) < 1e-9
