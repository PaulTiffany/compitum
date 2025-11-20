from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def _write_sc_csv(path: Path, n: int = 50, dim: int = 6) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(n, dim))
    y = (rng.random(n) > 0.5).astype(int)
    df = pd.DataFrame({f"x{i+1}": X[:, i] for i in range(dim)})
    df["label_sc"] = y
    df.to_csv(path, index=False)


def test_eval_supercon_offline_cli(tmp_path: Path):
    data = tmp_path / "sc"; data.mkdir()
    _write_sc_csv(data / "a.csv")
    _write_sc_csv(data / "b.csv")
    out = tmp_path / "metrics.csv"
    proc = subprocess.run([
        "python", "tools/eval_supercon_offline.py", str(data), "--state-dim", "6", "--rank", "3", "--alarm", "0.5", "--out", str(out)
    ], capture_output=True, text=True)
    assert proc.returncode == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert set(["file","tp","fp","tn","fn","precision","recall","accuracy"]).issubset(df.columns)