from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_audit_materials_manifold_cli_offline(tmp_path: Path) -> None:
    out = tmp_path / "audit.csv"
    cp = subprocess.run(
        [
            sys.executable,
            "tools/audit_materials_manifold.py",
            "--offline-mock",
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
    assert len(df) == 2
    assert set([
        "material_id",
        "formula",
        "srmf_phase",
        "curvature_kappa",
        "stability_leak",
        "prediction",
    ]).issubset(df.columns)

