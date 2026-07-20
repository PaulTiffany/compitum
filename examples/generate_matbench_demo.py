#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic Matbench-style CSV for demos")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/matbench_demo.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    band_gap = rng.uniform(0.0, 3.0, size=args.n)
    density = rng.uniform(4.0, 9.0, size=args.n)
    nsites = rng.integers(2, 20, size=args.n)
    fe = rng.normal(-1.0, 0.5, size=args.n)
    # Define an objective (higher better) that loosely correlates with small band_gap and high density
    y_true = 2.0 - band_gap + 0.1 * density + 0.01 * nsites - 0.1 * np.abs(fe)
    # A simple group label to exercise per-group regret
    group = np.where(band_gap < 1.0, "low_gap", "hi_gap")

    df = pd.DataFrame(
        {
            "band_gap": band_gap,
            "density": density,
            "nsites": nsites,
            "formation_energy_per_atom": fe,
            "y_true": y_true,
            "mid": [f"mp-{i}" for i in range(args.n)],
            "formula": [f"LaNiO{i}" for i in range(args.n)],
            "group": group,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote demo CSV: {args.out} ({args.n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
