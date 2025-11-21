#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_file(path: Path, n: int, dim: int, p_sc: float, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    mu = np.ones(dim) * 0.8
    X_sc = rng.normal(loc=mu, scale=0.4, size=(int(n * p_sc), dim))
    X_ns = rng.normal(loc=-mu, scale=0.6, size=(n - int(n * p_sc), dim))
    X = np.vstack([X_sc, X_ns])
    y = np.array([1] * len(X_sc) + [0] * len(X_ns))
    rng.shuffle(y)
    df = pd.DataFrame({f"x{i+1}": X[:, i] for i in range(dim)})
    df["label_sc"] = y
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic SC dataset")
    ap.add_argument("--out", type=Path, default=Path("data/samples/supercon"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dim", type=int, default=8)
    args = ap.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    make_file(out / "sc_train.csv", n=args.n, dim=args.dim, p_sc=0.5, seed=1)
    make_file(out / "sc_test.csv", n=args.n, dim=args.dim, p_sc=0.5, seed=2)
    print(f"Wrote synthetic SC files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

