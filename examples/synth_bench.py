from __future__ import annotations

import argparse
import json
import numpy as np

from compitum.metric import SymbolicManifoldMetric  # type: ignore[import]


def main() -> int:
    p = argparse.ArgumentParser(description="Synthetic SPD metric sanity check.")
    p.add_argument("--D", type=int, default=35, help="Embedding dimension")
    p.add_argument("--rank", type=int, default=8, help="Low-rank factor for SPD metric")
    p.add_argument("--n", type=int, default=500, help="Samples per cluster")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--quiet", action="store_true", help="Print only the JSON result")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    D = int(args.D)
    M = SymbolicManifoldMetric(D, min(args.rank, D))
    # Two clusters: math-like vs code-like
    math_center = rng.normal(0, 1, size=D)
    code_center = rng.normal(0, 1, size=D)
    code_center[:5] += 2.0
    X_math = rng.normal(0, 0.6, size=(args.n, D)) + math_center
    X_code = rng.normal(0, 0.6, size=(args.n, D)) + code_center
    dm = float(np.mean([M.distance(x, math_center)[0] for x in X_math]))
    dc = float(np.mean([M.distance(x, code_center)[0] for x in X_code]))
    result = {"avg_d_math": dm, "avg_d_code": dc, "D": D, "rank": int(min(args.rank, D))}
    if not args.quiet:
        print("Synthetic SPD sanity check (two clusters)")
        print(f"Seed={args.seed} D={D} rank={result['rank']} n={args.n}")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
