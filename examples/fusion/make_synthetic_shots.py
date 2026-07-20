#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def write_shot(path: Path, steps: int, crash_at: int | None) -> None:
    t = np.arange(steps, dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    if crash_at is None:
        q = 1.5 - 0.001 * t  # stays > 1.0
    else:
        q = 1.5 - 0.01 * t  # drops < 1.0 around t�50 when steps=60
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic fusion shot CSVs")
    ap.add_argument("--out", type=Path, default=Path("data/samples/fusion_shots"))
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_shot(out / "shot_demo.csv", steps=args.steps, crash_at=50)
    write_shot(out / "shot_stable.csv", steps=args.steps, crash_at=None)
    print(f"Wrote sample shots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
