from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def run(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Compitum evaluation across multiple eval_names and aggregate per-eval CSVs")
    ap.add_argument("--config", type=str, default="data/rb_clean/evaluate_routers.yaml")
    ap.add_argument("--evals", nargs="*", type=str, default=[
        "grade-school-math",
        "hellaswag",
        "mbpp",
        "mmlu-high-school-mathematics",
        "mmlu-logical-fallacies",
    ])
    ap.add_argument("--max-evals", type=int, default=0)
    ap.add_argument("--wtp-list", type=str, default="0.0001,0.001,0.01,0.1,1.0,10.0")
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rb_clean/eval_results/compitum_multitask_combined.csv")
    args = ap.parse_args()

    root = Path.cwd()
    py = root / ".venv-routerbench" / "Scripts" / "python.exe"
    out_dir = root / "data" / "rb_clean" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for ev in args.evals:
        cmd = [
            str(py),
            str(root / "tools" / "evaluate_compitum.py"),
            "--config", args.config,
            "--wtp-list", args.wtp_list,
            "--filter-eval", ev,
        ]
        if args.max_evals:
            cmd += ["--max-evals", str(args.max_evals)]
        rc = run(cmd, cwd=root, timeout=args.timeout or None)
        if rc != 0:
            continue
        # Find the latest compitum CSV labeled with this eval
        latest = sorted(out_dir.glob(f"eval_results-eval-{ev}-*-val_split.csv"))
        if latest:
            generated.append(latest[-1])

    # Aggregate
    all_rows: List[pd.DataFrame] = []
    for p in generated:
        try:
            df = pd.read_csv(p)
            all_rows.append(df)
        except Exception:
            continue
    if not all_rows:
        print("No per-eval CSVs found; nothing to aggregate.")
        return
    combined = pd.concat(all_rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote combined CSV: {out_path} (rows={len(combined)})")


if __name__ == "__main__":
    main()

