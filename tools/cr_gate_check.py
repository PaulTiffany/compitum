from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print a Cosmic Ray shard's mutation score and optionally gate on a threshold"
    )
    ap.add_argument("summary_json", type=Path, help="Path to cr_quick_summary_<group>.json")
    ap.add_argument("--group", required=True, help="Shard group name, for logging")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument(
        "--gate", action="store_true", help="Fail (exit 1) when score is below threshold"
    )
    args = ap.parse_args()

    try:
        data = json.loads(args.summary_json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"Cosmic Ray (quick shard) {args.group}: summary file missing at {args.summary_json}",
            file=sys.stderr,
        )
        return 0 if not args.gate else 1

    score = float(data.get("mutation_score", 0))
    print(
        f"Cosmic Ray (quick shard) {args.group} score: {score} threshold: {args.threshold} gate: {args.gate}"
    )
    if args.gate and score < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
