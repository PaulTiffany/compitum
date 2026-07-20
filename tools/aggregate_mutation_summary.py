from __future__ import annotations

import glob
import json
import os
from pathlib import Path


def aggregate() -> dict:
    cr_jsons = glob.glob("**/cr_quick_summary_*.json", recursive=True)
    mutmut_txts = glob.glob("**/mutmut_results_*.txt", recursive=True)

    survivors = []
    for path in mutmut_txts:
        try:
            txt = open(path, "r", encoding="utf-8", errors="ignore").read().lower()
            if "survived" in txt:
                survivors.append(os.path.basename(path))
        except Exception:
            pass

    scores = {}
    for path in cr_jsons:
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            scores[os.path.basename(path)] = data.get("mutation_score", 0)
        except Exception:
            pass

    return {"survivors": survivors, "scores": scores}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Aggregate mutmut/Cosmic Ray shard artifacts into one summary JSON"
    )
    ap.add_argument("out_path", type=Path, nargs="?", default=Path("MUTATION_SUMMARY.json"))
    args = ap.parse_args()

    payload = aggregate()
    args.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote mutation summary to: {args.out_path}")


if __name__ == "__main__":
    main()
