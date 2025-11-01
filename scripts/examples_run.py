#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class Example:
    name: str
    cmd: List[str]
    description: str
    subset: str = "quick"  # quick | all


def build_examples(repo_root: Path) -> List[Example]:
    py = sys.executable
    examples: List[Example] = [
        Example(
            name="demo_route",
            cmd=[py, "examples/demo_route.py", "--seed", "1"],
            description="One-line CLI route demo (prints JSON/certificate depending on flags).",
            subset="quick",
        ),
        Example(
            name="synth_bench",
            cmd=[py, "examples/synth_bench.py", "--quiet", "--seed", "0", "--D", "16", "--rank", "4", "--n", "50"],
            description="SPD metric sanity: two clusters, average distances (JSON).",
            subset="quick",
        ),
        Example(
            name="certificate_card",
            cmd=[py, "examples/certificate_card.py", "--prompt", "Sketch a proof of AM-GM.", "--seed", "2"],
            description="Render a short Markdown card summarizing a routing certificate.",
            subset="quick",
        ),
        Example(
            name="batch_route_demo",
            cmd=[py, "examples/batch_route_demo.py", "--n", "3", "--D", "35", "--seed", "7"],
            description="Batch-route tiny embeddings and print a compact JSON summary.",
            subset="quick",
        ),
        Example(
            name="pedagogy_control_of_error",
            cmd=[py, "examples/pedagogy_control_of_error.py"],
            description="Demonstrate practice improves evidence and prepared environment fixes constraints.",
            subset="all",
        ),
    ]

    # Optional: explain_certificate_file requires an input; include only if present
    jsonl = repo_root / "reports" / "certificates_demo.jsonl"
    if jsonl.exists():
        examples.append(
            Example(
                name="explain_certificate_file",
                cmd=[py, "examples/explain_certificate_file.py", "--input", str(jsonl)],
                description="Read saved certificate (JSON/JSONL) and print a Markdown card.",
                subset="quick",
            )
        )
    return examples


def run(cmd: Sequence[str]) -> int:
    env = os.environ.copy()
    env.setdefault("HYPOTHESIS_PROFILE", "ci")
    env.setdefault("PYTHONUNBUFFERED", "1") # Mitigate Windows subprocess issues
    # Ensure local package is importable without install
    try:
        repo_root = Path(__file__).resolve().parents[1]
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    except Exception:
        pass
    try:
        proc = subprocess.run(cmd, env=env, check=False)
        return int(proc.returncode or 0)
    except KeyboardInterrupt:
        return 130


def main() -> int:
    ap = argparse.ArgumentParser(description="List and run Compitum examples.")
    ap.add_argument("action", choices=["list", "run"], help="List examples or run them")
    ap.add_argument("--name", help="Name of example to run (omit to run all in subset)")
    ap.add_argument("--subset", choices=["quick", "all"], default="quick", help="Subset to run")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--json", action="store_true", help="Print JSON summary of results")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    exs = build_examples(root)

    if args.action == "list":
        rows = [
            {
                "name": e.name,
                "subset": e.subset,
                "description": e.description,
                "cmd": " ".join(e.cmd),
            }
            for e in exs
        ]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print("Available examples (subset:name: description)")
            for r in rows:
                print(f"- {r['subset']}:{r['name']}: {r['description']}")
        return 0

    # action == run
    targets = [e for e in exs if (args.name == e.name) or (args.name is None and (e.subset == args.subset or args.subset == "all"))]
    if not targets:
        print("No matching examples.")
        return 1

    results = []
    for e in targets:
        if args.dry_run:
            rc = 0
        else:
            rc = run(e.cmd)
        results.append({"name": e.name, "returncode": rc})
        status = "OK" if rc == 0 else f"ERR({rc})"
        if not args.json:
            print(f"[{status}] {' '.join(e.cmd)}")

    if args.json:
        print(json.dumps(results))
    # Return first failing code if any
    for r in results:
        if r["returncode"] != 0:
            return int(r["returncode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
