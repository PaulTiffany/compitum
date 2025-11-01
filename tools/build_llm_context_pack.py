#!/usr/bin/env python3
"""
Build an LLM-ready context snapshot of the repo using Paleae, including .paleaeignore.

Examples:
  python tools/build_llm_context_pack.py --root . --out compitum_context.jsonl --target-size 1MB
  python tools/build_llm_context_pack.py --format json --target-size 800KB

The script prioritizes core code and key docs and ensures `.paleaeignore` is included.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def parse_size(s: str) -> int:
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024 * 1024}
    for unit in ("MB", "KB", "B"):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * multipliers[unit])
    return int(s)


def run_paleae(paleae_path: Path, out: Path, fmt: str, includes: List[str]) -> None:
    if not paleae_path.exists():
        raise FileNotFoundError(f"paleae.py not found at: {paleae_path}. Provide --paleae-path.")
    cmd = [sys.executable, str(paleae_path), "-f", fmt, "-o", str(out)]
    for inc in includes:
        cmd += ["--include", inc]
    # Be conservative: rely on Paleae defaults for excludes; user .paleaeignore applies.
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def load_snapshot(path: Path) -> Tuple[str, Any]:
    if path.suffix.lower() == ".jsonl":
        return "jsonl", [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return "json", json.loads(path.read_text(encoding="utf-8"))


def save_snapshot(kind: str, data: Any, path: Path) -> None:
    if kind == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record_path(rec: Dict[str, Any]) -> str:
    # Paleae uses "path" for each record
    return rec.get("path", "")


def prioritize(paths: Iterable[str]) -> Dict[str, int]:
    tiers = [
        (0, [r"^src/compitum/"]),
        (1, [r"^src/compitum/cli.py$"]),
        (2, [r"^README.md$", r"^PHILOSOPHY.md$", r"^ACCESSIBILITY.md$", r"^SECURITY.md$", r"^CONTRIBUTING.md$", r"^SUPPORT.md$", r"^CHANGELOG.md$"]),
        (3, [r"^compitum/docs/(Getting-Started|CLI|Philosophy|Invariants|LLM-Usage|API-Reference)\.md$"]),
        (4, [r"^pyproject.toml$", r"^compitum/.paleaeignore$", r"^compitum/docs/conf.py$"]),
    ]
    import re

    compiled: List[Tuple[int, List[Any]]] = [(prio, [re.compile(p) for p in pats]) for prio, pats in tiers]
    prio_map: Dict[str, int] = {}
    for p in paths:
        prio = 10
        for tier_prio, regexes in compiled:
            if any(r.search(p) for r in regexes):
                prio = tier_prio
                break
        prio_map[p] = prio
    return prio_map


def trim_to_budget(kind: str, data: Any, target_bytes: int) -> Any:
    tmp_path = Path("")  # not used; we measure via dumps len
    # Order records by priority and then path length
    if kind == "jsonl":
        paths = [record_path(r) for r in data]
        prio = prioritize(paths)
        ordered = sorted(data, key=lambda r: (prio.get(record_path(r), 10), len(record_path(r))))
        # Greedy keep until size fits
        kept: List[Dict[str, Any]] = []
        for rec in ordered:
            trial = kept + [rec]
            size = sum(len(json.dumps(x, ensure_ascii=False)) + 1 for x in trial)
            if size <= target_bytes or not kept:
                kept = trial
            else:
                # skip low-priority
                continue
        return kept
    else:  # json
        files = data.get("files") or data
        if isinstance(files, list):
            paths = [record_path(r) for r in files]
            prio = prioritize(paths)
            ordered = sorted(files, key=lambda r: (prio.get(record_path(r), 10), len(record_path(r))))
            kept: List[Dict[str, Any]] = []
            for rec in ordered:
                trial = {**data, "files": kept + [rec]}
                size = len(json.dumps(trial, ensure_ascii=False))
                if size <= target_bytes or not kept:
                    kept.append(rec)
                else:
                    continue
            return {**data, "files": kept}
        # Fallback: if structure not as expected, return original
        return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT), help="Repo root to snapshot")
    ap.add_argument("--out", default="compitum_context.jsonl", help="Output snapshot path")
    ap.add_argument("--format", choices=["json", "jsonl"], default="jsonl")
    ap.add_argument("--target-size", default="1MB", help="Size budget (e.g., 512KB, 1MB)")
    ap.add_argument("--paleae-path", default="paleae/paleae.py", help="Path to paleae.py (external repo)")
    args = ap.parse_args()

    target_bytes = parse_size(args.target_size)

    includes = [
        r"^src/compitum/",
        r"^README.md$",
        r"^PHILOSOPHY.md$",
        r"^SECURITY.md$",
        r"^CONTRIBUTING.md$",
        r"^SUPPORT.md$",
        r"^CHANGELOG.md$",
        r"^docs/(Getting-Started|CLI|Philosophy|Invariants|LLM-Usage|API-Reference)\.md$",
        r"^\.paleaeignore$",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_paleae(Path(args.paleae_path), out_path, args.format, includes)

    kind, data = load_snapshot(out_path)
    # Trim if needed
    if out_path.stat().st_size > target_bytes:
        trimmed = trim_to_budget(kind, data, target_bytes)
        save_snapshot(kind, trimmed, out_path)

    print(f"Wrote snapshot: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

