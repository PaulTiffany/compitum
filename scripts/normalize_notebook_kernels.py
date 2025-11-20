#!/usr/bin/env python3
"""
Normalize kernelspec metadata in notebooks to a portable default:
  name: python3, display_name: Python 3, language: python

Usage:
  python scripts/normalize_notebook_kernels.py [paths...]

If no paths are given, processes:
  - notebooks/**/*.ipynb
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable


def iter_ipynb(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    if paths:
        for p in paths:
            pp = pathlib.Path(p)
            if pp.is_dir():
                yield from pp.rglob("*.ipynb")
            elif pp.suffix == ".ipynb" and pp.exists():
                yield pp
    else:
        root = pathlib.Path("notebooks")
        yield from root.rglob("*.ipynb")


def normalize_one(path: pathlib.Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    md = data.setdefault("metadata", {})
    ks = md.setdefault("kernelspec", {})
    changed = False
    if ks.get("name") != "python3":
        ks["name"] = "python3"; changed = True
    if ks.get("display_name") != "Python 3":
        ks["display_name"] = "Python 3"; changed = True
    if ks.get("language") != "python":
        ks["language"] = "python"; changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)
    changed = 0
    total = 0
    for p in iter_ipynb(args.paths):
        total += 1
        if normalize_one(p):
            changed += 1
            print(f"updated: {p}")
    print(f"Checked {total} notebooks; updated {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

