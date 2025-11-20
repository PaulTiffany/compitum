#!/usr/bin/env python3
"""
Create simple notebooks from example Python scripts so the Examples wiki page
can embed many focused notebooks instead of linking repeatedly to one.

Rules:
- For each `examples/*.py` script (selected allowlist), create
  `notebooks/examples/<name>.ipynb` with:
  - A top markdown title cell
  - One code cell containing the script contents
- Update notebooks/wiki_map.yaml by appending entries that embed each new
  notebook into `compitum.wiki/Examples.md` with collapsed presentation.
  (Does not duplicate existing entries.)

No execution is performed; notebooks are rendered statically by the embed job.
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Dict, Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOKS_DIR = ROOT / "notebooks" / "examples"
WIKI_MAP = ROOT / "notebooks" / "wiki_map.yaml"
WIKI_PAGE = "compitum.wiki/Examples.md"

ALLOWLIST = [
    "demo_route.py",
    "certificate_card.py",
    "synth_bench.py",
    "pedagogy_control_of_error.py",
    "batch_route_demo.py",
    "explain_certificate_file.py",
    "bridge_demo.py",
]


def make_nb(title: str, code: str) -> Dict[str, Any]:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [code if code.endswith("\n") else code + "\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def read_yaml_list(path: pathlib.Path) -> List[Dict[str, Any]]:
    import yaml  # type: ignore

    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or []


def write_yaml_list(path: pathlib.Path, items: List[Dict[str, Any]]) -> None:
    import yaml  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(items, sort_keys=False), encoding="utf-8")


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    existing_map = read_yaml_list(WIKI_MAP)
    existing_keys = {(e.get("page"), e.get("marker")) for e in existing_map}
    new_entries: List[Dict[str, Any]] = []

    for fname in ALLOWLIST:
        src_path = EXAMPLES / fname
        if not src_path.exists():
            continue
        base = src_path.stem
        nb_path = NOTEBOOKS_DIR / f"{base}.ipynb"
        marker = f"examples_{base}"
        title = f"Example: {base}"
        code = src_path.read_text(encoding="utf-8")
        nb = make_nb(title, code)
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

        key = (WIKI_PAGE, marker)
        if key not in existing_keys:
            new_entries.append(
                {
                    "page": WIKI_PAGE,
                    "notebook": str(nb_path.relative_to(ROOT).as_posix()),
                    "marker": marker,
                    "heading": f"{title}",
                    "collapse": True,
                    "strip_title": True,
                    "summary": f"{title} (rendered)",
                }
            )

    if new_entries:
        write_yaml_list(WIKI_MAP, existing_map + new_entries)
        print(f"Added {len(new_entries)} example notebooks and map entries.")
    else:
        print("No new notebooks added (allowlist already present).")


if __name__ == "__main__":
    main()

