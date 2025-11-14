#!/usr/bin/env python3
"""
Scan wiki pages for fenced Python code blocks, generate minimal notebooks for each
block, replace the block with a NOTEBOOK marker pair, and update notebooks/wiki_map.yaml.

This prepares pages so the embed_notebooks_in_wiki.py script (locally or via CI)
can render the notebooks to Markdown and insert them into the wiki.

Usage:
  python scripts/migrate_wiki_code_to_notebooks.py [--dry-run] [--lang python] [--min-lines 3]

Behavior:
  - For each compitum.wiki/*.md page, find code fences ```python ... ```.
  - For each block, create notebooks/wiki_snippets/<page-stem>/<marker>.ipynb.
  - Replace the code fence with markers:
        <!-- NOTEBOOK:<marker>:BEGIN -->
        [filled later by embed step]
        <!-- NOTEBOOK:<marker>:END -->
  - Append mapping entries to notebooks/wiki_map.yaml.
  - Skips blocks already converted (if markers present) and respects existing map entries.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Dict, Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIKI_DIR = ROOT / "compitum.wiki"
MAP_PATH = ROOT / "notebooks" / "wiki_map.yaml"
SNIPPETS_DIR = ROOT / "notebooks" / "wiki_snippets"


def read_text(p: pathlib.Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback binary read
    data = p.read_bytes()
    return data.decode("utf-8", errors="ignore")


def write_text(p: pathlib.Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def load_map() -> List[Dict[str, Any]]:
    if MAP_PATH.exists():
        if yaml is None:
            raise SystemExit("pyyaml required to read wiki_map.yaml (pip install pyyaml)")
        data = yaml.safe_load(read_text(MAP_PATH))
        if not data:
            return []
        if not isinstance(data, list):
            raise SystemExit("wiki_map.yaml must be a list")
        return data
    return []


def _normalize_page(p: str) -> str:
    try:
        path = pathlib.Path(p)
        # If already relative, keep as is
        if not path.is_absolute():
            return path.as_posix()
        # Make repo-relative if possible
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return p.replace("\\", "/")


def save_map(entries: List[Dict[str, Any]]) -> None:
    if yaml is None:
        raise SystemExit("pyyaml required to write wiki_map.yaml (pip install pyyaml)")
    # Normalize any absolute page paths to repo-relative
    normed: List[Dict[str, Any]] = []
    seen = set()
    for e in entries:
        e2 = dict(e)
        e2["page"] = _normalize_page(str(e2.get("page", "")))
        key = (e2.get("page"), e2.get("marker"))
        if key in seen:
            continue
        seen.add(key)
        normed.append(e2)
    write_text(MAP_PATH, yaml.safe_dump(normed, sort_keys=False))


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def make_notebook(code: str, title: str) -> str:
    # Minimal nbformat v4 JSON (avoid nbformat dependency)
    import json
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [code if code.endswith("\n") else code + "\n"]},
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python (compitum)",
                "language": "python",
                "name": "compitum",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, indent=2)


def replace_blocks_with_markers(md: str, page_stem: str, min_lines: int, lang: str) -> tuple[str, List[tuple[str, str]]]:
    """Return new_md and list of (marker, code)."""
    # Regex for fenced code blocks ```lang ... ``` capturing content lazily
    pattern = re.compile(r"^```(?P<lang>[A-Za-z0-9_-]+)?\s*\n(?P<code>[\s\S]*?)\n```\s*$", re.MULTILINE)
    out_parts = []
    last = 0
    collected: List[tuple[str, str]] = []
    index = 0
    for m in pattern.finditer(md):
        out_parts.append(md[last:m.start()])
        last = m.end()
        block_lang = (m.group("lang") or "").lower()
        code = m.group("code")
        lines = [ln for ln in code.splitlines() if ln.strip() != ""]
        if block_lang == lang and len(lines) >= min_lines:
            index += 1
            marker = f"auto_{slugify(page_stem)}_{index}"
            collected.append((marker, code))
            out_parts.append(f"<!-- NOTEBOOK:{marker}:BEGIN -->\n[Notebook content will be embedded by CI]\n<!-- NOTEBOOK:{marker}:END -->\n")
        else:
            # keep original block untouched
            out_parts.append(m.group(0))
    out_parts.append(md[last:])
    return ("".join(out_parts), collected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--min-lines", type=int, default=3)
    args = ap.parse_args()

    pages = sorted(WIKI_DIR.glob("*.md"))
    if not pages:
        print("No wiki pages found under compitum.wiki/", file=sys.stderr)
        sys.exit(1)

    existing_map = load_map()
    # Build a quick lookup to avoid duplicates
    existing_keys = {(e.get("page"), e.get("marker")) for e in existing_map}
    new_entries: List[Dict[str, Any]] = []
    total_blocks = 0

    for page in pages:
        # Skip sidebar and home if preferred? Keep all for now.
        text = read_text(page)
        new_text, blocks = replace_blocks_with_markers(text, page.stem, args.min_lines, args.lang)
        if not blocks:
            continue
        if not args.dry_run:
            if new_text != text:
                write_text(page, new_text)

        # Write notebooks and map entries
        for marker, code in blocks:
            nb_rel_dir = pathlib.Path("notebooks") / "wiki_snippets" / page.stem
            nb_rel = nb_rel_dir / f"{marker}.ipynb"
            nb_abs = ROOT / nb_rel
            if not args.dry_run:
                nb_json = make_notebook(code, f"Snippet from {page.name}")
                write_text(nb_abs, nb_json)
            if (str(page.as_posix()), marker) not in existing_keys and (str(page.as_posix()), marker) not in {(e.get("page"), e.get("marker")) for e in new_entries}:
                new_entries.append({
                    "page": str(page.as_posix()),
                    "notebook": str(nb_rel.as_posix()),
                    "marker": marker,
                    "heading": None,
                })
            total_blocks += 1

    if new_entries:
        all_entries = existing_map + new_entries
        if not args.dry_run:
            save_map(all_entries)
        print(f"Prepared {len(new_entries)} new mappings across {len(pages)} pages (blocks found: {total_blocks}).")
    else:
        # Still rewrite the map to normalize any absolute paths
        if not args.dry_run and MAP_PATH.exists():
            save_map(existing_map)
        print("No new Python code blocks found to migrate.")


if __name__ == "__main__":
    main()
