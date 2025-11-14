#!/usr/bin/env python3
"""
Convert selected Jupyter notebooks to GitHub-friendly Markdown and
inject them into wiki pages between marker blocks.

Usage:
  python scripts/embed_notebooks_in_wiki.py [--map notebooks/wiki_map.yaml] [--check]

Behavior:
  - For each mapping entry, runs `jupyter nbconvert --to markdown` to a
    generated folder under `compitum.wiki/_generated/<marker_id>/`.
  - Rewrites the target wiki page to replace the content between markers:
      <!-- NOTEBOOK:<marker_id>:BEGIN -->
      ... (generated content) ...
      <!-- NOTEBOOK:<marker_id>:END -->
    If markers are not present, inserts a new section with markers.
  - Adds small links to open the original `.ipynb` on GitHub and in nbviewer.

Requirements:
  - jupyter nbconvert must be available on PATH (install via `pip install jupyter nbconvert`).
  - pyyaml (pip install pyyaml).

Mapping YAML format (notebooks/wiki_map.yaml):
  - page: compitum.wiki/Examples.md
    notebook: notebooks/Examples_Tour.ipynb
    marker: examples_tour
    heading: Examples Tour Notebook
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Dict, Any, List

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Lazy error at runtime

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_text_relaxed(path: pathlib.Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    data = path.read_bytes()
    return data.decode("utf-8", errors="ignore")


def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def ensure_dirs(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mapping(path: pathlib.Path) -> List[Dict[str, Any]]:
    if yaml is None:
        raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.\n"
                        f"Mapping file required at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise SystemExit("Mapping YAML must be a list of entries.")
    for i, entry in enumerate(data):
        if not all(k in entry for k in ("page", "notebook", "marker")):
            raise SystemExit(f"Entry #{i} missing required keys (page, notebook, marker): {entry}")
    return data


def github_and_nbviewer_links(repo: str, branch: str, nb_rel: str) -> str:
    # Build plain links (badges optional; plain links render reliably in wiki)
    nb_url = f"https://github.com/{repo}/blob/{branch}/{nb_rel.replace(os.sep, '/')}"
    nbviewer_url = f"https://nbviewer.org/github/{repo}/blob/{branch}/{nb_rel.replace(os.sep, '/')}"
    # Binder link (optional; may be slow)
    try:
        from urllib.parse import quote as urlquote  # type: ignore
    except Exception:
        urlquote = lambda x: x  # noqa: E731
    binder_url = (
        f"https://mybinder.org/v2/gh/{repo}/{branch}?labpath="
        + urlquote(nb_rel.replace(os.sep, '/'))
    )

    parts = [
        f"[Open on GitHub]({nb_url})",
        f" | [Open in nbviewer]({nbviewer_url})",
        f" | [Launch in Binder]({binder_url})",
    ]
    return "".join(parts)


def nbconvert_to_markdown(nb_path: pathlib.Path, out_dir: pathlib.Path, out_name: str) -> pathlib.Path:
    ensure_dirs(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "markdown",
        str(nb_path),
        "--output",
        out_name,
        "--output-dir",
        str(out_dir),
    ]
    run(cmd)
    md_path = out_dir / f"{out_name}.md"
    if not md_path.exists():
        raise SystemExit(f"nbconvert did not produce expected file: {md_path}")
    return md_path


def rewrite_asset_paths(md_text: str, asset_dirname: str) -> str:
    # nbconvert emits relative references like `out_name_files/...`
    # When we inline into a wiki page, keep them relative to the wiki root:
    # We place assets into `_generated/<marker>/<out_name>_files/...`
    # Replace any pattern like `![](out_name_files/...)` with `![](_generated/<marker>/<out_name>_files/...)`
    pattern = re.compile(r"(!\[[^\]]*\]\()(?P<p>[^)]+)(\))")

    def repl(m: re.Match[str]) -> str:
        p = m.group("p")
        if p.startswith("http://") or p.startswith("https://"):
            return m.group(0)
        new_p = f"{asset_dirname}/{p}".replace("\\", "/")
        return f"{m.group(1)}{new_p}{m.group(3)}"

    return pattern.sub(repl, md_text)


def upsert_marker_block(page_text: str, marker: str, block_text: str, heading: str | None) -> str:
    begin = f"<!-- NOTEBOOK:{marker}:BEGIN -->"
    end = f"<!-- NOTEBOOK:{marker}:END -->"
    block = f"{begin}\n{block_text}\n{end}\n"
    if begin in page_text and end in page_text:
        # Replace existing block
        return re.sub(
            re.compile(re.escape(begin) + r"[\s\S]*?" + re.escape(end), re.MULTILINE),
            block,
            page_text,
        )
    else:
        # Insert near end by default
        insertion = block
        if heading:
            insertion = f"\n\n## {heading}\n\n" + insertion
        return page_text.rstrip() + "\n\n" + insertion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "notebooks" / "wiki_map.yaml"))
    ap.add_argument("--repo", default="PaulTiffany/compitum")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--check", action="store_true", help="Do not write files; only check commands.")
    args = ap.parse_args()

    mapping_path = pathlib.Path(args.map)
    entries = load_mapping(mapping_path)

    for entry in entries:
        page_rel = entry["page"]
        nb_rel = entry["notebook"]
        marker = entry["marker"]
        heading = entry.get("heading")

        page_path = ROOT / page_rel
        nb_path = ROOT / nb_rel
        if not page_path.exists():
            raise SystemExit(f"Wiki page not found: {page_path}")
        if not nb_path.exists():
            raise SystemExit(f"Notebook not found: {nb_path}")

        gen_root = ROOT / "compitum.wiki" / "_generated" / marker
        out_name = pathlib.Path(nb_rel).stem
        md_path = gen_root / f"{out_name}.md"
        assets_dirname = f"_generated/{marker}/{out_name}_files"

        print(f"[embed] Converting {nb_rel} -> {md_path}")
        if not args.check:
            # Clean previous
            if gen_root.exists():
                shutil.rmtree(gen_root)
            ensure_dirs(gen_root)
            produced_md = nbconvert_to_markdown(nb_path, gen_root, out_name)
            md_text = produced_md.read_text(encoding="utf-8")
            md_text = rewrite_asset_paths(md_text, assets_dirname)

            # Prepend quick links
            links = github_and_nbviewer_links(args.repo, args.branch, nb_rel)
            md_text = links + "\n\n" + md_text

            # Update wiki page
            page_text = read_text_relaxed(page_path)
            new_text = upsert_marker_block(page_text, marker, md_text, heading)
            if new_text != page_text:
                page_path.write_text(new_text, encoding="utf-8")
                print(f"[embed] Updated page: {page_rel}")
            else:
                print(f"[embed] No change needed: {page_rel}")
        else:
            print(f"[check] Would convert to: {md_path}")

    print("Done.")


if __name__ == "__main__":
    main()
