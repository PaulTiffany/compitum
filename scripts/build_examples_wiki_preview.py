from __future__ import annotations

import json
import pathlib
import re
from typing import Dict, Any, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "EXAMPLES_WIKI_CLEAN.md"
OUT = ROOT / "EXAMPLES_WIKI_RENDERED.md"
GEN = ROOT / "compitum.wiki" / "_generated"
MAP = ROOT / "notebooks" / "wiki_map.yaml"


def read_yaml(path: pathlib.Path) -> Any:
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def github_and_nbviewer_links(repo: str, branch: str, nb_rel: str) -> str:
    nb_rel = nb_rel.replace("\\", "/")
    nb_url = f"https://github.com/{repo}/blob/{branch}/{nb_rel}"
    nbviewer_url = f"https://nbviewer.org/github/{repo}/blob/{branch}/{nb_rel}"
    from urllib.parse import quote as urlquote
    binder_url = f"https://mybinder.org/v2/gh/{repo}/{branch}?labpath={urlquote(nb_rel)}"
    return " | ".join([
        f"[Open on GitHub]({nb_url})",
        f"[Open in nbviewer]({nbviewer_url})",
        f"[Launch in Binder]({binder_url})",
    ])


def rewrite_asset_paths(md_text: str, asset_dirname: str) -> str:
    pat = re.compile(r"(!\[[^\]]*\]\()(?P<p>[^)]+)(\))")
    def _repl(m: re.Match[str]) -> str:
        p = m.group("p")
        if p.startswith("http://") or p.startswith("https://"):
            return m.group(0)
        return f"{m.group(1)}{asset_dirname}/{p}{m.group(3)}"
    return pat.sub(_repl, md_text)


def load_generated(marker: str) -> str:
    d = GEN / marker
    if not d.exists():
        return f"<!-- missing generated for {marker} -->\n"
    md_files = list(d.glob("*.md"))
    if not md_files:
        return f"<!-- no md found for {marker} -->\n"
    return md_files[0].read_text(encoding="utf-8")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    pattern = re.compile(r"<!--\s*NOTEBOOK:(?P<marker>[^:]+):BEGIN\s*-->[\s\S]*?<!--\s*NOTEBOOK:(?P=marker):END\s*-->")

    mapping_list: List[Dict[str, Any]] = read_yaml(MAP) or []
    by_marker: Dict[str, Dict[str, Any]] = {e.get("marker"): e for e in mapping_list if isinstance(e, dict) and e.get("marker")}

    def repl(m: re.Match[str]) -> str:
        marker = m.group("marker")
        entry = by_marker.get(marker, {})
        nb_rel = entry.get("notebook")
        md_body = load_generated(marker)
        gen_dir = GEN / marker
        out_name = next((p.stem for p in gen_dir.glob("*.md")), marker)
        assets_dir = f"_generated/{marker}/{out_name}_files"
        md_body = rewrite_asset_paths(md_body, assets_dir)
        if entry.get("strip_title"):
            md_body = re.sub(r"^# .*\n+", "", md_body, count=1)
        if isinstance(nb_rel, str) and nb_rel:
            links = github_and_nbviewer_links("PaulTiffany/compitum", "main", nb_rel)
            md_body = links + "\n\n" + md_body
        if entry.get("collapse"):
            summary = entry.get("summary") or f"{out_name} (rendered)"
            md_body = f"<details><summary>{summary}</summary>\n\n" + md_body + "\n\n</details>"
        return f"<!-- NOTEBOOK:{marker}:BEGIN -->\n{md_body}\n<!-- NOTEBOOK:{marker}:END -->"

    rendered = pattern.sub(repl, text)
    OUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
