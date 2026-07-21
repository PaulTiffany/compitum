from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(paths: List[Path], root: Path) -> List[Dict[str, str]]:
    out = []
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        out.append(
            {
                # Repo-relative, not root.cwd()-absolute -- an absolute path
                # bakes one machine's directory layout into a committed
                # artifact, meaningless (and identifying) on anyone else's
                # checkout.
                "path": p.relative_to(root).as_posix(),
                "bytes": str(p.stat().st_size),
                "sha256": sha256sum(p),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate simple artifact manifest with SHA256 digests"
    )
    ap.add_argument("--out", type=str, default="reports/artifact_manifest.json")
    args = ap.parse_args()

    root = Path.cwd()
    candidates: List[Path] = []
    # Common artifacts
    candidates += sorted((root / "reports").glob("*.html"))
    candidates += sorted((root / "reports").glob("*.md"))
    candidates += sorted((root / "reports").glob("*.json"))
    candidates += sorted((root / "data" / "rb_clean" / "eval_results").glob("*.csv"))
    candidates += sorted((root / "data" / "rb_clean" / "eval_results").glob("*.pkl"))

    items = collect(candidates, root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    Path(args.out).write_text(_json.dumps(items, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {args.out} ({len(items)} items)")


if __name__ == "__main__":
    main()
