from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(paths: List[Path], exts: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for root in paths:
        if root.is_file():
            if not exts or root.suffix.lower() in exts:
                out[str(root.as_posix())] = sha256_file(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if exts and p.suffix.lower() not in exts:
                continue
            out[str(p.as_posix())] = sha256_file(p)
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Data lock: hash inputs and verify against a manifest")
    ap.add_argument("--write", type=Path, help="Write manifest JSON here (hashes of inputs)")
    ap.add_argument("--verify", type=Path, help="Verify against an existing manifest JSON")
    ap.add_argument("--paths", nargs="*", default=["data/rb_clean", "configs"], help="Paths to include")
    ap.add_argument("--exts", nargs="*", default=[".csv", ".json", ".yaml", ".yml"], help="Extensions to include (empty=all)")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths]
    exts = [e.lower() for e in args.exts]
    hashes = collect(roots, exts)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps({"hashes": hashes}, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {args.write}")
    if args.verify:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        expected = manifest.get("hashes", {})
        diffs = []
        # Missing or changed files
        for k, v in expected.items():
            cur = hashes.get(k)
            if cur is None:
                diffs.append((k, "missing"))
            elif cur != v:
                diffs.append((k, "changed"))
        # New files
        for k in hashes.keys():
            if k not in expected:
                diffs.append((k, "new"))
        if diffs:
            print("Manifest verification FAILED:")
            for k, why in diffs:
                print(f" - {why}: {k}")
            raise SystemExit(2)
        print("Manifest verification OK.")


if __name__ == "__main__":
    main()

