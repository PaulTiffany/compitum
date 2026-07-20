from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from generate_artifact_manifest import collect  # type: ignore


def gather_eval_dirs(root: Path) -> List[Path]:
    candidates = [
        root / "data" / "rb_clean" / "eval_results",
        root / "data" / "routerbench" / "eval_results",
    ]
    return [p for p in candidates if p.exists()]


def newest_first(files: Iterable[Path]) -> List[Path]:
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def archive_dir(src: Path, dst_root: Path, keep: int) -> List[Path]:
    src_files = [p for p in src.iterdir() if p.is_file()]
    kept = set(newest_first(src_files)[:keep])
    archived: List[Path] = []
    if not src_files:
        return archived
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_root / f"eval_results_{src.name}_{ts}"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src_files:
        if p in kept:
            continue
        target = dst / p.name
        shutil.move(str(p), str(target))
        archived.append(target)
    return archived


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive old eval result files to artifacts/legacy/")
    ap.add_argument(
        "--keep", type=int, default=4, help="Number of most recent files to keep per dir"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/legacy",
        help="Directory to move older files into (timestamped subdirs)",
    )
    args = ap.parse_args()

    root = Path.cwd()
    out_root = root / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_targets: List[Path] = []
    for d in gather_eval_dirs(root):
        archived = archive_dir(d, out_root, keep=args.keep)
        manifest_targets.extend(archived)

    # Write a manifest for archived files
    if manifest_targets:
        items = collect(manifest_targets)
        import json as _json

        manifest_path = out_root / "archived_manifest.json"
        manifest_path.write_text(_json.dumps(items, indent=2), encoding="utf-8")
        print(f"Archived {len(manifest_targets)} files. Manifest: {manifest_path}")
    else:
        print("No files archived. Nothing to do.")


if __name__ == "__main__":
    main()
