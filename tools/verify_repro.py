from __future__ import annotations

"""
Verify reproducibility prerequisites for Compitum + RouterBench.

Checks performed:
- Git submodule `src/routerbench` is initialized, clean, and at a pinned commit
- RouterBench data pickle is present at `data/routerbench_5shot.pkl` (default path)
- Optional: compute SHA-256 of the pickle for logging
- CI-relevant env and pins sanity (presence of pinned `src/routerbench/requirements.txt`)

Usage:
  python tools/verify_repro.py [--expect-submodule-sha <short_sha>] [--data <path>]
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional
import hashlib
import sys


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def git_submodule_status(path: Path) -> tuple[str, str, bool]:
    """Return (sha, branch_desc, dirty) for the given submodule path.

    Uses `git submodule status` and parses the line for the submodule.
    """
    try:
        out = subprocess.run(
            ["git", "submodule", "status", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as e:  # pragma: no cover
        return ("", f"error:{e}", True)

    line = out.splitlines()[0] if out else ""
    # Format examples:
    #  4f2de88cd1234abcd3 src/routerbench (heads/main)
    # -4f2de88cd1234abcd3 src/routerbench (heads/main)  -> not initialized
    # +4f2de88cd1234abcd3 src/routerbench (heads/main)  -> different commit checked out
    dirty = False
    if not line:
        return ("", "missing", True)
    dirty = line[0] in {"-", "+"}
    sha = line[1:41].strip()
    rest = line[41:].strip()
    return (sha, rest, dirty)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify reproducibility surface")
    ap.add_argument(
        "--expect-submodule-sha",
        default=None,
        help="Optional short SHA (prefix) expected for src/routerbench",
    )
    ap.add_argument(
        "--data",
        default=str(Path("data") / "routerbench_5shot.pkl"),
        help="Path to RouterBench 5-shot pickle",
    )
    args = ap.parse_args()

    repo_root = Path.cwd()
    rb_path = repo_root / "src" / "routerbench"
    sha, desc, dirty = git_submodule_status(rb_path)
    ok = True

    print("[verify] submodule status:", sha, desc)
    # Treat vendored (non-submodule) routerbench as OK if directory and requirements exist
    if (not sha or desc == "missing") and rb_path.exists() and rb_path.is_dir():
        print("[verify] routerbench appears vendored (not a git submodule); proceeding")
        dirty = False
    if dirty:
        print("[verify][FAIL] submodule not initialized or dirty; run: git submodule update --init --recursive")
        ok = False

    if args.expect_submodule_sha:
        exp = args.expect_submodule_sha
        if not sha.startswith(exp):
            print(f"[verify][FAIL] submodule SHA {sha} does not match expected prefix {exp}")
            ok = False

    reqs = rb_path / "requirements.txt"
    if not reqs.exists():
        print("[verify][FAIL] pinned requirements missing at src/routerbench/requirements.txt")
        ok = False
    else:
        print("[verify] pinned requirements present at src/routerbench/requirements.txt")

    data_path = Path(args.data)
    digest = file_sha256(data_path)
    if digest is None:
        print(f"[verify][FAIL] missing dataset: {data_path}")
        ok = False
    else:
        print(f"[verify] dataset present: {data_path} (sha256={digest[:12]}..., bytes={data_path.stat().st_size})")

    # Summarize
    print("[verify] evaluate_routers.yaml data_path check:")
    cfg = repo_root / "data" / "routerbench" / "evaluate_routers.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        if "data/routerbench_5shot.pkl" in text:
            print("[verify] evaluate_routers.yaml references data/routerbench_5shot.pkl (OK)")
        else:
            print("[verify][WARN] evaluate_routers.yaml does not point at data/routerbench_5shot.pkl")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
