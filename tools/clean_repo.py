from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def rm_path(p: Path, dry: bool) -> None:
    if not p.exists():
        return
    if dry:
        print(f"DRY: would remove {p}")
        return
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except Exception:
            pass


def glob_many(root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean build/test artifacts and optionally large eval outputs"
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be removed")
    ap.add_argument(
        "--artifacts", action="store_true", help="Remove caches, coverage, tox, .hypothesis, etc."
    )
    ap.add_argument(
        "--reports",
        action="store_true",
        help="Remove generated reports (*.html, *.json, *.md) under reports/",
    )
    ap.add_argument(
        "--eval-outputs", action="store_true", help="Remove eval outputs under data/*/eval_results/"
    )
    ap.add_argument(
        "--venvs", action="store_true", help="Remove local venvs like .venv-routerbench (DANGEROUS)"
    )
    ap.add_argument("--all", action="store_true", help="Do all of the above")
    args = ap.parse_args()

    root = Path.cwd()
    dry = args.dry_run

    if not any([args.artifacts, args.reports, args.eval_outputs, args.venvs, args.all]):
        ap.error("Select at least one of --artifacts, --reports, --eval-outputs, --venvs, or --all")

    if args.artifacts or args.all:
        # Common build/test caches
        for pat in [
            "**/__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".hypothesis",
            ".tox",
            ".nox",
            "htmlcov",
            "build",
            "dist",
            "*.egg-info",
            ".coverage",
            ".coverage.*",
        ]:
            for p in glob_many(root, [pat]):
                rm_path(p, dry)

    if args.reports or args.all:
        rep = root / "reports"
        for p in glob_many(rep, ["*.html", "*.json", "*.md"]):
            rm_path(p, dry)

    if args.eval_outputs or args.all:
        for sub in [
            root / "data" / "rb_clean" / "eval_results",
            root / "data" / "routerbench" / "eval_results",
        ]:
            if sub.exists():
                for p in sub.glob("*"):
                    rm_path(p, dry)

    if args.venvs or args.all:
        for p in [root / ".venv-routerbench", root / ".venv"]:
            rm_path(p, dry)

    print("Clean complete (" + ("dry-run" if dry else "executed") + ")")


if __name__ == "__main__":
    main()
