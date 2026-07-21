"""
Generate a machine-readable attestation of this release run.

The attestation consolidates artifact manifest info, mutation summary,
coverage summary, benchmark/dataset provenance, test results, and
environment/commit details into a single JSON suitable for archival and
verification.

This script is pure-stdlib (plus optional inputs it reads as JSON) and safe
to run locally. It does not fetch network resources.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "compitum.release-attestation/v2"


def _read_json(path: Optional[Path]) -> Optional[Any]:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        # Outside the repo root -- keep as given rather than fabricate a
        # relative path that doesn't actually resolve to the same file.
        return path.as_posix()


def _file_info(path: Path, root: Path) -> Optional[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return {
        "path": _repo_relative(path, root),
        "size": len(data),
        "sha256": sha256(data).hexdigest(),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _latest_eval_csv(eval_dir: Path) -> Optional[Path]:
    try:
        files = sorted(eval_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except Exception:
        return None


def _git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_commit_short() -> Optional[str]:
    try:
        from compitum.security import git_commit_short as _g  # type: ignore

        return _g()
    except Exception:
        return _git(["rev-parse", "--short", "HEAD"])


def _dirty_tree() -> Optional[bool]:
    # Untracked files don't make the *tested source* dirty by themselves --
    # only modifications/staged changes to already-tracked files do, since
    # those are what could silently diverge the tested state from HEAD.
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    if status is None:
        return None
    return len(status) > 0


def _compitum_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("compitum")
    except Exception:
        return None


@dataclass
class EnvironmentInfo:
    python: str
    platform: str
    # Deliberately no `executable`/`cwd` here -- those are absolute,
    # machine-specific paths that leak local layout into a committed
    # artifact without adding any real reproducibility information beyond
    # what `python`/`platform` and the dependency versions already capture.
    key_dependency_versions: Dict[str, str]


def _key_dependency_versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pydantic",
        "pyyaml",
        "mutmut",
        "cosmic-ray",
        "pytest",
        "hypothesis",
    ]
    out: Dict[str, str] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            continue
    return out


def _env_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        key_dependency_versions=_key_dependency_versions(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate release attestation JSON")
    ap.add_argument("--manifest", type=Path, default=Path("reports/artifact_manifest.json"))
    ap.add_argument("--mutation", type=Path, default=Path("reports/mutation_summary.json"))
    ap.add_argument("--coverage", type=Path, default=Path("reports/coverage.json"))
    ap.add_argument("--snapshot", type=Path, default=Path("docs/repo_snapshot.jsonl"))
    ap.add_argument("--cr-db", type=Path, default=Path("cr_session.sqlite"))
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/rb_clean/eval_results"),
        help="Directory where eval CSVs are stored",
    )
    ap.add_argument(
        "--test-results", type=Path, default=None, help="JSON with passed/failed/skipped counts"
    )
    ap.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        default=[],
        help="Benchmark result JSON to hash and reference (repeatable)",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset to hash and reference, e.g. routerbench=routerbench_5shot.pkl (repeatable)",
    )
    ap.add_argument(
        "--notebook-validation", type=Path, default=None, help="JSON notebook-execution summary"
    )
    ap.add_argument("--sphinx-build", type=Path, default=None, help="JSON Sphinx HTML build result")
    ap.add_argument(
        "--sphinx-linkcheck", type=Path, default=None, help="JSON Sphinx linkcheck result"
    )
    ap.add_argument(
        "--package-build", type=Path, default=None, help="JSON package build/install-check result"
    )
    ap.add_argument(
        "--publication-commit",
        type=str,
        default=None,
        help="Commit B SHA, once known (unknown at Commit A time)",
    )
    ap.add_argument("--out", type=Path, default=Path("reports/release_attestation.json"))
    args = ap.parse_args()

    root = Path.cwd()
    now = datetime.now(tz=timezone.utc).isoformat()

    att: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now,
        "compitum_version": _compitum_version(),
        "tested_commit": _git_commit_short(),
        "publication_commit": args.publication_commit,
        "dirty_tree": _dirty_tree(),
        "environment": asdict(_env_info()),
    }

    # Inputs already produced by other tools
    att["artifact_manifest"] = _read_json(args.manifest)
    att["mutation_summary"] = _read_json(args.mutation)
    att["coverage_json"] = _read_json(args.coverage)
    att["test_results"] = _read_json(args.test_results)
    att["notebook_validation"] = _read_json(args.notebook_validation)
    att["sphinx_build"] = _read_json(args.sphinx_build)
    att["sphinx_linkcheck"] = _read_json(args.sphinx_linkcheck)
    att["package_build"] = _read_json(args.package_build)

    # Key artifact hashes (repo-relative paths only)
    key_files = {
        "snapshot": args.snapshot,
        "cr_db": args.cr_db,
        "manifest": args.manifest,
        "mutation": args.mutation,
        "coverage": args.coverage,
    }
    att["artifacts"] = {
        name: _file_info(path, root) for name, path in key_files.items() if Path(path).exists()
    }

    att["benchmarks"] = {
        path.stem: _file_info(path, root) for path in args.benchmark if path.exists()
    }

    dataset_hashes: Dict[str, Any] = {}
    for spec in args.dataset:
        if "=" not in spec:
            continue
        name, _, raw_path = spec.partition("=")
        info = _file_info(Path(raw_path), root)
        if info:
            dataset_hashes[name] = info
    att["dataset_hashes"] = dataset_hashes

    # Latest eval CSV info
    latest_eval = _latest_eval_csv(args.eval_dir)
    if latest_eval:
        att["latest_eval_csv"] = _file_info(latest_eval, root)

    # Persist
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(att, indent=2), encoding="utf-8")
    print(f"Wrote attestation to: {args.out}")


if __name__ == "__main__":
    main()
