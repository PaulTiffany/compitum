"""
Generate a machine-readable attestation of this release run.

The attestation consolidates artifact manifest info, mutation summary,
coverage summary, key artifact hashes, environment details, and repo commit.

This script is pure-stdlib and safe to run locally. It does not fetch network
resources. Outputs a single JSON file suitable for archival and verification.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_info(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return {
        "path": str(path),
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


def _git_commit_short() -> Optional[str]:
    try:
        # Import from project if available
        from compitum.security import git_commit_short as _g  # type: ignore

        return _g()
    except Exception:
        return None


@dataclass
class EnvironmentInfo:
    python: str
    platform: str
    executable: str
    cwd: str


def _env_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        executable=sys.executable,
        cwd=str(Path.cwd()),
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
    ap.add_argument("--out", type=Path, default=Path("reports/release_attestation.json"))
    args = ap.parse_args()

    now = datetime.now(tz=timezone.utc).isoformat()

    att: Dict[str, Any] = {
        "schema": "compitum.release-attestation/v1",
        "generated_at": now,
        "commit": _git_commit_short(),
        "environment": asdict(_env_info()),
    }

    # Inputs
    att["artifact_manifest"] = _read_json(args.manifest)
    att["mutation_summary"] = _read_json(args.mutation)
    att["coverage_json"] = _read_json(args.coverage)

    # Key artifact hashes
    key_files = {
        "snapshot": args.snapshot,
        "cr_db": args.cr_db,
        "manifest": args.manifest,
        "mutation": args.mutation,
        "coverage": args.coverage,
    }
    att["artifacts"] = {
        name: _file_info(path) for name, path in key_files.items() if Path(path).exists()
    }

    # Latest eval CSV info
    latest_eval = _latest_eval_csv(args.eval_dir)
    if latest_eval:
        att["latest_eval_csv"] = _file_info(latest_eval)

    # Persist
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(att, indent=2), encoding="utf-8")
    print(f"Wrote attestation to: {args.out}")


if __name__ == "__main__":
    main()

