#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as im
    except Exception:
        return None
    try:
        return im.version(name)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate attestation JSON for Matbench runs")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--calibration-json", type=Path, required=False)
    ap.add_argument("--regret-json", type=Path, required=False)
    ap.add_argument(
        "--extra", type=Path, nargs="*", default=None, help="Optional extra files to hash"
    )
    ap.add_argument("--out", type=Path, default=Path("reports/matbench_attestation.json"))
    args = ap.parse_args()

    files: List[Path] = [args.input_csv]
    if args.calibration_json:
        files.append(args.calibration_json)
    if args.regret_json:
        files.append(args.regret_json)
    if args.extra:
        files.extend(args.extra)

    file_hashes: Dict[str, str] = {}
    for p in files:
        if p and p.exists():
            file_hashes[str(p)] = _sha256(p)

    calib_payload: Dict[str, Any] = {}
    regret_payload: Dict[str, Any] = {}
    try:
        if args.calibration_json and args.calibration_json.exists():
            calib_payload = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        if args.regret_json and args.regret_json.exists():
            regret_payload = json.loads(args.regret_json.read_text(encoding="utf-8"))
    except Exception:
        pass

    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": {
            "compitum": _pkg_version("compitum"),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
        },
        "git_commit": _git_commit(),
        "files": file_hashes,
        "calibration": calib_payload,
        "regret": regret_payload,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote attestation: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
