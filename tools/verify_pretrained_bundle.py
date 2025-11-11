from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class BundleMeta:
    version: str
    compitum_version: Optional[str]
    embedding_model: Optional[str]
    files: Dict[str, str]  # relpath -> sha256


def sha256sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_meta(meta_path: Path) -> Optional[BundleMeta]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    files = meta.get("files") or {}
    return BundleMeta(
        version=str(meta.get("version", "unknown")),
        compitum_version=str(meta.get("compitum_version")) if meta.get("compitum_version") else None,
        embedding_model=str(meta.get("embedding_model")) if meta.get("embedding_model") else None,
        files={str(k): str(v) for k, v in files.items()},
    )


def extract_bundle(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def verify_files(root: Path, files_to_hash: Dict[str, str]) -> bool:
    ok = True
    for rel, expected in files_to_hash.items():
        p = root / rel
        if not p.exists():
            print(f"[bundle][MISS] {rel} missing under {root}")
            ok = False
            continue
        got = sha256sum(p)
        if expected and expected.lower() != got.lower():
            print(f"[bundle][HASH][FAIL] {rel}: got {got}, expected {expected}")
            ok = False
        else:
            print(f"[bundle][OK] {rel} ({got[:8]}...) ")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and optionally extract pretrained bundle")
    ap.add_argument("--bundle", type=str, default=None, help="Path to bundle archive (.zip/.tar.gz)")
    ap.add_argument("--out", type=str, default=str(Path("data") / "pretrain_predictors"), help="Extraction directory")
    ap.add_argument("--verify-only", action="store_true", help="Only verify existing directory; do not extract")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.bundle and not args.verify_only:
        archive = Path(args.bundle)
        if not archive.exists():
            print(f"Bundle not found: {archive}", file=sys.stderr)
            return 2
        # Extract to a temp, then move into place atomically
        tmp_dir = out_dir.with_suffix(".part")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        extract_bundle(archive, tmp_dir)
        tmp_dir.replace(out_dir)

    # Verify
    meta = load_meta(out_dir / "metadata.json")
    if not meta:
        # Best-effort presence check for default predictors path used by tools/evaluate_compitum.py
        default_pred = out_dir / "predictors_all-MiniLM-L12-v2_0.1.joblib"
        if default_pred.exists():
            print("[bundle][WARN] No metadata.json; found predictors file. Proceeding.")
            return 0
        print("[bundle][FAIL] No metadata.json and predictors file not found.", file=sys.stderr)
        return 2

    ok = verify_files(out_dir, meta.files)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

