#!/usr/bin/env python
"""
Fetch the RouterBench 5-shot pickle and place it in expected locations.

Defaults:
- Download URL: Hugging Face dataset file (resolve URL)
- Primary output: data/routerbench_5shot.pkl
- Also copy to: src/routerbench/routerbench_5shot.pkl (to satisfy existing defaults)

Usage:
  python scripts/fetch_routerbench.py \
    --url https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_5shot.pkl \
    --sha256 <optional_hex_digest>

Notes:
- This file is third-party content. Verify license and checksum.
- Loading .pkl files can execute code. Only load from trusted sources.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    print("requests is required: pip install requests", file=sys.stderr)
    raise


DEFAULT_URL = (
    "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/"
    "routerbench_5shot.pkl"
)


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "compitum-ci/0.1"}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"Downloading {out_path.name}: {pct}% ({downloaded}/{total} bytes)", end="\r")
        if total:
            print()  # newline after progress
        tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch RouterBench 5-shot .pkl")
    ap.add_argument("--url", default=DEFAULT_URL, help="Download URL")
    ap.add_argument(
        "--out",
        default=str(Path("data") / "routerbench_5shot.pkl"),
        help="Primary output path",
    )
    ap.add_argument(
        "--sha256",
        default=None,
        help="Optional SHA256 checksum to verify (hex)",
    )
    ap.add_argument(
        "--also-copy-to-src",
        action="store_true",
        help="Also copy to src/routerbench/routerbench_5shot.pkl",
    )
    args = ap.parse_args()

    primary = Path(args.out)
    if primary.exists() and args.sha256:
        digest = sha256sum(primary)
        if digest.lower() == args.sha256.lower():
            print(f"Found existing file with matching SHA256 at {primary}")
        else:
            print("Existing file checksum mismatch; re-downloading…")
            primary.unlink()

    if not primary.exists():
        print(f"Downloading from {args.url} → {primary}")
        download(args.url, primary)

    if args.sha256:
        digest = sha256sum(primary)
        if digest.lower() != args.sha256.lower():
            print(
                f"Checksum mismatch for {primary}: got {digest}, expected {args.sha256}",
                file=sys.stderr,
            )
            return 2
        print("SHA256 verified.")

    # Optional secondary copy to satisfy existing defaults
    if args.also_copy_to_src:
        secondary = Path("src") / "routerbench" / "routerbench_5shot.pkl"
        secondary.parent.mkdir(parents=True, exist_ok=True)
        if not secondary.exists() or sha256sum(secondary) != sha256sum(primary):
            print(f"Copying to {secondary}")
            shutil.copy2(primary, secondary)
        else:
            print(f"Secondary already up to date at {secondary}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

