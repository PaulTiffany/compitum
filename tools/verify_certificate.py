from __future__ import annotations

"""
Verify and hash Compitum routing certificates.

Provides:
- Canonical JSON hashing (stable key order, no whitespace variance)
- Optional JSON Schema validation (if assets/certificate.schema.json present)

Usage:
  python tools/verify_certificate.py <certificate.json>
  python tools/verify_certificate.py --stdin
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


def canonical_hash(obj: Dict[str, Any]) -> str:
    # Canonical dump with sorted keys and no insignificant whitespace
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def maybe_validate_schema(obj: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:  # pragma: no cover
        print("[verify-cert][WARN] jsonschema not installed; skipping schema validation", file=sys.stderr)
        return
    schema_path = Path("assets") / "certificate.schema.json"
    if not schema_path.exists():
        print("[verify-cert][INFO] certificate.schema.json not found; skipping schema validation", file=sys.stderr)
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8", errors="ignore"))
    jsonschema.validate(instance=obj, schema=schema)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Compitum certificate JSON")
    ap.add_argument("path", nargs="?", help="Path to certificate JSON (or --stdin)")
    ap.add_argument("--stdin", action="store_true", help="Read certificate JSON from stdin")
    args = ap.parse_args()

    if args.stdin:
        try:
            obj = json.loads(sys.stdin.read())
        except Exception as e:
            print(f"[verify-cert][FAIL] could not read JSON from stdin: {e}", file=sys.stderr)
            return 2
    else:
        if not args.path:
            ap.error("Provide a path or use --stdin")
        p = Path(args.path)
        if not p.exists():
            print(f"[verify-cert][FAIL] missing file: {p}", file=sys.stderr)
            return 2
        obj = load_json(p)

    try:
        maybe_validate_schema(obj)
    except Exception as e:
        print(f"[verify-cert][FAIL] schema validation error: {e}", file=sys.stderr)
        return 2

    h = canonical_hash(obj)
    print(json.dumps({"sha256": h}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

