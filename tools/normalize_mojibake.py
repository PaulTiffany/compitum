from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple


DEFAULT_ALLOWLIST = (
    "README.md",
    "docs/*.md",
    "docs/**/*.md",
)


REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    # Common mojibake seen in previous revisions
    (re.compile(r"I\"V"), "ΔV"),
    (re.compile(r"A�v"), "±v"),
    (re.compile(r"�%\^\s*0"), "≥ 0"),
    (re.compile(r"\b5\?�?`?shot\b"), "5-shot"),
    (re.compile(r"\bmulti\?�?`?step\b"), "multi-step"),
    (re.compile(r"RouterBench\?�?`?based"), "RouterBench-based"),
    # Stray replacement character U+FFFD in ASCII contexts
    (re.compile("\uFFFD"), ""),
)


def iter_paths(patterns: Iterable[str]) -> Iterable[Path]:
    for pat in patterns:
        for p in Path().glob(pat):
            if p.is_file():
                yield p


def normalize_file(p: Path, write: bool = False) -> int:
    text = p.read_text(encoding="utf-8", errors="replace")
    orig = text
    for pat, repl in REPLACEMENTS:
        text = pat.sub(repl, text)
    changed = int(text != orig)
    if changed and write:
        p.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize mojibake in docs/README only")
    ap.add_argument("--write", action="store_true", help="Write changes (default: dry-run)")
    ap.add_argument("--include", action="append", default=[], help="Extra glob to include")
    args = ap.parse_args()

    patterns = list(DEFAULT_ALLOWLIST) + list(args.include)
    changed_total = 0
    scanned = 0
    for p in iter_paths(patterns):
        scanned += 1
        changed = normalize_file(p, write=args.write)
        if changed:
            changed_total += 1
            print(f"[normalize] fixed: {p}")
    print(f"[normalize] scanned={scanned} changed={changed_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

