from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    rep = "\uFFFD"  # Unicode replacement char
    bad: list[str] = []
    roots = [Path("README.md"), *Path("docs").rglob("*.md")]
    for p in roots:
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            # If decoding fails strictly, re-read with replace to catch any hidden issues
            text = p.read_text(encoding="utf-8", errors="replace")
        if rep in text:
            bad.append(str(p))
    if bad:
        print("[mojibake][FAIL] replacement character found in:")
        for b in bad:
            print(" -", b)
        return 2
    print("[mojibake][OK] no replacement characters in README/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

