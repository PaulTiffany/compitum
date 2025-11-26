"""Lightweight mojibake checker used in CI docs lane."""

from __future__ import annotations

import sys
from pathlib import Path

# File extensions that might contain human-readable text.
TEXT_EXTS = {
    ".md",
    ".rst",
    ".txt",
    ".html",
    ".yml",
    ".yaml",
    ".py",
}


def main() -> int:
    bad_files: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\ufffd" in text:  # Unicode replacement char is a good mojibake proxy
            bad_files.append(path)

    if bad_files:
        print("Found potential mojibake markers in:")
        for p in sorted(bad_files):
            print(f"- {p}")
        return 1

    print("No mojibake markers found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
