"""
Derive which test files actually exercise a given source module, from an
already-generated coverage database with per-test contexts.

Used by the cr-quick-shard job (.github/workflows/mutation.yml) to scope
Cosmic Ray's per-mutant test-command down to the test files that cover the
shard's target module, instead of re-running the entire suite for every
mutant.

Requires the coverage database to have been produced with
`--cov-context=test` (or equivalent), so each covered line is attributed to
the pytest node ID that exercised it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coverage


def derive_test_files(coverage_file: str, target: str) -> list[str]:
    cov = coverage.CoverageData(basename=coverage_file)
    cov.read()

    target_resolved = Path(target).resolve()
    measured = [f for f in cov.measured_files() if Path(f).resolve() == target_resolved]
    if not measured:
        return []

    test_files: set[str] = set()
    for f in measured:
        by_line = cov.contexts_by_lineno(f)
        for contexts in by_line.values():
            for ctx in contexts:
                if not ctx:
                    continue
                # pytest-cov contexts look like "tests/test_x.py::test_y|run"
                node_id = ctx.split("|", 1)[0]
                file_part = node_id.split("::", 1)[0]
                if file_part:
                    test_files.add(file_part)
    return sorted(test_files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-file", default=".coverage")
    ap.add_argument(
        "--target", required=True, help="Repo-relative source path, e.g. src/compitum/energy.py"
    )
    ap.add_argument("--out", required=True, help="Output path: one test file per line")
    args = ap.parse_args()

    test_files = derive_test_files(args.coverage_file, args.target)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" disables platform newline translation -- this file is read
    # by a bash script (mapfile) in CI, which needs bare \n regardless of
    # what OS generated it (Windows' default text-mode write would otherwise
    # emit \r\n, leaving a stray \r glued onto every path).
    if test_files:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(test_files) + "\n")
        print(f"Derived {len(test_files)} covering test file(s) for {args.target}:")
        for t in test_files:
            print(f"  {t}")
    else:
        with out_path.open("w", encoding="utf-8", newline=""):
            pass
        print(
            f"WARNING: no coverage data found for {args.target} in {args.coverage_file} -- "
            "falling back to full-suite test selection",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
