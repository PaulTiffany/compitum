from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import List

# mutmut's actual `mutmut results` output looks like:
#
#   Suspicious (3)
#
#   ---- src/compitum/metric.py (3) ----
#
#   30, 32, 34
#
#   Survived (20)
#
#   ---- src/compitum/metric.py (20) ----
#
#   9, 11, 13, 28, 40, 42-44, 48, 57, 61-62, 64-65, 70, 76, 80, 84-85, 91
#
# Real survivor IDs are the bare comma/range-separated numbers on their own
# line under the "Survived" section -- not the parenthesized counts on every
# header and "---- <path> (n) ----" separator line. A naive `\((\d+)\)` match
# picks up those counts instead (e.g. the "(3)" from "Suspicious (3)"), never
# the real IDs, which don't have parentheses around them at all.
_OUTCOME_HEADERS = ("Killed", "Timeout", "Suspicious", "Survived", "Skipped")


def _extract_survivor_ids(text: str) -> List[int]:
    m = re.search(r"^Survived\b.*$", text, re.MULTILINE)
    if not m:
        return []
    rest = text[m.end():]
    next_header = re.search(
        r"^(?:" + "|".join(h for h in _OUTCOME_HEADERS if h != "Survived") + r")\b",
        rest, re.MULTILINE,
    )
    section = rest[: next_header.start()] if next_header else rest
    id_lines = [
        line for line in section.splitlines()
        if line.strip() and not line.strip().startswith("----")
    ]
    ids: List[int] = []
    for line in id_lines:
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo, hi = token.split("-", 1)
                ids.extend(range(int(lo), int(hi) + 1))
            else:
                ids.append(int(token))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Append mutmut survivor details for one shard to a report file")
    ap.add_argument("--base", required=True, help="Sanitized shard basename, e.g. metric or integrations_matbench_adapter")
    args = ap.parse_args()

    results_path = Path(f"reports/mutmut_results_{args.base}.txt")
    out_path = Path(f"reports/mutmut_survivors_{args.base}.txt")

    try:
        text = results_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        text = ""

    ids = [str(i) for i in _extract_survivor_ids(text)]
    with out_path.open("a", encoding="utf-8") as f:
        for mutant_id in ids:
            try:
                details = subprocess.check_output(["mutmut", "show", mutant_id], text=True, errors="ignore")
            except Exception as exc:
                details = f"<error showing {mutant_id}: {exc}>"
            f.write(f"\n## survivor {mutant_id}\n\n{details}\n")


if __name__ == "__main__":
    main()
