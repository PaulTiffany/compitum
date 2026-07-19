from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


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

    ids = re.findall(r"\((\d+)\)", text)
    with out_path.open("a", encoding="utf-8") as f:
        for mutant_id in ids:
            try:
                details = subprocess.check_output(["mutmut", "show", mutant_id], text=True, errors="ignore")
            except Exception as exc:
                details = f"<error showing {mutant_id}: {exc}>"
            f.write(f"\n## survivor {mutant_id}\n\n{details}\n")


if __name__ == "__main__":
    main()
