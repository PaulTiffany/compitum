from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable


def _iter_json_lines(text: str) -> Iterable[dict]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Some cosmic-ray dumps output in pretty JSON arrays per line; attempt fallback
            try:
                if line.startswith("[") and line.endswith("]"):
                    arr = json.loads(line)
                    if isinstance(arr, list):
                        for obj in arr:
                            if isinstance(obj, dict):
                                yield obj
            except Exception:
                continue


def summarize_dump_text(dump_text: str, group: str | None = None) -> Dict[str, object]:
    outcomes = Counter()
    mutations = 0
    jobs = 0
    for obj in _iter_json_lines(dump_text):
        # The dump generally alternates metadata and result objects; capture test_outcome if present
        if isinstance(obj, dict) and "test_outcome" in obj:
            outcomes[obj.get("test_outcome", "unknown")] += 1
            jobs += 1
        if isinstance(obj, dict) and "mutations" in obj:
            try:
                mutations += int(len(obj.get("mutations", [])))
            except Exception:
                pass

    # Fallback: tolerant regex scan if structured parse yields nothing
    if jobs == 0:
        for m in re.finditer(r'"test_outcome"\s*:\s*"([^"]+)"', dump_text):
            outcomes[m.group(1)] += 1
            jobs += 1
        # Count approximate mutation blocks
        for _ in re.finditer(r'"mutations"\s*:\s*\[', dump_text):
            mutations += 1

    killed = outcomes.get("killed", 0)
    total = sum(outcomes.values())
    score = (killed / total) if total else 0.0
    return {
        # Self-describing: the shard/group name used to only be recoverable
        # from the output filename, which is lost the moment the JSON is
        # read out of context (e.g. after artifact download/merge).
        "group": group,
        "jobs": jobs,
        "mutations_seen": mutations,
        "outcomes": dict(outcomes),
        "mutation_score": round(score, 4),
    }


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Summarize Cosmic Ray dump into a compact JSON")
    ap.add_argument("dump_path", type=str)
    ap.add_argument("out_path", type=str)
    ap.add_argument("--group", default=None, help="Shard/module name, embedded into the output JSON")
    args = ap.parse_args()

    text = Path(args.dump_path).read_text(encoding="utf-8")
    summary = summarize_dump_text(text, group=args.group)
    Path(args.out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote mutation summary to: {args.out_path}")


if __name__ == "__main__":
    main()
