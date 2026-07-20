#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_markdown(data: dict) -> str:
    comps = data.get("utility_components", {})
    cons = data.get("constraints", {})
    bound = data.get("boundary", {})
    lines = ["# Routing Certificate", ""]
    lines.append(f"- Model: `{data.get('model')}`  ")
    lines.append(f"- Utility: `{data.get('utility')}`  ")
    lines.append("\n## Components")
    for k in ("quality", "latency", "cost", "distance", "evidence", "uncertainty"):
        if k in comps:
            lines.append(f"- {k}: `{comps[k]}`")
    lines.append("\n## Constraints")
    lines.append(f"- feasible: `{cons.get('feasible')}`")
    sp = cons.get("shadow_prices", {}) or {}
    if sp:
        sp_view = ", ".join(f"{k}={v}" for k, v in sp.items())
        lines.append(f"- shadow_prices: `{sp_view}`")
    lines.append("\n## Boundary")
    for k in ("winner", "runner_up", "utility_gap", "entropy", "uncertainty", "is_boundary"):
        if k in bound:
            lines.append(f"- {k}: `{bound[k]}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize a routing certificate JSON as Markdown")
    ap.add_argument("--input", required=True, help="Path to certificate JSON")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(to_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
