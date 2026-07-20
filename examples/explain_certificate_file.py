from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.certificate_card import render_markdown_card  # reuse


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Explain an existing certificate JSON as a Markdown card."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to certificate JSON or JSONL (uses first line)",
    )
    args = ap.parse_args()

    p = args.input
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        line = text.splitlines()[0]
        data = json.loads(line)
    else:
        data = json.loads(text)
    print(render_markdown_card(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
