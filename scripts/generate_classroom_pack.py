#!/usr/bin/env python
from __future__ import annotations

import json
import zipfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / "pedagogy_pack.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Demo script
        zf.write(root / "examples" / "pedagogy_control_of_error.py", arcname="examples/pedagogy_control_of_error.py")
        # Lab worksheet
        zf.write(root / "docs" / "Pedagogy-Lab.md", arcname="docs/Pedagogy-Lab.md")
        # Sample certificate (if exists)
        certs = list((root / "reports").glob("certificates_demo.jsonl"))
        if certs:
            zf.write(certs[0], arcname="reports/certificates_demo.jsonl")
        # Small prompt set
        prompt_txt = root / "data" / "pedagogy_prompts.txt"
        prompt_txt.parent.mkdir(parents=True, exist_ok=True)
        prompt_txt.write_text("Prove AM-GM.\nExplain Bayes' rule.\nFactor x^2+2x+1.\n")
        zf.write(prompt_txt, arcname="data/pedagogy_prompts.txt")
    print(f"Wrote: {zpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

