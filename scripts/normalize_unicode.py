#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def normalize_readme(p: Path) -> None:
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for line in text:
        # Fix the Core Science bullets if present
        if line.lstrip().startswith("- Coherence:"):
            out.append(
                "- Coherence: monotone outward, ±v symmetry, inward score direction, mixture discrimination."
            )
            continue
        if line.lstrip().startswith("- Constraints:"):
            out.append(
                "- Constraints: feasibility monotone; duals slack ≈ 0, boundary ≈ 0; monotone/scale sanity."
            )
            continue
        if line.lstrip().startswith("- Pedagogy:"):
            out.append(
                "- Pedagogy: practice raises evidence/utility (beta_s > 0); prepared environment fixes constraints."
            )
            continue
        if line.lstrip().startswith("- Stability:"):
            out.append(
                "- Stability: Lyapunov decay/saturation/recovery; ΔV proxy sequences; combined update boundedness."
            )
            continue
        if line.startswith("## RouterBench Data ("):
            out.append("## RouterBench Data (5-shot pickle)")
            continue

        # Generic character cleanups
        fixed = (
            line.replace("A�v", "±v")
            .replace("�v", "±v")
            .replace("I\"V", "ΔV")
            .replace("I�_s", "beta_s")
            .replace("5�?`shot", "5-shot")
            .replace("5�?`", "5-")
        )
        out.append(fixed)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def normalize_invariants(p: Path) -> None:
    text = p.read_text(encoding="utf-8", errors="replace")
    text = text.replace("A � x � b", "A x <= b")
    text = text.replace("(�v)", "(±v)")
    text = text.replace("(�V", "(ΔV")
    p.write_text(text, encoding="utf-8")


def normalize_control_perspective(p: Path) -> None:
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for line in lines:
        if line.strip().startswith("> Related:"):
            out.append(
                "Related: [cs.LG](Learning-Perspective.md), [cs.CL](Language-Perspective.md), [stat.ML](Statistical-Notes.md), [SRMF & Lyapunov](SRMF-as-Lyapunov.md), [Peer Review Protocol](PEER_REVIEW.md), [Certificate Schema](Certificate-Schema.md)"
            )
            continue
        if "r_{t+1} =" in line:
            out.append(
                "  - `r_{t+1} = clip(r_t + f(EMA(d_t), integral(d_t)), r_min, r_max)`"
            )
            continue
        if "I�_cap" in line or "I_cap" in line:
            out.append("  - `I_cap = I / (||grad|| + I)`")
            continue
        if "I�_eff" in line or "I_eff" in line:
            out.append("  - Effective step for metric update: `I_eff = min(I_user, I_cap)`")
            continue
        if "r_t" in line and "[r_min" in line and ("�" in line or "in [" not in line):
            out.append(
                "- Bounded control signals: `r_t in [r_min, r_max]`, `I_eff <= I / (||grad|| + I)`, `||L_t||_F <= L_max`."
            )
            continue
        if "multi" in line and "step" in line and "`step" in line:
            out.append(line.replace("multi�?`step", "multi-step"))
            continue
        if "symmetry (" in line and "symmetry (±v)" not in line and "symmetry" in line:
            out.append(line.replace("(A�v)", "(±v)").replace("(�v)", "(±v)"))
            continue
        # Generic replacements
        fixed = (
            line.replace(" ��", " &")
            .replace(" A� ", ", ")
            .replace("A� ", ", ")
            .replace(" �^� ", " grad ")
            .replace("I�", "I")
        )
        out.append(fixed)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    normalize_readme(root / "README.md")
    normalize_invariants(root / "docs" / "Invariants.md")
    normalize_control_perspective(root / "docs" / "Control-Perspective.md")
    # Also normalize and lightly edit the project homepage index.html to add Examples links
    index_html = root / "index.html"
    if index_html.exists():
        raw = index_html.read_text(encoding="latin-1", errors="replace")
        if "docs/Examples.html" not in raw:
            updated = raw
            # Prefer inserting within the <nav class="nav" ...> element
            nav_start = raw.find('<nav class="nav"')
            if nav_start != -1:
                nav_close = raw.find('</nav>', nav_start)
                if nav_close != -1:
                    insertion = (
                        '\n                <a href="docs/Examples.html" class="nav-link">Examples</a>'
                        '\n                <a href="https://github.com/PaulTiffany/compitum/tree/main/examples" class="nav-link">Examples Folder</a>'
                    )
                    updated = raw[:nav_close] + insertion + raw[nav_close:]
            if updated != raw:
                index_html.write_text(updated, encoding="utf-8")
    print("Normalization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
