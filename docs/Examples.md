---
title: Examples
description: Runnable examples demonstrating core Compitum concepts.
---

# Examples

This page lists small, runnable examples that illustrate key ideas. Each runs in seconds.

- Folder on GitHub: https://github.com/PaulTiffany/compitum/tree/main/examples
- In-repo overview: examples/README.md

## Quickstart

Runs the CLI to route a single prompt. Add `--trace` for a full certificate.

```bash
compitum route --prompt "Prove the harmonic series diverges." --trace
```

See also `examples/demo_route.py`.

## Learning / Geometry

Synthetic two‑cluster setup; reports average SPD distances to each cluster center.

```bash
python examples/synth_bench.py
# Output: {"avg_d_math": <float>, "avg_d_code": <float>}
```

Interpretation: distances are computed under the SPD metric; clusters closer to their own center.

## Pedagogy / Control of Error

Demonstrates that “practice” (coherence updates) increases evidence (and, if `beta_s > 0`, utility), and that a “prepared environment” (supported region) resolves a constraint loop.

```bash
python examples/pedagogy_control_of_error.py
```

What to observe:
- Before vs. after: higher evidence component (and utility if `beta_s > 0`).
- JP region infeasible; US region feasible.

## Bridge Demo

Internal tooling illustration: `examples/bridge_demo.py` shows how BridgeBlocks mark code spans that can be referenced by external artifacts (e.g., LaTeX snippets).

## Certificate Card (Markdown)

Produce a short, human‑readable summary from a routing certificate.

```bash
python examples/certificate_card.py --prompt "Sketch a proof of AM-GM."
```

Outputs a Markdown card with model, utility, top utility components, boundary diagnostics, feasibility, and trust‑radius.

## Batch Routing

Route a small batch of random embeddings and print a compact JSON summary per sample.

```bash
python examples/batch_route_demo.py --n 3 --D 35
```

## Explain a Saved Certificate

Render a Markdown summary from an existing JSON or JSONL file.

```bash
python examples/explain_certificate_file.py --input reports/certificates_demo.jsonl
```
