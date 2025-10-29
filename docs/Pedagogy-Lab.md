---
title: Pedagogy Lab
description: A short, hands-on worksheet to demonstrate Control of Error with Compitum.
---

# Pedagogy Lab (Control of Error)

This worksheet shows how "practice" (coherence) increases evidence (and, with I_s > 0, utility), how a prepared environment fixes constraint loops, and how to read a routing certificate.

## 1) Run the Demo

```bash
python examples/pedagogy_control_of_error.py
```

Observe:
- Before practice: decision, utility, distance, evidence.
- After practice: evidence increases and (with I_s > 0) utility increases.
- Prepared environment: region JP is infeasible; region US is feasible.

## 2) Tweak beta_s (coherence weight)

Edit `configs/router_defaults.yaml` and adjust `beta_s` (coherence weight). Re-run the demo and observe how evidence affects utility.

## 3) Explain a Flip

Change the prompt (e.g., add punctuation or replace a word). If the decision flips, use the certificate fields (distance/evidence/feasible) to explain why.

For a single certificate JSON, you can render a quick Markdown card:

```bash
python tools/visualize_certificate.py --input reports/certificates_demo.json
```

## 4) Classroom Pack

Generate a small, self-contained pack for workshops:

```bash
python scripts/generate_classroom_pack.py
```

This writes `artifacts/pedagogy_pack.zip` with:
- Demo script, lab worksheet, a sample certificate JSONL, and a tiny prompt set.

