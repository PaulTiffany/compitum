---
title: Compitum Documentation
description: Geometrically-aware routing with instantaneous, mechanistic feedback (SPD metrics, constraints, Lyapunov-inspired trust-region updates).
---

# Compitum

Geometrically-aware AI routing with instantaneous, mechanistic feedback.

Compitum routes prompts across models using SPD metric geometry, constraint-aware selection, and Lyapunov-inspired trust-region updates. Feedback is immediate and mechanistic (process signals), not judge-based.

```{toctree}
:maxdepth: 2
:caption: Contents

Getting-Started
CLI
Examples
Reviewer-Quickstart
API-Quickstart
ACCESSIBILITY
LLM-Usage
Philosophy
Invariants
Math-Brief
Control-of-Error
Control-Perspective
Trust-From-Regret
Learning-Perspective
Language-Perspective
Instantaneous-Feedback
Mathematical-Bridge
SRMF-as-Lyapunov
Statistical-Notes
Certificate-Schema
Glossary
API-Reference
api/compitum
PEER_REVIEW
REPRODUCIBILITY
Results-Summary
Results-Fixed-WTP
Per-Baseline-WinRate
Frontier-Gap
Panel-Summary
Results-By-Task
RouterBench-Summary
RouterBench-Fairness
Media
Public-API
Artifact-README
Compliance
Operations-Runbook
Performance-Notes
Pedagogy
Pedagogy-Lab
Teach-Compitum
FUTURE
Executive-Overview
```

Useful links: README.md, CONTRIBUTING.md, ACCESSIBILITY.md, SECURITY.md, SUPPORT.md.

## LLM Snapshots

Small, ready-to-share repository snapshot for LLM context:

- {download}`repo_snapshot.jsonl` - lean pack (~39k tokens) with core docs and invariants.

Notes:
- JSONL files live alongside this page and are excluded from future snapshots.
- Prefer sharing the lean pack with Claude; it stays well under typical limits.

### Ask Your LLM

- Attach the JSONL in chat and say: "Use the attached repo snapshot (JSONL lines with path+content). Cite `path:line` in answers."
- Or with browsing enabled, ask it to fetch: `https://compitum.space/docs/repo_snapshot.jsonl`.







