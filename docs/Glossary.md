---
title: Glossary
description: Plain-language definitions of key terms used in Compitum.
---

# Glossary

- Utility (U): A single score to compare choices at a given budget, U = performance - lambda * cost.
- Willingness-to-pay (lambda): How strongly cost is penalized in U; higher lambda means cost matters more.
- Constraint: A hard rule the system must follow (e.g., region availability). Evaluated as A x <= b.
- Shadow price:
  - In economics/optimization: a Lagrange multiplier measuring how much relaxing a constraint would improve the objective.
  - In Compitum today: an approximate local sensitivity diagnostic computed by a small finite relaxation; report-only, not used in selection.
- Frontier: The best achievable tradeoff between cost and performance across all choices.
- Frontier gap: Difference between the best utility on a task and Compitum's utility at the same WTP.
- Near-frontier: When the frontier gap is small and/or the model is often at the frontier.
- Boundary diagnostics: Signals about ambiguity/uncertainty — gap (runner-up difference), entropy (score uncertainty), sigma (score spread).
- Drift/trust radius: Mechanisms to keep updates stable and small when conditions change.
- Certificate: The structured JSON record emitted for each decision with all signals for auditing.

