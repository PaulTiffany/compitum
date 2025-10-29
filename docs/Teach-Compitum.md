---
title: Teach Compitum
description: An educator-friendly guide with lesson objectives, glossary links, and simple classroom activities.
---

# Teach Compitum (Educator Guide)

Audience

- Upper-undergraduate/graduate CS, data science, or AI ethics courses; technical managers learning cost-quality tradeoffs.

Learning Objectives

- Interpret a routing certificate (utility components, constraints, boundary diagnostics, drift).
- Explain utility U = performance - lambda * cost and willingness-to-pay (WTP).
- Describe near-frontier behavior and why "close to optimal" can be valuable even without envelope wins.
- Connect formal constraints to policy/compliance requirements.

Prerequisites

- Comfortable with basic probability, optimization intuition, and Python. No deep ML required.

Before Class (10-15 min)

- Read {doc}`Math-Brief` (plain-language summary).
- Skim {doc}`Certificate-Schema` and {doc}`PEER_REVIEW` (Routing Certificate section) to see what evidence looks like.
- Optional: Watch/describe the media clip {doc}`Media` for intuition.

In-Class Activity (30-45 min)

1) Certificate walk-through (10 min)
   - Run: `compitum route --prompt "Sketch a proof for AM-GM inequality." --trace`
   - Identify fields: `utility_components`, `constraints.feasible`, `constraints.shadow_prices`, `boundary_analysis.gap`.

2) Cost-quality tradeoff (10-15 min)
   - Define U = performance - lambda * cost. Discuss how increasing lambda penalizes cost.
   - Show that different WTP slices (0.1 vs. 1.0) can change selections.

3) Constraints as safety/policy hooks (10 min)
   - Map constraints to compliance (e.g., region limits). Interpret shadow prices: which limits "bind" the choice?

4) Near-frontier behavior (5-10 min)
   - Discuss frontier gap with 95% CIs. Why being near-frontier can be enough when constraints are respected.

5) Control of error (5-10 min)
   - Read: {doc}`Pedagogy` (first two sections). Connect "control of error" to the certificate.
   - Prompt: "Which certificate fields make mistakes legible at the moment of choice?"
   - Extension: Propose a small change to constraints or lambda and predict the effect; then test.

Assessment Ideas

- Short quiz: define utility and WTP; interpret one certificate; explain shadow prices in one sentence.
- Small assignment: change lambda and describe selection shifts.

Resources

- {doc}`Math-Brief` - plain-language summary.
- {doc}`Certificate-Schema` - fields and JSON schema.
- {doc}`PEER_REVIEW` - evidence pack and reproduction steps.
- {doc}`Media` - optional video + text description.
- {doc}`RouterBench-Fairness` - how we ensure fair evaluations.
- {doc}`Pedagogy` - educator perspective linking "control of error" to mechanistic feedback.

Accessibility

- Use the text description provided in {doc}`Media`.
- CLI outputs are JSON; screen-reader-friendly.
- Keep slides high-contrast and avoid color-only cues.

