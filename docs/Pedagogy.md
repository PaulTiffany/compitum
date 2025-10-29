---
title: Pedagogy & Control of Error
description: Connecting educator practice (e.g., Montessori's control of error) with Compitum's mechanistic feedback and constraint design.
---

# Pedagogy and Control of Error

Context

- Practical educators have long used immediate, interpretable feedback to guide learning. In the Montessori tradition, "control of error" means a learner can detect and correct mistakes through the design of the materials and environment, not only through an external judge.
- Compitum adopts a similar stance for artificial learners and decision systems: provide mechanistic, instant feedback about each routing decision so the system can self-correct without opaque scoring.

Analogies (Educator + Compitum)

- Control of error + Routing certificate fields that expose utility components, constraint feasibility, and boundary diagnostics (gap, entropy, sigma) so mistakes are visible where they occur.
- Prepared environment + Constraints (A x <= b) and trust-region updates (drift/EMA/integral) that bound behavior and keep changes small and comprehensible.
- Self-correction + Utility decomposition (quality, latency, cost) and shadow prices that show precisely which factors drove a choice, enabling targeted adjustments.
- Practical tasks + Fixed WTP slices and per-task summaries that link choices to clear, comparable objectives.

Design Principles for "Teachable" Systems

- Make errors legible: show local signals (gap to runner-up, uncertainty) at the moment of choice.
- Keep the budget explicit: use U = performance - lambda * cost; vary lambda to show cost-sensitivity.
- Constrain for safety: treat constraints as policy hooks; expose shadow prices so tradeoffs are auditable.
- Update gently: enforce trust regions to avoid destabilizing jumps; prefer iterative, interpretable change.

Classroom Bridges

- Show a certificate and ask: "What would you change if gap is small but entropy is high?" (Often: defer or choose a safer model.)
- Vary lambda (0.1 and 1.0) and observe selection shifts. Discuss how cost aversion mirrors classroom constraints (time, attention).
- Identify a binding constraint (nonzero shadow price) and propose a policy-aware relaxation; predict the effect on utility.

Further Resources

- {doc}`Teach-Compitum` - educator guide and activities
- {doc}`Certificate-Schema` - field definitions and schema
- {doc}`Math-Brief` - plain-language math overview
- {doc}`PEER_REVIEW` - reproducibility and evidence package

## Evidence of "Control of Error" (0.1.1)

- Practice improves performance where the environment encodes feedback
  - Coherence reservoir updates around the winner's whitened vector increase evidence and (with I_s > 0) utility.
  - Test: `tests/pedagogy/test_control_of_error_practice_improves.py`
- Prepared environment makes errors fixable
  - Constraint loops are visible in certificates (feasible=false) and fixed by setting supported context (e.g., region).
  - Test: `tests/pedagogy/test_control_of_error_constraints_loop.py`
- Teacherly override in ambiguous regions
  - When boundary flags an ambiguous decision (small gap, high uncertainty), a conservative override reduces uncertainty.
  - Test: `tests/pedagogy/test_boundary_override_teacher_action.py`

