# Invariants

Compitum defends a set of invariants checked by tests and by runtime process signals:

- Lyapunov-like energy drift does not increase under allowed updates.
- Constraint residuals respect `A · x ≤ b` with documented slack.
- Metric coherence: neighborhoods act like neighborhoods in the SPD metric.
- Boundary behavior: tie regions are observable with gap/entropy/uncertainty checks.

See the `tests/invariants/` suite for details and run with `pytest -m invariants`.
