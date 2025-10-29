# Invariants

Compitum defends a set of invariants checked by tests and by runtime process signals:

- Lyapunov-like energy drift does not increase under allowed updates.
- Constraint residuals respect `A · x ≤ b` with documented slack.
- Metric coherence: neighborhoods act like neighborhoods in the SPD metric.
- Boundary behavior: tie regions are observable with gap/entropy/uncertainty checks.

See the `tests/invariants/` suite for details and run with `pytest -m invariants`.

## Coverage (selected)

- Geometry (SPD metric)
  - Ray monotonicity: `test_invariants_metric_ray.py`
  - Triangle inequality: `test_invariants_metric_triangle.py`
  - Update descent + SPD: `test_invariants_metric_update.py`
  - SPD eigenvalue bounds: `tests/metric/test_metric_spd_bounds.py`

- Coherence / OOD
  - Monotone outward: `test_invariants_coherence.py`
  - Symmetry (±v): `test_invariants_coherence_symmetry.py`
  - Density–distance coupling: `test_invariants_density_energy_coupling.py`
  - Score directionality (finite diff): `test_invariants_coherence_score_dir.py`
  - Mixture discrimination: `tests/coherence/test_coherence_mixture_discrimination.py`

- Control / Lyapunov
  - Properties (bounds, monotonicity): `test_invariants_control_props.py`
  - Sequences (ΔV non-increase/bounded, recovery): `test_invariants_control_sequences.py`, `test_invariants_control_lyapunov.py`, `test_invariants_control_deltaV_strong.py`, `test_invariants_control_combined_proxy.py`

- Constraints / Duals
  - Feasibility monotone in b: `test_invariants_constraints_monotone.py`
  - Dual sanity/monotone/near-binding: `test_invariants_constraints_duals.py`, `test_invariants_duals_monotone.py`, `test_invariants_duals_near_binding.py`
  - Dual scaling sanity: `tests/invariants/test_invariants_duals_scaling.py`
  - Argmax stability: `test_invariants_solver_argmax.py`

- Router / Certificates
  - Determinism (repeated route): `test_invariants_router_determinism.py`
  - Paraphrase flip budget + explainability: `test_paraphrase_suite.py`, `test_paraphrase_explainability.py`
  - JSON structure/schema: `tests/certificates/*.py`

Run the fast suite:

```
pytest -q tests/invariants
```
