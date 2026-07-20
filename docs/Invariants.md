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

## Mutation-Hardening Coverage (Boundary & Edge-Case Invariants)

A separate category from the property-based suite above: exact-value and boundary-condition
invariants found via a real mutmut sweep across the full `src/compitum` shard matrix (16/17 files
run to completion; see `MUTATION_HARDENING_STATUS.md` for per-file scores). Each line below was a
genuine survivor -- an existing test exercised the code path but didn't pin down the exact value or
boundary needed to actually kill a behavioral mutation -- confirmed against real code before the
fix, and in `energy.py`'s case confirmed killed against the real mutant diff.

- `effort_qp.py`: `e_star` resolves to `0.0` (not `1.0`) exactly at the `grad == 0` boundary; q1/t1/c1
  multiplier terms are exercised with non-unity values — `tests/test_effort_qp.py`
- `capabilities.py`: `Capabilities.deterministic` defaults to `False` when omitted —
  `tests/test_capabilities.py`
- `boundary.py`: `uncertainty` defaults to `0.0` (and correctly resolves `is_boundary=False`) when
  the winning model is missing from `u_sigma` — `tests/test_boundary.py`
- `predictors.py`: isotonic calibration clips out-of-domain raw values to the boundary's fitted `y`
  rather than extrapolating — `tests/test_predictors.py`
- `control.py`: the EMA trust-region branch is neutral (no change) exactly at the 1.5x/0.7x
  thresholds, not just clearly above/below them — `tests/test_control.py`
- `symbolic.py`: `+`, `*`, `@` operators evaluate correctly, not just `-`/`/` — `tests/test_symbolic.py`
- `security.py`: SHA-256 outputs match the exact expected digest (not just length 64);
  `is_offline()`/`redaction_enabled()` default to `False` when unset — `tests/security/test_security_utils.py`
- `constraints.py`: shadow price matches the exact `(Δutility)/1e-5` value, not just its sign;
  multi-constraint infeasible fallback zeros every `lambda_i` — `tests/test_constraints.py`
- `integrations/matbench_adapter.py`: CSV columns map to the correct attribute *values*, not just
  present attributes — `tests/integrations/test_matbench_adapter.py`
- `coherence.py`: `WeightedReservoir` clamps nonpositive weights to `1e-6` — `tests/test_coherence.py`
- `energy.py`: `comps["uncertainty"]`/`U_var` match the exact variance formula; debug/timing prints
  match their full exact (or regex-matched) content, not just a leading substring; evidence uses
  `xR - model.center`, not `+` — every prior test used a zero center, where the two are
  indistinguishable — `tests/test_energy.py`, `tests/test_energy_debug_paths.py`,
  `tests/energy/test_symbolic_free_energy.py`
- `router.py`: `update_stride` floors to `1` for `stride <= 0`; `router.srmf is router.controller`
  (legacy alias); disabled-controller `drift_status` matches the controller's exact current state,
  not just key presence — `tests/test_router_simple.py`

Known true equivalent mutant (not a gap): `d_best = abs(-comps[...]["distance"])` in both
`route()` and `batch_route()` — `abs(-x) == abs(x)` for all real `x`, so no test can ever
distinguish the two.
