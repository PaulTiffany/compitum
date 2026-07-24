# ADR 0002: independent constraint-pressure oracle (tranche 2)

Status: accepted, observation-only. Supersedes nothing in ADR 0001; narrows
tranche 2's research question per explicit user direction after tranche 1's
generic-trajectory negative result.

## Correction to the roadmap

Tranche 1 tested and closed the broad hypothesis: "generic FabricPC
trajectory summaries add held-out routing information beyond frozen
Compitum v0.2.0." That negative result stands, published in the branch
history, and is not reinterpreted here.

The actual long-term objective is narrower: can FabricPC estimate
*prospective constraint pressure* -- which constraints are approaching
activation, how much relaxation would change the feasible optimum, what
marginal utility a constraint currently suppresses -- earlier or better
than Compitum's static state and its existing finite-difference
`shadow_prices` diagnostic? Tranche 2 tests this narrower question, still
observation-only.

## Preserved semantics

Unchanged in this tranche: `SwitchCertificate`, `constraints.shadow_prices`,
`ReflectiveConstraintSolver`, routing decisions, utility calculations,
controller behavior, the `v0.2.0` tag, public documentation. The existing
`shadow_prices` (a fixed 1e-5 finite-difference probe on one competitor,
first-found rather than exact) remains the frozen reference diagnostic --
read-only, one baseline comparison arm, never the target authority.

## Naming discipline

Nothing produced by `compitum.constraint_oracle` or tranche 2's FabricPC
observers is named `shadow_price` or "dual variable". Field and schema names
in use: `critical_relaxation`, `constraint_pressure`,
`predicted_constraint_pressure` (reserved for a future learned estimate, not
yet implemented), `marginal_utility_improvement`, `best_suppressed_competitor`.
The intended three-way distinction, for later tranches:

```text
existing shadow_prices  = retrospective local finite-difference diagnostic
online dual variables   = persistent controller state (not yet implemented)
FabricPC pressure est.  = prospective observation informing, not replacing, either
```

## Independent oracle, not a copy of shadow_prices

`src/compitum/constraint_oracle/static.py` computes exact ground truth by
directly analyzing `ReflectiveConstraintSolver`'s own frozen feasibility
structure -- not by relaxing at the same fixed 1e-5 step the production
code uses. Two structural facts, confirmed by reading the frozen source and
cross-checked against a numeric bisection search of the actual solver
(`tests/constraint_oracle/test_static_oracle.py`), make the critical
relaxation Δb_i* an exact closed-form quantity rather than requiring
adaptive search:

1. `_is_feasible`'s linear check `A @ xB <= b + 1e-10` uses the same `xB`
   for every model; only `capabilities.supports(...)` varies per model. The
   slack vector `b - A @ xB` is identical across the whole pool for a case.
2. `_is_feasible` requires `np.all(...)` over every constraint row at once,
   so a single violated constraint makes every model simultaneously
   linearly infeasible. "A feasible route already exists" and "some
   constraint is currently violated" never co-occur -- there are exactly
   two live branches (linear-feasible, decided by capability alone; or
   linear-infeasible, unconditionally the frozen `infeasible_fallback`
   path), not four.

This is a genuine simplification discovered while implementing, not an
assumption -- an earlier draft of the oracle modeled a third, impossible
branch and its test suite caught the contradiction immediately.
