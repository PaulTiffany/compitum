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

## Declared channel mapping (tranche 2.4)

FabricPC observes only state Compitum itself already has; it never consumes
the oracle's own targets. `src/compitum/constraint_oracle/channels.py`
defines a fixed, 17-dimensional, documented-order vector: normalized
per-constraint slack, a feasibility mask by model, per-model utilities, the
winner/runner-up utility gap, utility-distribution entropy (same formula as
`BoundaryAnalyzer.analyze`), the frozen `LyapunovController`'s own drift
state (reused unmodified, mutated once per sequence step exactly as
`CompitumRouter` does), and two step-to-step transition indicators
(violated-constraint-set Jaccard distance; selected-model change). Not
modeled in the controlled track: per-model utility *components* and
resource-utilization history, since the controlled dataset generator
bypasses `CompitumRouter` and has no `SymbolicFreeEnergy` breakdown or real
usage telemetry to draw on -- both are candidates for a later realized
-routing track, not fabricated here.

`experiments/fabricpc/tranche2/fabricpc_channel_observer.py` (JAX-side, runs
only under `.venv-fabricpc`) observes this vector through a
source(17)-hidden(8)-latent(4) graph, reusing tranche 1's established
verified-receipt/lightweight-history pattern. One finding worth recording:
an all-zero channel vector is a degenerate probe -- a linear map of zero is
zero regardless of network weights, so it forces "hidden"'s prediction
error (and therefore its energy) to exactly 0.0 regardless of the network's
random seed. Confirmed directly (two different network seeds against an
all-zero vector both yield terminal hidden energy 0.0; a real, non
-degenerate channel vector from an actual generated sequence does not).
This must not be used as a "no signal" baseline probe in later comparison
arms -- it is mathematically forced, not evidence of anything.
