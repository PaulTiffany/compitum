# Claims

Separated by evidentiary strength. Every claim below is traceable to a
specific artifact in `artifact_manifest.json`; none is asserted without
a corresponding report, test, or JSON result.

## Proved or exact

- The discrete action shadow charge
  `C_t(a) = V_{t+1}(B_t,q_{t+1}) - V_{t+1}(B_t-c_t(a),q_{t+1})` is
  mathematically equivalent, up to an action-independent additive
  constant, to full Bellman-Q action selection. This is a closed-form
  algebraic identity (see ADR 0008), not an empirical approximation.
- Given the exact belief, the shadow-charge policy's selected action is
  bit-identical to the literal Bellman-optimal online policy at every
  step. Verified with zero mismatches across the original 35-sequence
  held-out test set and five independent robustness seeds (tranche 6.5),
  and separately re-verified at Gate 0's tuned (non-default) transition
  /observation parameters (15 additional seeds, tranche 7).
- The telescoping identity — the sum of exact unit marginal prices
  equals the closed-form action shadow charge — holds exactly, including
  for negative (credit-direction) net consumption, verified across many
  `(budget, belief, num_units)` combinations.
- Zero constraint violations occur in any shadow-charge arm across
  either tranche's pilot, for a provable, not merely observed, reason:
  action selection is always restricted to the feasible set.

## Empirically supported

- Exact-belief shadow-charge pricing recovers the full economic gap over
  frozen pacing in the belief-sensitive environment (tranche 7):
  recoverable gap 0.371, exact-belief regret 0.000.
- The belief-estimation task in the tranche-7 environment is genuinely
  learnable: ordinary ridge regression recovers exactly zero regret, tied
  with the exact-belief oracle and the true-parameter HMM filter.
- A genuinely trained FabricPC model (both predictive-coding and
  same-topology-backprop training) recovers belief information
  materially above a naive constant-prediction baseline.
- FabricPC's predictive-coding and backprop training runs are
  statistically indistinguishable from each other under the tested,
  fixed, small topology (tied exactly on held-out regret: 0.314 mean,
  zero variance in their paired difference across 35 test sequences).

## Negative findings

- Tranche 6's scalar `lambda * consumption` pricing, using the
  economically-exact Bellman marginal price, does not beat frozen
  pacing (Gate A failed). This is a genuine, well-supported negative
  result about that specific *linear translation*, not about Bellman
  pricing or belief estimation generally — a distinction later tranches
  confirmed by fixing the translation and recovering the gap in full.
- Tranche 6.5's environment (before tranche 7's redesign) had no
  reachable state where belief affected the Bellman-optimal action —
  confirmed two independent ways (a direct belief-sensitivity scan and
  the pilot's own 0% boundary-crossing measurement across all arms,
  including a shuffled control).
- FabricPC, under the fixed small topology and bounded training design
  (one topology, three seeds, no architecture or hyperparameter search),
  does not clear the primary economics gate in the belief-sensitive
  environment: it does not significantly beat frozen pacing, a
  fixed-prior control, or a shuffled-belief control; it captures only
  15.4% of the recoverable regret gap; its belief-prediction MSE is
  roughly 600x worse than a plain ridge regression on the identical
  declared features.

## Limitations

- All quantitative results come from small, bounded pilots (35 held-out
  test sequences, 3 training seeds, one fixed FabricPC topology, 10
  -step horizons). No claim here should be read as established at
  production scale or under a wider architecture/hyperparameter search.
- The FabricPC-vs-ridge comparison used one declared small topology
  (`source(55)->hidden(16,sigmoid)->belief(1,sigmoid,GaussianEnergy)`,
  ~30 training epochs). Whether a larger or differently configured
  FabricPC network would close the measured ~600x belief-MSE gap is an
  explicitly open question this program did not investigate, per its own
  runtime-discipline mandate.
- Statistical significance in the ten-arm pilot is limited by sample
  size (35 test sequences); several point-estimate differences (e.g.
  FabricPC vs. fixed-prior, vs. shuffled) do not reach 95% confidence in
  either direction.
- Gate 0's own selection criterion (exact belief beating belief-blind
  controls) was checked by point-estimate direction, not full
  statistical significance, appropriate to its role as a 30-sequence
  feasibility screen — the pilot's own primary economics gate correctly
  requires full statistical significance, and that is where FabricPC's
  results fail to clear the bar.

## Not claimed

- **FabricPC improved Compitum's production routing.** It did not; no
  production code path was ever changed, and the pilot's own primary
  economics gate for FabricPC failed.
- **Predictive coding is inferior to backpropagation in general.** The
  two training rules tied exactly under this specific small, fixed
  topology and bounded budget — this is not evidence about predictive
  coding as a learning paradigm more broadly.
- **The belief-sensitive environment or shadow-charge mechanism is
  unsuitable for further work.** The opposite: the economic mechanism
  and the environment's belief-learnability are both validated; only
  FabricPC's specific learned representation, at this bounded scale,
  fell short of a simple baseline.
- **Any change to production `constraints.shadow_prices` or
  `SwitchCertificate` is justified by this work.** No such change was
  made or is recommended by this record; the shadow-charge insight is
  noted only as a portable lesson for a hypothetical future redesign.
