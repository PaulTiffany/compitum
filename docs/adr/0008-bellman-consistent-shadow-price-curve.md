# ADR 0008: Bellman-consistent discrete shadow-price curve (tranche 6.5)

Status: accepted, observation-only. Corrects tranche 6's price-to-action
translation; does not reopen tranche 6's own environment, oracle, or
belief-estimation infrastructure, all of which are reused unchanged.

## Governing correction

Tranche 6 was accepted as identifying a precise translation failure, not
as closing FabricPC×Compitum: Gate A stopped before Part B ever trained
or evaluated FabricPC, so no conclusion about trained belief estimation
is supported by that tranche. Its actual, narrow result:

> A scalar Bellman marginal price, translated through the existing
> linear greedy `utility - lambda * consumption` rule, did not beat
> pacing.

`BeliefPricingController`'s routing rule assumes the continuation value
is locally linear over the *entire* amount a candidate action consumes:

```text
opportunity_cost(action) ~= lambda_at_current_budget * action_consumption
```

That approximation is invalid here because resource quantities are
discrete, route consumptions are lumpy (1, 2, or 4 units), the Bellman
value function has kinks and feasibility steps (hand-verified in
`belief_bellman.py` to spike sharply at exact feasibility boundaries),
and a single action can cross several of those marginal-value regions at
once while the scalar price is only ever evaluated at one budget margin.

## The correction: price the whole action, not a per-unit rate

For a route consuming net budget `k * delta` (consumption net of this
step's own replenishment), the exact opportunity cost is the actual drop
in continuation value the action causes -- the telescoping sum of exact
unit marginal prices, which collapses to two value-function lookups:

```text
lambda_unit[j] = V(t+1, B-(j-1)*delta, belief_next) - V(t+1, B-j*delta, belief_next)
shadow_charge(action) = sum(lambda_unit[j] for j in 1..k)
                       = V(t+1, B, belief_next) - V(t+1, B-k*delta, belief_next)

score(action) = immediate_utility(action) - shadow_charge(action)
```

`score(action)` is equal, up to an action-independent additive constant
(`V(t+1, B, belief_next)`, the same term for every candidate), to the
full Bellman action value `Q(action) = immediate_utility(action) +
continuation_value_after(action)` -- so `argmax score == argmax Q`.
Implemented in `src/compitum/regret_lab/belief_action_pricing.py`
(`unit_marginal_prices`, `action_shadow_charge`, `run_shadow_charge_policy`).
Never overwrites or redefines production `constraints.shadow_prices`.

## Belief timing audit

The continuation value after this step's action must use `belief_next`
-- the belief *after* incorporating this step's own now-observed signal,
projected forward one transition -- never `belief_prior_t` (before
observing) or `belief_posterior_t` (after observing, not yet projected).
This is genuinely different from tranche 6's scalar price, which
evaluated `marginal_price` at `belief_prior_t` (correct for *that*
formula's own definition, which marginalizes over the not-yet-observed
signal, but wrong for a per-action continuation-value lookup once the
signal is known). `run_shadow_charge_policy` computes, at every step,
before selecting an action: `prior = belief_estimator.current_belief()`,
`posterior = filtered_belief(prior, observed)`, `belief_next =
predict_belief(posterior)` -- and uses `belief_next`, not `prior`, for
every `action_shadow_charge` lookup. Every step's full trace (prior,
observation, filtered posterior, predicted next belief, budget
before/after, scalar price, per-action shadow charge, per-action
Bellman-Q, selected action) is recorded in `StepTrace` for direct
auditability.

## Gate A-prime: translation correctness (a theorem, not a statistic)

Before spending any compute on FabricPC: with the exact belief, this
module's selected action must be bit-identical, at every step, to
`belief_online_optimum.run_online_optimal_policy`'s own choice --
because `score(action)`'s argmax is mathematically forced to equal
`Q(action)`'s argmax once `belief_next` is exact, this is a correctness
theorem of the implementation, not a statistical hope. Verified in
`tests/regret_lab/test_belief_action_pricing.py`: identical choices and
cumulative utility across 35 independent seeds, zero regret against the
exact online optimum, the telescoping identity holding exactly for many
`(budget, belief, num_units)` combinations including negative
(credit-direction) cases, zero violations, and correct tie-breaking
("defer" as the initial candidate, then feasible models in
`seq.model_names` order, strict `>` -- reproducing
`BellmanOracle._best_given_observation`'s own convention exactly). All
tests passed on the first fully-corrected implementation (one sign bug
in the negative-`num_units` branch of `unit_marginal_prices` was caught
and fixed before any test ran green).

## Preserved as a failed ablation

Tranche 6's scalar-price arm (`BeliefPricingController` + exact belief)
is retained unmodified as arm 3 in the new pilot -- the decisive
demonstration that scalarization, not belief quality or the environment,
was the bottleneck.

## Required arms (tranche 6.5 pilot)

(1) no pricing; (2) frozen pacing; (3) exact belief + scalar marginal
-price translation (tranche 6's failed ablation, unchanged); (4) exact
belief + Bellman action shadow charge; (5) true-parameter HMM belief +
shadow charge (oracle-quality structured ceiling, *not* a bar FabricPC
must clear -- see below); (6) ridge belief + shadow charge; (7) same
topology trained by backprop + shadow charge; (8) same topology trained
by FabricPC predictive coding + shadow charge; (9) shuffled FabricPC
belief + shadow charge (negative control).

## Gates

**Gate A-prime** (above): exact belief + shadow charge must exactly
reproduce the online optimum. If it fails, there is an implementation or
timing bug (belief update ordering, current-vs-next-step value lookup,
replenishment timing, realized-vs-expected consumption, feasibility, or
terminal-value handling) -- resolve it, do not tune parameters.

**Gate B / Gate C** (only after A-prime passes): reuses tranche 6's
already-built Part B/C training infrastructure (`fabricpc_belief_model.py`,
run once, not re-searched) unchanged. FabricPC is **not** required to
beat the true-parameter HMM (arm 5) -- that arm is a near-exact oracle
-quality structured ceiling in this synthetic environment, not a
baseline. FabricPC must instead beat frozen pacing, beat the shuffled
-belief control, beat or materially improve on plain ridge, be
non-inferior to the same topology trained by backprop, and capture a
positive, reportable fraction of the recoverable gap:

```text
recoverable_gap = regret(pacing) - regret(exact_belief_action_charge)
captured_fraction(model) = (regret(pacing) - regret(model)) / recoverable_gap
```

Report the exact captured fraction rather than forcing a large arbitrary
threshold on this first run.

## Prediction gates: boundary-sensitive belief quality

Beyond aggregate MSE/Brier/log-loss/regime-accuracy (tranche 6, unchanged),
report belief quality specifically at states where a belief error can
change the optimal action: precompute, per `(time, budget, observation)`
state, the belief interval over which each action is Bellman-optimal,
then report action-region classification accuracy, distance to the
nearest decision boundary, boundary-crossing error rate, and regret
attributable to belief error near a boundary. Aligns FabricPC's
prediction task with the economics without training it on actions or
prices directly.

## Runtime discipline

No new environment, predictor architecture, training seeds beyond the
existing three, or hyperparameter search. Gate A-prime is verified in
pure Python first; the existing bounded FabricPC/backprop training path
runs once, with all predictions and checkpoints cached, exactly as
tranche 6 already built it.

## Stop boundary

Complete locally: the discrete shadow-price curve and action opportunity
-cost controller; the belief-timing audit and its trace; the exact
-equivalence test against the online optimum (Gate A-prime); one bounded
execution of the already-built training path for arms 6-9; a regret and
boundary-sensitive belief report; all quality gates. Stop before:
production integration, changing `constraints.shadow_prices`, changing
`SwitchCertificate`, push, PR, or wiki/paper updates.
