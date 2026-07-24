# ADR 0007: trained belief-state FabricPC with Bellman-derived shadow prices (tranche 6)

Status: accepted, observation-only. Supersedes nothing in ADR 0001-0006;
corrects a foundational modeling issue in how tranche 5 tested FabricPC.

## Outcome (see experiments/fabricpc/tranche6/REPORT.md for full detail)

**Stopped at Gate A.** Exact-belief Bellman pricing (arm 3) did not beat
frozen pacing (arm 2) on held-out regret against the exact online
optimum: mean paired delta `+0.114` (arm 3 worse), 95% CI
`[-0.229, +0.457]` -- straddles zero and leans the wrong way. Robust
across five independent test-set seeds (deltas ranged `-0.229` to
`+0.457`, never significantly negative). Both priced arms crushed no
-pricing (mean regret `1.83`-`1.94` vs `2.63`), confirming pricing in
general matters here; the null result is specifically that the
mathematically exact marginal-value price does not outperform pacing's
simple budget/time-ratio heuristic. Part B (ridge/HMM/FabricPC training)
and arms 4-8 were never run, per the ADR's own "stop on Gate A failure"
instruction -- this is the responsible, cost-disciplined outcome, not an
oversight.

This is a genuine finding about the greedy `price_utilities` decision
rule's interaction with the Bellman price's own shape (verified sharply
spiking at discrete feasibility boundaries, see Part A below), not a
flaw in the price calculation: the exact-belief arm switched routes far
more often than pacing (73% vs 48% of steps) without a matching regret
improvement, and rejected the immediate-best-utility option more rarely
than pacing did (139 vs 155 rejections) while still not converting that
into lower regret -- consistent with tranche 4's established finding
that a technically-correct price signal can still underperform a
smoother heuristic once translated into action through linear greedy
routing. Per the ADR's own interpretive framework: **exact belief does
not help -> economics/environment (more precisely, the price-to-action
translation) bottleneck**, not a prediction-quality bottleneck -- so
training FabricPC would not have been informative here regardless of
how well it learned to predict belief.

## Governing correction

Tranche 5 tested:

```text
fixed random FabricPC transform + trajectory summary + ridge residual head
```

It did **not** test:

```text
trained predictive-coding model -> inferred future-scarcity state
                                 -> economically derived shadow price
```

Tranche 5's FabricPC graph was randomly initialized from the sequence ID
and *observed* during inference; its predictive-coding weights were never
trained on the environment's temporal dynamics across examples. The
supported conclusion is narrow: "the fixed-feature residual construction
did not improve regret" -- not "FabricPC cannot improve shadow prices."
That negative result is preserved untouched (`experiments/fabricpc/tranche5/`,
ADR 0006) and not generalized beyond what it actually tested.

## Governing principle for this tranche

> FabricPC should predict the future state of scarcity. The Bellman value
> function should determine what scarce resources are worth. Do not ask a
> predictive model to invent economics that can be calculated exactly.

FabricPC never predicts `lambda`, a pacing residual, an oracle-compatible
action boundary, or the production `shadow_prices` field. It estimates the
posterior belief over a hidden scarcity/opportunity regime; a deterministic
economic layer (an exact, precomputed Bellman table) converts that belief
into a marginal resource value:

```text
observed routing/resource history
    -> trained FabricPC belief
    -> exact finite-horizon continuation value (precomputed)
    -> marginal value of remaining resource
    -> shadow-priced routing decision
```

## Part A: the economically correct price oracle (built)

`src/compitum/regret_lab/belief_regime.py` + `belief_bellman.py`. Per the
"cost and runtime discipline" directive, deliberately narrow: one
resource, one fixed horizon (`STEPS=10`), exactly two hidden regimes
(NORMAL, HIGH), no scenario sweep.

**Generative model:** a 2-state Markov chain governs, each step, the
probability that a high-value "opportunity" action becomes available
(`P_OPPORTUNITY[NORMAL]=0.05`, `P_OPPORTUNITY[HIGH]=0.35`;
`TRANSITION[NORMAL->HIGH]=0.2`, `TRANSITION[HIGH->HIGH]=0.6`). The policy
never observes the regime directly -- only whether the opportunity was
available this step, which is the sole stochastic observation used for
exact Bayesian filtering (`filtered_belief`/`predict_belief`). Three
declared models per step (`conserve`, `spend`, `opportunity`), reusing
`DynamicCase`/`DynamicSequence` unchanged, exactly like
`scarcity_scenarios.py`: `opportunity` is priced to be unconditionally
infeasible on steps it did not become available. Ground-truth regimes and
exact belief trajectories are returned **separately** from the sequence
(`generate_belief_sequence` returns `(sequence, true_regimes, belief_priors,
belief_posteriors)`) -- never placed in any field a policy can read.

**Exact online continuation value**, per the brief's formula, computed via
memoized recursion (`BellmanOracle.value`) rather than a lossy grid:
because actions never affect the hidden regime or its observation, the
belief trajectory is independent of the policy's own choices, and because
the environment starts from one fixed initial belief, the reachable belief
set is a finite tree of depth `steps` -- no approximation beyond ordinary
floating-point exactness (all consumption/replenishment/budget values are
exact multiples of `GRID_UNIT=0.5`, the gcd of every consumption and
replenishment amount in this environment, distinct from
`scarcity_scenarios.py`'s `GRID_UNIT=0.25`).

```text
V(r, B, q) = sum_{o in {0,1}} P(o|q) * max_a [ u(a) + V(r-1, B-c(a)+replen, q'(o)) ]
V(0, B, q) = 0
lambda*(r, B, q) = [V(r, B, q) - V(r, B-delta, q)] / delta,  delta = GRID_UNIT
```

Hand-verified (`tests/regret_lab/test_belief_bellman.py`): price is exactly
zero when budget is already abundant relative to the horizon, spikes
sharply at the exact discrete scarcity boundary (e.g. `budget=4.0` vs
`3.5`, where `opportunity` costs exactly `4.0`), and increases
monotonically with belief in the HIGH regime -- confirming the price
responds to genuine, not spurious, scarcity.

**Online comparator, not hindsight leakage:** `BellmanOracle.best_action_given_observation`
is the principal scientific comparator -- the exact Bayes-optimal policy
operating with the same observations available to every tested arm
(current belief, current budget, this step's already-revealed opportunity
signal), never future realizations. The perfect-foresight hindsight
optimizer (tranches 3-5's oracle) is retained only as a separate,
unattainable upper bound; both regrets are reported, never conflated.

## Part B: FabricPC's genuine predictive task (built, never invoked)

Built and empirically verified end-to-end against the pinned FabricPC
checkout (`experiments/fabricpc/tranche6/fabricpc_belief_model.py`) --
training genuinely converges, and a live per-step estimator wired
through `simulate_policy` runs successfully. Never invoked by the final
pilot run because Gate A stopped the pilot first (see Outcome above);
kept and committed as verified, reusable infrastructure, not dead code,
in case a future tranche revisits the price-to-action translation layer
rather than the belief-estimation layer.

Confirmed feasible via direct investigation of the pinned FabricPC 0.3.2
checkout: it exposes real parameter-training machinery, not just
single-shot inference --

- `fabricpc.training.train_pcn` -- genuine predictive-coding weight
  learning (local per-node gradient of each node's own settled energy,
  not global backprop) via `optax`, across a real `train_loader` and
  `num_epochs`.
- `fabricpc.training.train_backprop` -- ordinary end-to-end backprop on
  the *same* graph topology, provided as the required control (needs
  `FeedforwardStateInit`, incompatible with cyclic/lateral topologies --
  this tranche's declared topology is a plain feedforward
  source->hidden->latent graph, so this is not a constraint here).

FabricPC's task (preferred for the first bounded pilot, per the brief, for
its exact ground truth and clean interpretation): **infer the current
hidden-regime belief** (a scalar in `[0, 1]`, `P(regime=HIGH)`) from a
declared window of observable history (previous route choices, realized
consumption revealed so far, utility observations, replenishment
observations, remaining resource, time remaining, opportunity indicators
observed so far). As actually built, the target is the *next step's*
belief prior (`belief_priors[t+1]`, entering the recursion the same way
`BellmanOracle.marginal_price` consumes it) rather than this step's own
posterior, so the trained predictor's output is directly usable by the
Bellman table with no extra transformation -- mathematically equivalent
information (`predict_belief(posterior_t) == belief_priors[t+1]`), never
a Bellman price or hindsight choice supervised directly. Regression via
`SigmoidActivation()` (bounding the output to `[0, 1]`) + `GaussianEnergy()`
(`Linear`'s own default energy) on the output node.

Training is real: both `train_pcn` and `train_backprop` run on the same
initialized `params`/`structure`, with topology, initialization seeds,
optimizer, epoch count, and train/validation split all recorded, and a
final checkpoint (plain-pytree pickle, since `GraphParams`/`NodeParams`
are ordinary registered JAX pytrees -- no bespoke serialization needed)
hashed for provenance.

## Part C: deriving prices from beliefs (built, never invoked)

Per step: the predictor estimates belief -> the precomputed `BellmanOracle`
receives `(remaining_steps, budget, estimated_belief)` -> returns
continuation value and marginal price -> routing uses
`priced_utility[m] = immediate_utility[m] - lambda_star * expected_consumption[m]`,
the same `price_utilities` function every prior tranche has used unchanged.
FabricPC never alters the Bellman table and never bypasses feasibility. A
declared non-learned belief estimate (the previous step's belief, or the
prior mean if none) is the deterministic fallback on any predictor failure
or refusal -- never an arbitrary zero price.

## Required arms

Paired held-out stochastic sequences: (1) no pricing; (2) frozen pacing
(tranche 4/4.6, unmodified, retained for continuity though it has no
belief awareness); (3) exact latent belief + Bellman price (oracle-belief
upper bound); (4) simple Bayesian/HMM filter + Bellman price (strongest
simple structured baseline -- in this environment the exact filter *is*
the Bayesian/HMM baseline, so this arm additionally serves as a
sanity-check that arm 3 and a from-scratch-implemented filter agree); (5)
ordinary sequential neural predictor + Bellman price; (6) same graph
trained by backprop + Bellman price; (7) same graph trained through
FabricPC predictive coding + Bellman price; (8) FabricPC belief shuffled
or sequence-mismatched + Bellman price (negative control).

## Decisive gates, evaluated in order

**Gate A (economic opportunity):** exact-belief Bellman pricing (arm 3)
must beat frozen pacing on held-out regret without increasing violations.
If it fails, the environment does not provide enough value for improved
scarcity prediction to help at all -- stop.

**Gate B (learnable state):** at least one learned predictor (arms 5-7)
must beat the simple Bayesian/HMM filter (arm 4) on held-out belief/forecast
quality. If none does, the latent process is already captured by the
simple filter or is not learnable from the declared history -- stop.

**Gate C (FabricPC value):** FabricPC predictive-coding training (arm 7)
passes only if it lowers regret vs. frozen pacing, lowers regret vs. the
Bayesian/HMM filter (arm 4), is non-inferior to the same graph trained by
backprop (arm 6), beats shuffled beliefs (arm 8), adds no violations,
captures a meaningful fraction of the pacing-to-exact-belief gap, and
remains useful after latency. Report:

```text
FabricPC gain over pacing / exact-belief Bellman gain over pacing
```

as the primary "fraction of recoverable opportunity captured" statistic --
more informative than a bare confidence-interval check.

## Metrics

Regret remains primary: vs. the exact information-matched online optimum
(principal comparator) and separately vs. perfect-foresight hindsight;
violations and magnitude; utility per resource unit; captured
opportunities; depletion/premature-conservation regret; terminal unused
resource; route disagreement; latency. Prediction diagnostics: belief
calibration, log loss, Brier score, regime accuracy, forecast error.
Price diagnostics: marginal-value error, sign error, ordering accuracy,
boundary error, monotonicity in budget, Bellman consistency. Prediction
or price accuracy never substitutes for the regret gate.

## Runtime discipline

One resource, two regimes, one fixed horizon, one declared FabricPC
topology, one backprop control topology, no architecture search, a tiny
preregistered development grid, at most three training seeds, one held
-out evaluation set. Every unique history is observed by FabricPC once and
its belief reused across every applicable metric -- never repeated merely
because multiple metrics consume the same belief. An explicit compute
ceiling is set before execution; if the bounded design would exceed it,
sequence count or horizon is reduced rather than opening an unbounded
optimization loop.

## Preserve prior tranches

Tranche 5's supported conclusion ("fixed, untrained FabricPC trajectory
features used through a bounded ridge residual did not improve regret
beyond frozen pacing") is not rewritten and is not generalized to trained
FabricPC predictive modeling. Tranches 1-4.6's results likewise stand
unchanged.

## Stop boundary

Complete locally: the hidden-regime environment and exact Bellman oracle
(done); the Bayesian/HMM comparator, simple sequential predictor, and
trained FabricPC belief model (PC and backprop variants); the eight-arm
paired regret experiment with gates A/B/C evaluated in order (stopping
early on any gate failure, per the brief); exact checkpoint/artifact
hashes; an honest report. Stop before: production integration, changing
`constraints.shadow_prices`, changing `SwitchCertificate`, live routing
changes, push, PR, or wiki/paper updates.
