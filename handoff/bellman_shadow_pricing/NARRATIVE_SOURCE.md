# Narrative source: Bellman-consistent discrete shadow-charge pricing

Content-layer only. No presentation-specific markup, styling, or
platform-specific formatting — suitable as raw input to
`notebook_compiler` or adaptation into Sketched/book source.

## Problem

A router repeatedly chooses among discrete actions that each consume a
different, integer-multiple amount of a shared, replenishing resource
under a finite horizon. Some actions are worth more when a hidden
condition — a "regime" — favors them, but that regime is never directly
observed, only inferred from noisy signals. The question: how should the
router price the opportunity cost of consuming resource on one action
instead of preserving it for a possibly better one later, and does a
learned predictor of the hidden regime improve on simpler alternatives?

## Why scalar prices fail

The natural first approach prices a marginal unit of the resource once
— a single scalar `lambda`, typically derived from a Bellman value
function's local derivative — and charges each action `lambda` times
its consumption. This works when consumption is effectively continuous
and the value function is smooth. It fails when actions consume several
discrete units at once and the value function has kinks: a resource
level that crosses a feasibility boundary (for example, the exact budget
needed to afford a valuable but resource-hungry action later) causes the
per-unit price to spike sharply at that boundary. An action that
consumes resource spanning several such regions is then priced by a
single rate sampled at one point, which can be very wrong for the actual
total opportunity cost.

Empirically, in a controlled environment with an exactly-computable
Bellman value function, exact-belief pricing under this scalar
translation did not outperform a much simpler non-Bellman heuristic
(fixed-rate pacing) — even though the underlying price was itself
computed exactly. The translation from price to action, not the price
computation, was the defect.

## Discrete shadow-charge construction

The fix prices the whole action by the actual drop in continuation value
it causes, not a rate times its size. Define the unit marginal price at
the `j`-th unit of resource consumed, holding belief fixed, as the exact
difference in continuation value between having that unit and not:

```text
lambda_unit[j] = V(B - (j-1)*delta, q) - V(B - j*delta, q)
```

An action consuming `k` units of net resource is charged the sum of
these unit prices, which telescopes to a closed form requiring only two
value-function evaluations:

```text
shadow_charge(a) = sum(lambda_unit[j] for j in 1..k)
                  = V(B, q) - V(B - k*delta, q)
```

Routing then selects the action maximizing immediate utility minus this
charge.

## Exact equivalence

Selecting by `utility(a) - shadow_charge(a)` is provably equivalent, up
to an action-independent constant, to selecting by the full Bellman
action value `Q(a) = utility(a) + continuation_value_after(a)`, because
`shadow_charge(a)` differs from `-continuation_value_after(a)` only by
the same term (`V(B, q)`) for every candidate action. This was not only
proven algebraically but verified computationally: given the exact
belief, the shadow-charge policy's chosen action was required, and
confirmed, to be bit-identical to the literal Bellman-optimal policy at
every decision point across many independent test sequences and
parameter settings, with zero exceptions.

## Belief-sensitive experiment

The first environment used to validate this mechanism turned out, after
the correction, to have no state where the router's belief about the
hidden regime ever changed which action was actually best — belief
affected the *value* used for bookkeeping but never the *decision*. A
second environment was constructed specifically so that a valuable
action's payoff itself depends on the hidden regime, not merely its
availability, making belief genuinely decision-relevant. Before running
any learned predictor, this environment was validated directly: the
Bellman-optimal action was enumerated across a dense grid of belief
values at every reachable state, confirming a majority of reachable
states have a genuine belief-dependent decision boundary, and that an
agent with the exact belief measurably outperforms agents with a fixed,
inverted, or shuffled belief.

## FabricPC comparison

Once belief was confirmed to matter, several ways of estimating it were
compared under the same pricing mechanism: the exact Bayesian filter,
a generic hidden-Markov-model filter given the true generative
parameters, an ordinary ridge regression trained on a declared window of
observable history, and a small predictive-coding network (FabricPC)
trained two ways from identical initial parameters — via its native
local predictive-coding learning rule, and via ordinary end-to-end
backpropagation on the same graph, as a required control.

## Results

The exact belief and the true-parameter filter recovered the full
economic advantage over a simple pacing heuristic, exactly as designed.
Ridge regression, trained on nothing but the declared window of past
observations, matched that result exactly — proving the underlying
prediction task is genuinely learnable by an ordinary model, not only by
an oracle. The predictive-coding network and its backpropagation control
converged to statistically indistinguishable results with each other,
but both were substantially less accurate than ridge at the belief
-prediction task, and this quality gap translated directly into worse
routing regret: the network captured only a small fraction of the
available economic opportunity that ridge captured in full.

## Limitations

The comparison used one small, fixed network topology and a bounded
training budget (a handful of seeds, tens of epochs, no architecture or
hyperparameter search), by deliberate design to keep the experiment
fast and bounded. Whether a larger or differently configured network
would close the measured gap with ridge was not investigated. Sample
sizes (a few dozen held-out sequences) limit the statistical confidence
of several pairwise comparisons.

## Implications

The clearest, best-supported result of this whole line of work is a
general pricing-mechanism correction: when a discrete action can span
several marginal-value regions, price it by its total, exact opportunity
cost, not by a locally-sampled rate. This generalizes beyond the
specific network architecture tested here. The specific finding that a
small predictive-coding network underperformed ordinary linear
regression on this bounded task is narrower and architecture-specific —
it says nothing about predictive coding as a learning principle in
general, only about this particular network, at this particular scale,
on this particular task, under this particular training budget.
