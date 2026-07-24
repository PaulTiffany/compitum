# Tranche 2: constraint-pressure oracle + FabricPC trajectory pilot — report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`
(`a8de8cbafa5eb00b523f539c340ba81a146aa781`). FabricPC pin unchanged from
tranche 1: v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f` (external
checkout `C:\src\FabricPC`; not vendored, not patched). Nothing in this
tranche affects route selection, `main`, the `v0.2.0` tag, the frozen
`SwitchCertificate` schema, `constraints.shadow_prices`, or upstream FabricPC.

## Roadmap correction this tranche answers

Tranche 1 tested the broad hypothesis ("do generic FabricPC trajectory
summaries add routing-prediction information?") and got a negative result.
That closes the broad question but not the narrower one that motivated the
whole integration: **can FabricPC trajectories predict which constraints are
approaching activation, and how much relaxation would change the feasible
optimum, better than Compitum's own static state and its finite-difference
`shadow_prices` stub?** Tranche 2 builds an independent, Compitum-owned
oracle for that exact target first, then tests FabricPC against it — never
using the existing `shadow_prices` diagnostic as ground truth, since it is
itself only a fixed-1e-5 finite-difference probe, not a validated authority.

## What was built

- **`src/compitum/constraint_oracle/static.py`** — exact, closed-form oracle
  for `critical_relaxation`, `marginal_utility_improvement`,
  `best_suppressed_competitor`, and a reason taxonomy
  (`not_currently_binding` / `blocked_by_other_constraint` /
  `capability_blocked_only` / `recovers_feasibility`). This exploits a
  structural fact discovered in the frozen `ReflectiveConstraintSolver`
  (`src/compitum/constraints.py`, untouched): its linear feasibility check
  uses one shared `xB` and a single `np.all(...)` over every constraint row,
  so a case is either linear-feasible for every model at once, or
  linear-infeasible for every model at once (unconditional
  `infeasible_fallback`) — there is no "feasible route already exists, but a
  different constraint is separately violated" branch. This makes the
  critical relaxation for the sole violated constraint an exact quantity
  (`Δb_i* = max(0, -slack_i - ε)`), not something requiring search.
  Cross-validated against `validation.numeric_critical_relaxation`, a
  bisection search against the real, unmodified solver.
- **`src/compitum/constraint_oracle/horizon.py`** — sequence-level
  `binding_within_horizon` / `time_to_binding` / `realized_future_slack`
  targets, cross-validated against the dataset generator's own known-answer
  labels.
- **`src/compitum/constraint_oracle/dataset.py`** — 9-scenario controlled
  generator (permanently slack, single/multi-constraint ramps with and
  without capability blocking, discontinuous ties, unbinding recovery,
  permanent infeasibility), independently varying slack, utility gaps, and
  suppressed-competitor identity per the pre-registration.
- **`src/compitum/constraint_oracle/channels.py`** — declared, dependency
  -free 17-dimensional FabricPC input channel (normalized slack,
  feasibility mask, per-model utilities, utility-gap/entropy, the frozen
  `LyapunovController`'s drift state, and violated-set/selection-change
  transition signals). Documented a genuine mathematical degeneracy in ADR
  0002: an all-zero channel vector forces the observed network's prediction
  error (and thus energy) to exactly 0.0 for any weights/seed — a trap to
  avoid using as a "no signal" probe.
- **`src/compitum/constraint_oracle/experiment.py`** — dependency-free
  two-part model (P(consequential) classifier + magnitude-only regressor,
  fit separately so a large all-zero class can't manufacture a misleadingly
  good blended error), metrics, and feature extraction from materialized
  trajectory evidence. Includes `calibrate_threshold` (added this session,
  see "Methodological bug found and fixed" below).
- **`experiments/fabricpc/tranche2/fabricpc_channel_observer.py`** — JAX-side
  observer: source(17)→hidden(8)→latent(4) PC graph, latent clamped to zero
  to force real settling, deterministic per-sequence seeding.
- **`experiments/fabricpc/tranche2/run_pilot.py`** — the bounded,
  observation-only pilot orchestrating all of the above.
- 84 new tests across `tests/constraint_oracle/` (all oracle logic,
  including the two-part model and metrics, is 100% line+branch covered and
  mypy `--strict` clean without ever importing FabricPC/JAX; the JAX-side
  orchestration script is verified empirically instead, matching tranche 1's
  established split).

## Methodological bug found and fixed

The first full pilot run produced byte-identical `accuracy=0.9201`,
`recall=0.0`, `precision=NaN` across all four comparison arms **and** the
shadow-price reference. Rather than accept this as a uniform negative
result, I investigated directly (per this project's standing practice of
never trusting a suspiciously uniform result without a probe): the test
positive ("consequential") rate is 7.99% (46/576); the raw ridge-regression
classifier score genuinely separates the classes (mean 0.204 for true
positives vs 0.096 for true negatives, a real ~2x gap) but its maximum
observed value (0.416) never reaches the hardcoded default
`threshold=0.5` — a plain ridge score regresses toward the mean and, at this
base rate, never crosses an arbitrary 0.5 cutoff regardless of how much
signal it carries. This was a real bug in the pilot's own evaluation code,
not evidence about FabricPC. Fix: `calibrate_threshold(p_train, y_train)`
picks a threshold from **training predictions only** (quantile-matched to
the training positive rate), never touching test labels — leak-free. Wired
into every classification call in `run_pilot.py` (the main
consequential/magnitude metrics and the horizon binding-within-window
metrics, which reuse the same classifier scores against a different label
and are recalibrated separately for that reason). After the fix, recall
ranges 0.31–0.46 and precision ~0.04–0.80 depending on arm/constraint — the
metrics below are genuinely informative, not threshold artifacts.

## Pilot results (9 scenarios x 4 sequences x 8 steps = 288 observations, 1152 case-constraint rows, sequence-level train/test split, 2 train / 2 test sequences per scenario)

**Core question — does a FabricPC trajectory improve held-out estimation of
`critical_relaxation` (magnitude) and whether a constraint is consequential
(classification), beyond static Compitum state and terminal FabricPC state
alone?**

| arm | classification accuracy | recall | regression MAE (n=46) |
| --- | --- | --- | --- |
| 1: static (17-channel state + one-hot constraint index) | 0.8247 | 0.413 | 0.1791 |
| 2: + FabricPC terminal state | 0.8212 | 0.457 | 0.1717 |
| 3: + FabricPC trajectory summary | 0.8212 | 0.457 | **0.1701** |
| 4: + shuffled/temporally-destroyed trajectory (control) | **0.8281** | 0.413 | 0.1673 |
| shadow_prices reference (descriptive only) | 0.9201 | 0.0 | — |

**Pre-registered activation gate: trajectory arm must beat BOTH static
baseline AND shuffled control on held-out accuracy AND MAE.**

- vs static baseline: accuracy 0.8212 is *not* > 0.8247 → **fails**.
- vs shuffled control: accuracy 0.8212 is *not* > 0.8281, and MAE 0.1701 is
  *not* < 0.1673 → **fails**.
- **Gate result: `passed: false`.**

Ranking accuracy (argmax over predicted magnitude, per case) is identical
(0.4348) across all four arms — the arms agree on which constraint is most
consequential whenever there's a real ranking question, regardless of which
extra features they carry. Combined with the accuracy/MAE gate failure, this
is a genuine negative result on the narrower, better-targeted hypothesis,
not a repeat of tranche 1's broad-question null under a different name: this
time the target is the exact oracle (not a proxy), the model is properly
calibrated, and the trajectory arm still does not clear the shuffled-control
bar it must clear to be trusted.

The shadow-price reference's 0.9201 accuracy is the base-rate accuracy
(recall 0.0) — it never flags a constraint as consequential in this dataset,
consistent with it being a fixed 1e-5 finite-difference probe rather than a
validated predictor; it is reported descriptively only, per the
pre-registration, and was never fit into any model.

**Latency:** observe p50 620 ms, p95 794 ms, max 2.77 s, 288/288 real
FabricPC observations, 0 governed failures.

**Baseline integrity:** full worktree suite: 481 passed, 1 skipped (known
Windows subprocess skip), 1 pre-existing failure unrelated to this tranche
(`test_git_commit_short_resolves_real_repo_head` — this worktree's `.git` is
a worktree-pointer file, not a directory, which `git_commit_short()`'s
default repo-root resolution doesn't handle; orthogonal to
`constraint_oracle`, not touched, `src/compitum/security.py` is out of this
tranche's scope). `src/compitum` (all packages including
`constraint_oracle` and `trajectory`) remains 100.00% line+branch covered,
mypy `--strict` clean, ruff clean.

## Since the gate did not pass

Per the pre-registration, the offline shadow simulation is gated on a
positive result and was not run. No mechanism from the tranche 3 activation
sequence (observe-only predicted pressure, calibrated deferral trigger,
price-adjusted shadow utility, online primal-dual variables, route-affecting
activation) is introduced. `constraints.shadow_prices` and the finite
-difference implementation remain exactly as in `v0.2.0`.

## Open items and smallest defensible next step

Unresolved: whether this negative result reflects a genuine absence of
useful trajectory structure for this problem, or an artifact of the toy PC
graph (source→hidden→latent, 17→8→4, fixed eta/steps) and the controlled
dataset's scenario design (synthetic slack ramps rather than realized
routing dynamics). The two-part model is a plain ridge fit with no
hyperparameter search, matching tranche 1's methodology but not ruling out
that a better-tuned model would find signal a linear one cannot.

Smallest defensible next step, still observation-only and still not
assuming FabricPC helps: before spending more effort on FabricPC graph
architecture, check whether a **non-FabricPC sequential model** (e.g. a
small recurrent or windowed model over the same static channel history,
with no PC-graph observation at all) beats the static single-step baseline
on this same oracle target. If a simple sequential baseline over Compitum's
own state already can't beat the static arm, that would suggest the
bottleneck is the amount of exploitable temporal structure in this
controlled dataset design generally, not FabricPC specifically — informing
whether tranche 3 (were it ever authorized) should focus on richer PC-graph
inputs or on a fundamentally different dataset (realized routing dynamics,
per the deferred track 2 design) before revisiting trajectory features at
all.
