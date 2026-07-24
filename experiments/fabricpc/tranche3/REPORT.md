# Tranche 3: dynamic-constraint regret pilot — report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`
(`a8de8cbafa5eb00b523f539c340ba81a146aa781`). FabricPC pin unchanged:
v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f` (external checkout
`C:\src\FabricPC`; not vendored, not patched). Nothing in this tranche
affects route selection, `main`, the `v0.2.0` tag, the frozen
`SwitchCertificate` schema, `constraints.shadow_prices`, or upstream
FabricPC.

## The governing correction this tranche answers

Tranche 2 tested whether FabricPC could predict a *present-slack-derived*
constraint-pressure label and found a negative result, then discovered why:
the frozen `ReflectiveConstraintSolver`'s shared feasibility test does not
ordinarily create a route-specific feasible set, so that oracle target
carried little temporal information for FabricPC to add to (see ADR 0002's
addendum). Per explicit user direction, tranche 3 stopped treating
classification accuracy/MAE as the activation criterion and built the
missing substrate instead: an experiment-owned environment
(`compitum.regret_lab`) where different model choices genuinely consume
different, cumulative, time-varying resources, and where the real target
is **held-out cumulative constrained regret** — not a static label.

## What was built

- **`src/compitum/regret_lab/environment.py`** — route-specific,
  time-varying resource consumption over 8 controlled scenarios
  (permanently slack, single-resource scarce period, demand burst,
  conserve-now-enables-better-future, premature-conservation-regret,
  multi-resource interaction, forecast error, delayed realization), grid
  -quantized for exact downstream arithmetic.
- **`src/compitum/regret_lab/hindsight.py`** — exact memoized-search
  hindsight constrained optimum (perfect foresight of realized
  consumption), with a documented greedy fallback and reported optimality
  gap if a sequence's state space were to exceed a bound (never silently
  approximate; not triggered by this pilot's sequence lengths).
- **`src/compitum/regret_lab/dual_controller.py`** — an experiment-only
  online primal-dual reference controller (`lambda_price`, clipped
  gradient update, `priced_utility`), established *before* any FabricPC
  involvement and kept structurally separate from `constraints.shadow_prices`.
- **`src/compitum/regret_lab/forecaster.py`** — `EWMAForecaster`, the
  non-FabricPC sequential baseline FabricPC must also beat, not just the
  no-predictor dual controller; only learns from the realized outcome of
  the model it actually chose, matching real online information access.
- **`src/compitum/regret_lab/simulator.py`** — runs one policy across a
  full sequence with a reservation-then-true-up ledger; delayed revelation
  can produce genuine violations, always reported separately from regret.
- **`src/compitum/regret_lab/channels.py`** — declared, dependency-free
  15-dimensional FabricPC input channel (remaining budget/quota, per
  -model expected consumption, base utilities, current dual price, steps
  remaining, this-step replenishment).
- **`src/compitum/regret_lab/metrics.py`** — `regret_metrics`,
  `paired_regret_deltas`, `bootstrap_ci`: violations are always reported
  separately from the regret scalar, never folded in as a penalty.
- **`experiments/fabricpc/tranche3/fabricpc_regret_observer.py`** —
  JAX-side single-step observer (source(15)→hidden(8)→latent(4), latent
  clamped to zero), reusing tranches 1-2's pinned-receipt/lightweight
  -history pattern; reuses the exact same raw schema as tranches 1-2 so
  `compitum.trajectory.evidence.build_evidence` validates it unchanged.
- **`experiments/fabricpc/tranche3/run_pilot.py`** — trains an EWMA
  forecaster and a per-(model, resource) ridge regression from FabricPC
  trajectory features on training sequences, freezes both, then runs five
  paired arms on held-out test sequences.
- 68 new tests in `tests/regret_lab/`, all `src/compitum` code (including
  `regret_lab`) 100% line+branch covered, mypy `--strict` clean.

## Pilot results (8 scenarios × 4 sequences × 8 steps; 16 train / 16 test sequences, sequence-level split)

**Core question — does FabricPC trajectory observation improve an
otherwise-valid online dual-pricing policy's held-out cumulative
constrained regret, beyond a no-predictor dual baseline and a simple
non-FabricPC (EWMA) sequential predictor?**

| arm | mean regret | median regret | total violations | depleted-budget events | deferrals |
| --- | --- | --- | --- | --- | --- |
| 1: static/frozen (no pricing) | 2.778 | ~0 | 0 | 10 | 8 |
| 2: online dual, no predictor | 2.982 | ~0 | 0 | 10 | 8 |
| 3: online dual + EWMA (non-FabricPC) | 3.671 | 1.438 | 0 | 0 | 6 |
| 4: online dual + FabricPC | 3.669 | 1.178 | 0 | 2 | 3 |
| 5: online dual + FabricPC, shuffled control | 3.427 | 3.059 | 1 | 3 | 2 |

**Pre-registered activation gate: arm 4 must show a paired
bootstrap-CI-significant reduction in mean regret vs BOTH arm 2 and arm 3,
must not increase violations vs either, and must be significantly better
than arm 5.**

- vs arm 2 (dual, no predictor): mean regret delta **+0.687**
  (arm 4 has *more* regret), 95% CI `[0.159, 1.282]` — entirely positive.
  Arm 4 is significantly **worse** than the plain no-predictor baseline.
- vs arm 3 (EWMA): mean delta −0.003, 95% CI `[-0.455, 0.497]` — straddles
  zero. Not distinguishable from the non-FabricPC sequential baseline.
- vs arm 5 (shuffled control): mean delta +0.242, 95% CI `[-1.744, 3.665]`
  — straddles zero widely. Arm 4 is **not** distinguishable from a version
  of itself fed a temporally-destroyed trajectory.
- violations not increased: true (the one criterion arm 4 satisfies).
- **Gate result: `passed: false`** on all three regret comparisons.

**This is a clean, informative negative result, not a repeat of tranche 2
under a different name.** The target this time is genuine cumulative
regret in an environment with real route-specific, time-varying resource
consumption — not a present-slack label — and FabricPC still shows no
detectable benefit, failing to beat either baseline and failing to clear
its own shuffled control.

**A second, independently interesting finding fell out of the same
numbers.** The plain online dual controller (arm 2) already has *higher*
mean regret than static/no-pricing at all (arm 1: 2.778 vs arm 2: 2.982) —
consistent with the ADR's own framing that this reactive baseline is not
claimed to be optimal. Adding either forecaster (EWMA, arm 3; FabricPC,
arm 4) makes mean regret higher still (3.67 vs 2.98), while simultaneously
driving depleted-budget events from 10 down to 0–3 and deferrals from 8
down to 2–6. In other words: **both forecast-correction mechanisms make
the policy more conservative (it runs out of budget less often and defers
less) at the cost of *higher* regret** — the `premature_conservation_regret`
failure mode the dataset was explicitly built to expose, showing up in
aggregate across scenarios, not only in the one scenario named for it.
This is exactly the kind of decision-consequence information a raw
classification/MAE metric (tranche 2's activation criterion) would have
hidden, and exactly why the user's correction to a regret-centered gate
was the right one methodologically, independent of how it happened to come
out for FabricPC specifically.

**Latency:** observe p50 413 ms, p95 499 ms, max 2.46 s, 384/384 real
FabricPC observations (128 training + 256 test-arm), 0 governed failures.

**Baseline integrity:** full worktree suite: 540 passed, 1 skipped (known
Windows subprocess skip), 1 pre-existing failure unrelated to this tranche
(`test_git_commit_short_resolves_real_repo_head` — a worktree-only
`.git`-file-vs-directory artifact, orthogonal to `regret_lab`, not
touched). `src/compitum` (including `regret_lab`) remains 100.00%
line+branch covered, mypy `--strict` clean, ruff clean.

## Honest methodology notes

- FabricPC's ridge models and EWMA are fit **offline on training
  sequences only**, then **frozen** before any test-sequence simulation —
  never touching test-sequence ground truth, matching tranche 2's own
  fit-on-train/predict-on-test discipline.
- The FabricPC ridge models were trained on channel-vector trajectories
  induced by an EWMA-driven reference rollout on training sequences, not
  on trajectories induced by their own eventual test-time deployment
  policy. This is a legitimate but imperfect offline-training setup (a
  common asymmetry in practice), flagged rather than hidden.
- "Trajectory" here means the PC graph's own settling dynamics within one
  inference call (as in tranches 1-2), not a multi-environment-step
  window. Whether letting FabricPC observe several past environment steps
  directly (rather than just the current step's channel vector) would
  perform differently is untested by this pilot.
- Ridge regression per (model, resource) pair, no hyperparameter search,
  matching tranches 1-2's own methodology.
- Controlled synthetic scenarios only — no realized routing labels in
  this tranche.

## Since the gate did not pass

No shadow simulation beyond this pilot is warranted (the pilot *is* the
shadow simulation the ADR called for: fully offline, non-route-affecting).
No tranche 4 activation mechanism (calibrated deferral trigger,
price-adjusted utility, persistent online duals, route-affecting
activation) is introduced. `constraints.shadow_prices`,
`SwitchCertificate`, routing behavior, and the `v0.2.0` tag remain
untouched.

## Open items and smallest defensible next step

Unresolved: whether FabricPC's null result here reflects a genuine absence
of exploitable structure in this controlled environment's resource
dynamics, or an artifact of (a) training the ridge models on an
EWMA-induced rather than self-induced trajectory distribution, (b)
single-step rather than windowed FabricPC input, or (c) the toy PC graph
architecture. The **more consequential and separable finding** — that
naive reactive dual pricing, and both forecast-correction mechanisms
tested, trade higher regret for fewer depleted-budget events and deferrals
— does not depend on FabricPC at all and is worth investigating on its own
before any further FabricPC work: a dual controller whose pricing is this
easy to overreact with is not yet a baseline FabricPC (or anything else)
should be asked to improve.

Smallest defensible next step, still observation-only and not assuming
FabricPC helps: tune or replace the reactive dual controller (arm 2) itself
— e.g. a pacing-style target rate with slower/bounded lambda adjustment,
or a hysteresis band — and check whether a *better-tuned, still-non
-learned* dual baseline closes some of the gap to the hindsight optimum
before reintroducing any learned forecaster on top of it. Only after that
baseline is itself defensible does it make sense to ask whether FabricPC
(windowed or otherwise) can improve it further.
