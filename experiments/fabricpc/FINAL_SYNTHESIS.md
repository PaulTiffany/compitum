# FabricPC × Compitum: final synthesis (tranches 1–7)

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`.
Every tranche below is observation-only: none of this program has ever
changed `constraints.shadow_prices`, `SwitchCertificate`, production
route selection, or `main`. This document closes the program per the
tranche 7 brief's stop boundary ("a final synthesis of tranches 1-7 and
a merge-candidate inventory"). No push, PR, tag, or wiki/paper update
follows from it.

## What the program was asking

Across seven tranches (eight counting 6.5 separately), one question
recurs in increasingly rigorous form: **can FabricPC (a predictive
-coding network) improve Compitum's constrained routing decisions, and
if so, through which mechanism?** Each tranche corrected a specific flaw
in how the previous one tested that question, rather than simply
retrying with different data.

## Tranche-by-tranche results

| # | What was tested | Result |
| --- | --- | --- |
| 1 | Frozen/inference-only FabricPC trajectory features → routing | **Negative.** Generic trajectory features added no held-out information beyond frozen certificate features. |
| 2 | Constraint-pressure oracle vs. production `shadow_prices` | **Negative**, but found and fixed a real `calibrate_threshold` bug in the process. |
| 3 | Dynamic-constraint regret substrate (`compitum.regret_lab` born here) | **Negative**, but found `ReflectiveConstraintSolver`'s shared feasibility test collapses to two branches, explaining tranche 2's null result. |
| 4 / 4.5 / 4.6 | Pricing-controller repair + scarcity phase-diagram study | **Conditional positive** (a narrow pacing benefit, found only after widening the parameter grid); real findings along the way (a calibration mismatch, an extreme-payoff cliff with three distinct causes). |
| 5 | FabricPC residual shadow pricing (fixed/untrained features + ridge residual) | **Negative: inert, not harmful.** Byte-identical regret to frozen pacing alone. |
| 6 | Trained belief-state FabricPC + exact Bellman-derived shadow price | **Stopped at Gate A.** Even the *exact* latent belief, priced through the exact Bellman continuation-value oracle, did not beat frozen pacing — diagnosed as a price-to-action *scalarization* bottleneck, not a prediction-quality one. FabricPC training was correctly never exercised. |
| 6.5 | Bellman-consistent discrete shadow-price curve (corrects tranche 6's scalarization) | **Decisive positive for the economic mechanism.** Gate A-prime proved the corrected translation exactly reproduces the true online optimum; exact-belief pricing then recovered the *entire* economic gap over pacing (regret 0.000 vs. 1.829). But every belief source, including a shuffled FabricPC control, tied at zero regret — that environment had no state where belief actually changed the optimal action. |
| 7 | Belief-sensitive environment (Gate 0 passed on a second development-grid pass) + ten-arm pilot | **Economics and learnability both confirmed real; FabricPC specifically underperforms.** Exact belief clearly beat belief-blind controls this time. Ridge (plain linear regression) achieved belief MSE of 5.1e-7 and exactly zero regret, tied with the oracle. FabricPC (both training rules, tied exactly with each other) had ~600x worse belief MSE and captured only 15.4% of the recoverable gap — primary economics gate FAILED. |

## The throughline

**The single clearest, most decisive positive result in the entire
program is tranche 6.5's shadow-charge correction — an economics/pricing
-mechanism fix, not a FabricPC result.** Whenever the pilot included a
strong, cheap comparator (frozen pacing, a true-parameter HMM filter, or
plain ridge regression), that comparator either tied with or beat
FabricPC. In no tranche, under any environment or training regime, did
FabricPC demonstrate a regret improvement over the strongest available
non-FabricPC baseline that was both (a) statistically distinguishable
from zero and (b) not also achieved by a much simpler model.

Tranche 7 is the most informative negative result for FabricPC
specifically, because it is the first environment where belief
*genuinely* mattered and a cheap baseline (ridge) *proved* the task was
worth solving well — isolating FabricPC's shortfall to its own learned
representation, at the declared bounded scale (one small topology, ≤30
epochs, 3 seeds, no architecture or hyperparameter search), rather than
to "no headroom exists" or "the task isn't learnable."

## What would change this conclusion

This program deliberately never ran an architecture search or scaled
FabricPC's topology/training budget up, per every tranche's runtime
-discipline mandate. Tranche 7's report states this explicitly: whether
a larger or differently-configured FabricPC network would close the
~600x belief-MSE gap with ridge is an open question this program did not
chase. Any future work should treat that as the concrete next question
-- not "does FabricPC help Compitum" in the abstract, but "does a larger
FabricPC topology, trained longer, close a *specific, already-measured*
600x quality gap against a specific, already-built ridge baseline, in an
environment already proven to make belief quality matter."

## Merge-candidate inventory

Checked directly against `main` (`git diff main...HEAD`), not assumed:

- **`src/compitum/security.py` — genuine, ready merge candidate.**
  `git_commit_short` previously returned `None` (or a wrong value) when
  run from inside any git worktree, not just this one; `_resolve_git_dir`/
  `_resolve_common_dir` (commit `9b2d2fd`) fix this generally. This is
  the *only* change in the whole program to a file that exists on `main`
  and is used by production code paths outside this research effort. It
  has no coupling to FabricPC, regret_lab, or any experiment-specific
  logic, and is worth cherry-picking on its own regardless of whether
  anything else here ever merges.
- **`src/compitum/regret_lab/`, `src/compitum/constraint_oracle/`,
  `src/compitum/trajectory/` — not merge candidates.** All three are new
  packages that do not exist on `main` at all, and none is imported by
  `router.py`, `constraints.py`, `cli.py`, or any other production
  routing path (verified directly, not assumed). They are additive,
  self-contained, and dependency-free of FabricPC/JAX at the `compitum`
  package level (FabricPC/JAX imports are confined to
  `experiments/fabricpc/*`, which never ships). Nothing found in seven
  tranches justifies promoting any of this into a production shadow
  -pricing or routing path.
- **The shadow-charge pricing *insight* (tranche 6.5) — a lesson, not a
  merge candidate.** "Price a discrete, lumpy action by its exact total
  opportunity cost (a value-function difference), not a linear
  per-unit-rate times consumption" is a real, portable economic
  -modeling correction. If Compitum's production `constraints.shadow_prices`
  mechanism is ever redesigned to price genuinely discrete, multi-unit
  actions, this insight is directly relevant groundwork — but the
  current production mechanism is a different, simpler kind of pricing,
  and nothing here demonstrates that redesigning it would be worthwhile
  today.
- **Everything else** (belief environments, Bellman oracles, FabricPC
  training scripts, all pilot reports) is experiment-owned research
  infrastructure with no production dependency, left in place for
  provenance and reproducibility, not intended for further use unless a
  future tranche explicitly revives it.

## Stop boundary

This closes the FabricPC×Compitum research program as authorized.
Complete: seven correction cycles, each addressing a real, specific flaw
found in the previous one; one decisive economic-mechanism result
(tranche 6.5); one clear, well-evidenced FabricPC-specific negative
result at bounded scale (tranche 7); a verified merge-candidate
inventory. Not done, and not authorized by any tranche's stop boundary:
production integration, changes to stable schemas or `shadow_prices`,
push, PR, tag, release, or wiki/paper updates.
