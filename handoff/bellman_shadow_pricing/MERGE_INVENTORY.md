# Production merge inventory

Classification of everything touched across tranches 1–7 of the
FabricPC×Compitum research program, verified directly against `main`
(`git diff main...HEAD`), not assumed. Nothing in this inventory has
been cherry-picked or merged as part of this submission — this is a
classification record only, per the preservation-mode stop boundary.

## Production candidate

- **`src/compitum/security.py` git-worktree resolution fix** —
  `_resolve_git_dir`/`_resolve_common_dir`, making `git_commit_short`
  work correctly when invoked from inside a git worktree (previously
  returned `None` or an incorrect value in that case). This is the
  *only* changed file across the entire program that (a) exists on
  `main` and (b) is used by production code paths outside this research
  effort. It has zero coupling to FabricPC, `regret_lab`, or any
  experiment-specific logic.

  **Exact cherry-pick commit:** `9b2d2fdc1202ab93128883bd10fe21c726e541af`
  ("Fix git_commit_short to resolve git worktrees, not just plain
  repos"). **Not yet cherry-picked or merged** — provided here for a
  future, separate decision.

- **Its focused tests** — the test file(s) covering
  `_resolve_git_dir`/`_resolve_common_dir` in the same commit, providing
  the coverage for the fix above. Travel with the same cherry-pick.

## Research-only

Additive, new packages that do not exist on `main` and are not imported
by any production entry point (`router.py`, `constraints.py`, `cli.py`
— verified directly, not assumed):

- `src/compitum/regret_lab/` — Bellman oracles, shadow-charge pricing,
  belief estimators, environments, metrics, and every pilot's supporting
  library code.
- `src/compitum/constraint_oracle/` — tranche 2's constraint-pressure
  oracle.
- `src/compitum/trajectory/` — tranche 1's trajectory-observation
  infrastructure (sensors, evidence schema, capability/receipt
  verification).
- FabricPC adapters (`experiments/fabricpc/tranche6/fabricpc_belief_model.py`
  and equivalents in earlier tranches) — JAX/FabricPC-dependent code,
  confined entirely to `experiments/fabricpc/*`, never imported by the
  `compitum` package itself.
- Bellman environments and oracles (`belief_regime.py`, `belief_regime_v2.py`,
  `belief_bellman.py`, `belief_bellman_v2.py`, and all of
  `belief_action_pricing*.py`/`belief_online_optimum*.py`).
- All reports, ADRs, and experiment runner scripts under
  `experiments/fabricpc/` and `docs/adr/0001`–`0009`.

None of the above is recommended for production activation. They remain
in place for provenance and reproducibility.

## Rejected for activation

Mechanisms explicitly tested and found not to warrant production use,
across the program's tranches:

- **Reactive dual pricing** (tranche 3's `DualController`/`ReactiveController`)
  — underperformed even a no-pricing baseline in its own tranche.
- **Scalar Bellman pricing** (tranche 6's linear `lambda * consumption`
  translation of the exact marginal price) — did not beat frozen pacing
  despite using an economically exact price; superseded by the discrete
  shadow-charge correction (tranche 6.5), which is itself research-only
  infrastructure, not a production change.
- **FabricPC residual pricing** (tranche 5's fixed/untrained FabricPC
  trajectory features through a bounded ridge residual) — inert: regret
  identical to frozen pacing alone.
- **Tranche-7 FabricPC belief controller, under the fixed topology** —
  did not clear the primary economics gate; captured only 15.4% of the
  measured recoverable regret gap against a ridge baseline that captured
  it in full.

None of these were ever wired into a production code path; this section
records the experimental verdict, not a rollback.
