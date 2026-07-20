# Mutation Hardening Status

**Update, day 2**: `router.py` (the last file) finished -- 137/137, 105 killed, 32 survived. All
16 runnable shards are now complete (`pgd.py` remains the one Windows-only mutmut crash). 3 real
gaps found and fixed in `router.py` (`7c3200f`): the `update_stride <= 0` clamp, the `srmf`/
`controller` legacy-alias identity, and the disabled-controller `drift_status` exact values. Also
confirmed one genuine, unfixable equivalent mutant: `abs(-comps[...]["distance"])` -- `abs(-x) ==
abs(x)` always, so no test can ever kill a mutation removing that unary minus (appears twice, in
`route()` and `batch_route()`, likely accounting for 2 of the 32 survivors by construction).

These findings are now also catalogued in `docs/Invariants.md`'s new "Mutation-Hardening Coverage"
section, following the doc's existing terse per-invariant convention, so they're visible as
verified CI invariants rather than only living in this working-notes file.

Real, local mutmut runs across the full 17-file shard matrix (`.github/workflows/mutation.yml`'s
`mutmut-shard` matrix), using the fixed coverage-guidance logic (commit `0cd02a3`) and the
`mutation_ci` Hypothesis profile (matches CI exactly). This is raw data for continuing the sweep
efficiently in a future session/run -- not a finished report.

**Archived caches and full run logs are persisted at `artifacts/mutation_cache/`** (gitignored via
`/artifacts/`, so they survive on disk across sessions without polluting git). Each file has
`<name>.mutmut-cache` (real mutmut SQLite result cache -- `mutmut show <id>` works against it if
copied to `.mutmut-cache` in repo root) and `<name>_run.log` (the full captured run output).

## Per-file results

All "Killed"/"Survived" counts below are the **real, as-run numbers from the original mutmut
sweep**, before any of the listed fixes. None of the fixes have been re-verified against mutmut yet
(that requires restoring the shared `.mutmut-cache`, which was busy with the next file in the batch
each time a fix was written) -- that re-verification is the single most important next step, not
a re-run of the whole file.

| File | Total | Killed (as-run) | Survived (as-run) | Suspicious | Fixes committed (not yet re-verified) |
|---|---:|---:|---:|---:|---|
| `embedding_pgd.py` | 1 | 1 | 0 | 0 | **Green**, no fix needed |
| `utils.py` | 9 | 9 | 0 | 0 | **Green**, no fix needed |
| `models.py` | 1 | 1 | 0 | 0 | **Green**, no fix needed |
| `capabilities.py` | 9 | 7 | 2 | 0 | 1 fix (`f9a98ca`, default-value gap) targets 1 of 2 survivors |
| `effort_qp.py` | 25 | 19 | 6 | 0 | 2 fixes (`689b6ea`, `9bfd747`) target 2 of 6 survivors |
| `boundary.py` | 44 | 35 | 9 | 0 | 1 fix (`0fea79e`) targets 1 of 9; rest likely the `1e-12` log-epsilon (probably equivalent) |
| `predictors.py` | 22 | 15 | 7 | 0 | 1 fix (`a84e04b`) targets 1 of 7; rest likely ML hyperparameters (probably equivalent) |
| `control.py` | 52 | 47 | 5 | 0 | 1 fix (`32d92fa`) targets 1 of 5 |
| `pgd.py` | — | — | — | — | **Crashed** -- real Windows-only bug in mutmut itself (`cache.py`'s `update_line_numbers` opens files via the system cp1252 locale, not UTF-8; unrelated to compitum code). CI runs on `ubuntu-latest`, so this specific crash is very unlikely to recur there. |
| `integrations/matbench_adapter.py` | 32 | 24 | 8 | 0 | 1 fix (`721a8b1`) targets 1 of 8 |
| `coherence.py` | 55 | 43 | 12 | 0 | 1 fix (`683e041`) targets 1 of 12 |
| `symbolic.py` | 36 | 30 | 6 | 0 | 1 fix (`e2743f0`) targets 3 of 6 (`+`, `*`, `@` operators all fixed by one commit) |
| `security.py` | 49 | 35 | 14 | 0 | 2 fixes (`8d575f5`) target 2+ of 14 (hash-value assertions cover 2 functions) |
| `constraints.py` | 72 | 47 | 23 | 2 | 2 fixes (`6eab4cb`) target 2 of 23; largest raw survivor count before energy.py |
| `energy.py` | 178 | 104 | 73 | 1 | 1 high-leverage fix (`3459682`, `da791ef`) targeting the dominant survivor category (debug-print internals -- only a leading substring was ever checked, now full exact/regex-matched content); likely accounts for a large share of the 73, not yet confirmed |
| `router.py` | 137 | 105 | 32 | 0 | 3 fixes (`7c3200f`) target 3 of 32; 2 more are a confirmed true equivalent mutant (`abs(-x)`), not a gap |

`energy.py` update: real, targeted `mutmut show 21` (against the archived cache) revealed the
survivor isn't debug-print noise at all -- it's `xw = W @ (xR - model.center)`, and *every* energy.py
test uses `center=np.zeros(...)`, where `-center` and `+center` are identical. Fixed with a
nonzero-center test (commit `21e0622`) and **verified with a real targeted re-check: KILLED 1**
(previously SURVIVED). This means the "mostly debug-print internals" read on energy.py's 73
survivors was only partly right -- there are real core-logic gaps mixed in, found by actually
inspecting a diff rather than guessing from source alone.

## Efficiency lessons learned this session (read before continuing)

1. **`mutmut run <id>` re-tests one mutant at a time, but re-runs the full baseline test suite
   every single call (~40-80s overhead each time in this repo).** For a handful of known survivors
   (1-5), targeted per-ID re-checks are the right tool -- confirmed real with energy.py mutant 21.
   For a **large** survivor set (10+), the per-call baseline overhead dominates and a single fresh
   full re-sweep (which pays that baseline cost once, not once per mutant) is actually more
   efficient in aggregate. Don't loop through dozens of IDs individually -- batch fixes, then do
   one fresh sweep.
2. **Never repurpose the shared `.mutmut-cache` for a different file's spot-check without
   archiving the current file's in-progress cache first.** This session lost `router.py`'s 65/137
   in-progress run this exact way -- `.mutmut-cache` was overwritten with `energy.py`'s archived
   copy mid-run, with no backup of router's own progress since it hadn't finished (and therefore
   hadn't been archived to `artifacts/mutation_cache/` yet -- only *completed* shards get archived
   by `run_shard.sh`). If you need to pause a long-running shard to spot-check something else,
   copy `.mutmut-cache` to a scratch location first, unconditionally, before touching it further.
3. **Killing a mutmut process mid-mutant can leave the mutated source on disk.** Always
   `git status --short src/` (and revert if dirty) before trusting any test run or committing
   anything, immediately after any process kill.
4. For genuinely equivalent mutants (ML hyperparameters in `predictors.py`, the log-stability
   epsilon in `boundary.py`), don't force a test -- note them explicitly as equivalent and move on.
   Chasing 100% of survivors that don't represent real behavioral gaps burns time without adding
   real correctness value.

## Next steps, in priority order

All 16 runnable files have real mutmut data and at least one committed fix; none have been
re-verified with a fresh full sweep yet (only `energy.py` mutant 21 and `router.py`'s were spot-
confirmed via targeted single-ID re-checks). In priority order for a future session:

1. `energy.py`: run a single fresh full sweep (not per-ID loops -- 73 survivors is past the point
   where per-ID overhead pays off) to get a real post-fix count for both the debug-print fix
   (`da791ef`) and the center-sign fix (`21e0622`).
2. `constraints.py` (23 survivors) and `security.py` (14): next-largest counts, each with 2 fixes
   committed.
3. The remaining 7 files (`capabilities.py`, `effort_qp.py`, `boundary.py`, `predictors.py`,
   `control.py`, `matbench_adapter.py`, `coherence.py`, `symbolic.py`, `router.py`) each have
   single-digit-to-low-teens survivor counts -- a batched fresh sweep per file is cheap once a
   few more fixes accumulate per file, rather than re-sweeping after every single commit.
4. Decide whether to pursue `pgd.py` further (the Windows-only crash) -- likely not worth it locally
   given CI runs on `ubuntu-latest`; a real CI run of the fixed workflow would settle this for free.

## CI-side changes made this session (informed by this data)

- Fixed the `--cov` file-path-vs-dotted-module bug and the incomplete `TEST_PATHS` list that meant
  mutmut had likely been silently mutating zero lines across most/all of the 17-shard matrix
  (commit `0cd02a3`).
- Fixed `mutmut_survivor_details.py` extracting header count numbers instead of real survivor IDs
  (commit `fe91012`).
- Changed the scheduled sweep from daily to weekly (commit `18c1766`) -- real compute even though
  public-repo Actions minutes are free.
- Added `.mutmut-cache` restoration per shard, keyed on that shard's source + the full tests/ tree,
  so unrelated future runs reuse prior results instead of re-testing from scratch (commit
  `a37604a`).
- `ENABLE_MUTATION_SCHEDULE` repo variable: still unset (intentionally) -- set it only once the
  weekly-cadence commit is actually on `main`, or the old daily cadence would fire for real once
  before the fix takes effect.
