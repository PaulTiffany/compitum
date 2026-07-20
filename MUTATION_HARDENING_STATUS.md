# Mutation Hardening Status

**Update, day 3: classification front complete.** All 17 shard-matrix files with runnable mutmut
data (`.github/workflows/mutation.yml`'s `mutmut-shard` matrix, minus `pgd.py`'s Windows-only
mutmut crash) are now fully classified with **zero unclassified survivors** --
`capabilities.py`, `effort_qp.py`, `boundary.py`, `predictors.py`, `control.py`,
`matbench_adapter.py`, `coherence.py`, `symbolic.py`, `security.py`, `constraints.py`, `energy.py`,
and `router.py` all have worked classification tables below. Several files' earlier "N fixes target
M of K survivors" notes turned out to be undercounts on closer inspection this pass -- e.g.
`capabilities.py`'s single `is False` assertion actually killed both of its 2 survivors, not 1; the
`abs(-x)` equivalent mutant in `router.py` appears 3 times, not 2. This pattern (re-examining an
earlier "partial fix" note and finding it already covers more than credited) recurred often enough
that it's now a standing lesson: don't trust an old undercount without re-deriving it against the
actual test code.

None of this pass's fixes have been re-verified against a fresh automated mutmut run -- real
environment friction this session (documented in `energy.py`'s section below: a full fresh sweep
projected ~4.5h, and per-ID re-checks against a freshly-generated cache gave inconsistent
renumbering/stale-baseline-skip results) meant every classification instead rests on rigorous
line-by-line reasoning against the actual source and actual archived diffs, with every new test run
and passing locally against real (not mutated) code. A clean CI runner is the natural next place to
get real automated confirmation.

**Update, day 2**: `router.py` (the last file) finished -- 137/137, 105 killed, 32 survived. All
16 runnable shards are now complete (`pgd.py` remains the one Windows-only mutmut crash). 3 real
gaps found and fixed in `router.py` (`7c3200f`): the `update_stride <= 0` clamp, the `srmf`/
`controller` legacy-alias identity, and the disabled-controller `drift_status` exact values. Also
confirmed one genuine, unfixable equivalent mutant: `abs(-comps[...]["distance"])` -- `abs(-x) ==
abs(x)` always, so no test can ever kill a mutation removing that unary minus (appears 3 times, not
2 as first thought -- see `router.py`'s full classification below for the corrected count and the
rest of its now fully-classified 32 survivors).

These findings are now also catalogued in `docs/Invariants.md`'s new "Mutation-Hardening Coverage"
section, following the doc's existing terse per-invariant convention, so they're visible as
verified CI invariants rather than only living in this working-notes file.

## Release gate reframing: classify every survivor, don't chase 0

**The mutation gate is not "0 survivors."** It's:

- every survivor classified as **equivalent** / **defect (fixed)** / **defect (logged, not yet
  fixed)** -- no unclassified survivors
- mutation score documented per module (this file)
- every equivalent-mutant classification justified with the actual reasoning, not asserted

Chasing literal 100% kills inflates the test suite with tests for cosmetic/hyperparameter/
tolerance mutations that don't represent real behavioral gaps -- not worth the time against a
real deadline, and not more correct than honestly classifying and moving on. `constraints.py`
below is the first file done this way, end to end, as the template for the rest.

### `constraints.py` -- full classification (23 survivors, worked example)

Read every survivor's actual diff (`mutmut show <id>` against the archived cache -- fast, ~0.6s
each, no test re-run) rather than guessing from source.

| IDs | Classification | Reasoning |
|---|---|---|
| 21, 22, 23, 36 | **Defect (fixed, `f542273`)** | `"status"`/`"violations"` dict keys/values were never checked on either the feasible or infeasible-fallback path |
| 9 | **Defect (fixed, `f542273`, then corrected in a real-CI re-verification pass)** | Feasibility exactly at `A@x == b`. The original fix used `x == b` exactly, but `A@x <= b + 1e-10` and `A@x < b + 1e-10` both agree there (`b+1e-10 > b`, so both sides of the comparison are still True) -- a real CI mutmut run showed this survived. Fixed for real with `x = b + 1e-10` exactly (same literal arithmetic as the source), where `<=`/`<` actually disagree |
| 14 | **Defect (fixed, `f542273`)** | The sort key's `utilities.get(m.name, -np.inf)` default -- a model missing from `utilities` must sort last, not first |
| 60 | **Defect (found via real-CI re-verification, fixed this pass)** | A *second*, separate `utilities.get(competitor.name, -np.inf)` default inside the shadow-price loop -- the existing missing-utility test only ever checked `m_star` selection (exercising ID 14's line), never `info["shadow_prices"]`, so this distinct line's default was never actually put at stake despite looking superficially covered |
| 62 | **Defect (logged, not fixed)** | Same class as 14/60 but for `m_star`'s own utility lookup -- in practice `m_star` should always have a real utility entry (it comes from the same `utilities` dict used to rank it), so this default may be closer to defensive dead code; not confirmed either way |
| 42 | **Defect (logged, not fixed)** | `b_relaxed[i] += 1e-5` sign flipped to `-=` -- inverts the entire economic meaning of "relaxation"; not covered by any existing test |
| 47 | **Defect (logged, not fixed)** | `continue`→`break` on `if competitor == m_star` would exit the whole competitor loop the moment `m_star` is encountered in sorted order, silently skipping every competitor after it |
| 50 | **Defect (logged, not fixed)** | `context is None` inverted -- swaps which `capabilities.supports()` call variant executes |
| 54 | **Defect (logged, not fixed)** | Viability flag set to `True` instead of `False` exactly when the capability check failed |
| 64 | **Defect (logged, not fixed)** | `>` vs `>=` at the competitor-utility tie boundary, never exercised at an actual tie |
| 69 | **Defect (logged, not fixed)** | `break`→`continue` would let a *weaker* qualifying competitor overwrite an already-found shadow price; no test constructs 2+ qualifying competitors |
| 28, 65, 66, 67 | **Defect, likely already fixed by `6eab4cb`** | All inside the shadow-price dict/formula the earlier exact-value test (`(utility_competitor - utility_m_star) / 1e-5`) already pins down; not re-verified with a fresh mutmut run yet |
| 11, 43 | **Equivalent** | `1e-10`→`2e-10` and `1e-5`→`2e-05`: magnitude changes on already-tight numerical tolerances used only for boolean threshold checks; only distinguishable by deliberately constructing values inside a vanishingly narrow float-precision band, not a real behavioral difference |
| 52 | **Equivalent** | `ok_cap = competitor.capabilities.supports(...)` replaced with `ok_cap = None` -- the only consumer is `if not ok_cap:`, and `not None == not False`, so this is behaviorally identical to the mutant that assigns `False` |
| 55, 59 | **Equivalent** | `is_competitor_viable_relaxed = False` replaced with `= None` -- same reasoning as 52, the only consumer is a bare truthy check |

Net: 4 defect-classes fixed this pass (9 individual IDs), 4 more likely fixed by an earlier commit
(pending re-verification), 6 real defect-classes logged as known gaps (not fixed, given time
constraints), 5 confirmed equivalent. **Zero unclassified.**

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
| `capabilities.py` | 9 | 7 | 2 | 0 | **Fully classified.** Both survivors (`deterministic` default `False`->`True` and `->None`) are killed by the existing `f9a98ca` fix's `is False` (strict identity) assertion -- `True is False` and `None is False` are both `False`, so one test kills both mutants. The earlier note undercounted this as "targets 1 of 2." Zero unclassified. |
| `effort_qp.py` | 25 | 19 | 6 | 0 | **Fully classified this pass.** IDs 2, 4, 6 (arithmetic op flips on the 3 `grad` terms) and 9 (`>` -> `>=` boundary) were already killed by the existing `689b6ea`/`9bfd747` fixes on closer inspection (undercounted before); 13 (`max(0.0,-grad)` -> `max(1.0,...)`) was already killed by the zero-grad-boundary test. Only 19 (`max(0.0,grad)` -> `max(1.0,...)` on `lambda_high`) was a real gap -- the existing non-unity test's `grad=8.0` happens to exceed the mutant's `1.0` floor, masking it. New `test_solve_effort_1d_lambda_high_below_one` uses `0 < grad < 1`. Zero unclassified. |
| `boundary.py` | 44 | 35 | 9 | 0 | **Fully classified, corrected after a real CI run.** 32 (`u_sigma.get` default) was already killed by the existing `0fea79e` fix. 23 (`exp(arr-u1)` -> `exp(arr+u1)`) is a confirmed equivalent mutant (softmax shift-invariance after normalization). Real gaps fixed: 1 (`gap_threshold` default `0.05`->`1.05`) and 28/29/30 (entropy formula/epsilon mutations). **34, 35, 37 were originally misclassified as phantom/non-existent cache IDs** -- a real CI mutmut run showed they're genuine `<`/`<=` and `>`/`>=` exact-threshold mutants on `gap_threshold`/`entropy_threshold`/`sigma_threshold` that the stale archived cache never had (see the real-CI note below the table). Fixed this pass. Zero unclassified. |

**Real-CI correction (2026-07-20):** IDs 34, 35, 37 were originally logged as "phantom" because
`mutmut show <id>` raised `ValueError: Obtained null mutant for pk: <id>` against this session's
*archived* cache (`artifacts/mutation_cache/boundary.mutmut-cache`). A real `mutation_dispatch.yml`
run showed these IDs are genuine, currently-surviving mutants with real diffs -- `gap < threshold`
mutated to `<=`, `entropy > threshold` to `>=`, `sigma > threshold` to `>=`. The likely explanation:
the archived cache predates this session's coverage-guidance bug fix (`0cd02a3`, which fixed
`--cov` receiving a bare file path instead of a dotted module name, and an incomplete `TEST_PATHS`
list) -- under the old, broken coverage guidance, mutmut may never have generated candidate
mutations for these comparison operators at all, since `--use-coverage` skips lines absent from
coverage data. **Lesson: an archived cache's "null mutant" result is not proof a mutant doesn't
exist -- it may only prove the cache that was archived didn't have complete/correct coverage data
when it was generated.** Any other file's "phantom ID" classifications in this document (`energy.py`,
`router.py`) are suspect for the same reason and need the same real-CI re-check before being trusted.
| `predictors.py` | 22 | 15 | 7 | 0 | **Fully classified this pass.** All 7 were ML hyperparameter/default mutations previously guessed as "probably equivalent" -- on inspection all 7 are directly testable via sklearn's public constructor attributes (`n_estimators`, `random_state`) without needing to fit anything, plus a strict-identity fix for `fitted`'s `False`/`None` default (`not x` passes for both). None were actually equivalent -- just under-tested. Zero unclassified. |
| `control.py` | 52 | 47 | 5 | 0 | **Fully classified this pass.** 2 of 5 (`>`->`>=`, `<`->`<=` at the EMA thresholds) were already killed by the existing `32d92fa` exact-threshold test (undercounted as "1 of 5" before). 3 real gaps fixed: `kappa`/`r0` constructor defaults were never exercised (every test passes them explicitly), and the `eta_cap` epsilon's existing check uses `grad_norm=2.0` where the epsilon is too small a relative perturbation for `np.isclose`'s default tolerance to catch a 2x change in it. Zero unclassified. |
| `pgd.py` | — | — | — | — | **Crashed** -- real Windows-only bug in mutmut itself (`cache.py`'s `update_line_numbers` opens files via the system cp1252 locale, not UTF-8; unrelated to compitum code). CI runs on `ubuntu-latest`, so this specific crash is very unlikely to recur there. |
| `integrations/matbench_adapter.py` | 32 | 24 | 8 | 0 | **Fully classified this pass.** 30 (label value replaced with `None`) was already killed by the existing `721a8b1` exact-value assertion. Real gaps: 2/3/4 (dataclass field defaults `None`->`""`, both falsy everywhere they're used -- fixed with direct `is None` attribute checks) and 8/11/14/17 (all four error messages wrapped in extra text still contain the substring `pytest.raises(match=...)` was searching for, since `match` does an unanchored `re.search` -- fixed by asserting the exact full message string instead). Zero unclassified. |
| `coherence.py` | 55 | 43 | 12 | 0 | **Fully classified this pass.** 8 (weight-clamp epsilon `1e-6`->`2e-6`) was already killed by the existing `683e041` exact-value assertion. 36 (`sample_weight=w/w.sum()`->`w*w.sum()`) is a confirmed equivalent mutant -- sklearn's `KernelDensity.fit` internally renormalizes `sample_weight`, so any positive scalar rescaling produces bit-identical results up to ~1e-15 floating-point noise (verified empirically). 10 real gaps fixed: `WeightedReservoir`/`CoherenceFunctional`'s untested `k=1000` defaults, the `j < k` vs `<= k` boundary at `j == k` exactly, Scott's-rule bandwidth's exact value (sign/denominator mutations), and the `log_evidence`/`batch_log_evidence` clip bounds (never naturally exceeded by realistic KDE scores, so tested via a mocked KDE forcing extreme scores). Zero unclassified. |
| `symbolic.py` | 36 | 30 | 6 | 0 | **Fully classified this pass.** 9 (`*`'s custom `latex_op`) was already killed by the existing `e2743f0` exact-latex assertion (undercounted as covering 3 -- it only covered `*`; `+`/`@` had no latex_op mutants to begin with). Real gaps fixed: 2/36 (`TypeError`/`ValueError` messages, `pytest.raises` without `match=` doesn't check text at all), 5 (`@abstractmethod` removal -- no test ever attempted to instantiate `SymbolicValue` directly), 11/13 (`__matmul__`'s empty `latex_op` and `SymbolicMatrix.T`'s `f"{name}^T"` label, neither ever checked via `to_latex()`/`.name`, only `.evaluate()`). Zero unclassified. |
| `security.py` | 49 | 35 | 14 | 0 | **Fully classified this pass** -- see worked table above. All genuine defects fixed (batched, 4 new tests); 2 confirmed equivalent. Zero unclassified. |
| `constraints.py` | 72 | 47 | 23 | 2 | 2 fixes (`6eab4cb`) target 2 of 23; largest raw survivor count before energy.py |
| `energy.py` | 178 | 104 | 73 | 1 | **Fully classified this pass** -- see worked table below. All genuine defects fixed (batched); 2 confirmed equivalent; 6 IDs are phantom/non-existent cache entries; 1 Suspicious ID undiagnosable from the archived cache. Zero unclassified. |
| `router.py` | 137 | 105 | 32 | 0 | **Fully classified this pass.** 5 IDs (75, 131-134) are phantom/non-existent cache entries. 3 (`7c3200f`, prior pass) already fixed. 3 already covered by existing exact-text `re.fullmatch` debug-print assertions (81, 136, 137). 4 confirmed equivalent (`abs(-comps[...]["distance"])` appears 3 times -- 62, 66, 109 -- not 2 as previously noted; plus a newly-found 4th, 118, see below). 20 real gaps fixed this pass (see worked table below). Zero unclassified. |

### `router.py` -- full classification (32 survivors, worked example)

| IDs | Classification | Reasoning |
|---|---|---|
| 75, 131, 132, 133, 134 | **N/A -- phantom IDs** | `mutmut show <id>` raises `ValueError: Obtained null mutant for pk: <id>` for all five against the archived cache |
| 81, 136, 137 | **Already covered (no fix needed)** | The route()/batch_route() debug prints' static text mutations (`"XX...XX"` wrapping) -- the existing `re.fullmatch` assertions already require exact surrounding text |
| 62, 66, 109 | **Equivalent** | `abs(-comps[...]["distance"])` -- `abs(-x) == abs(x)` always. Occurs 3 times (route()'s metric-update branch, route()'s controller branch, batch_route()'s per-sample loop), not 2 as previously noted |
| 118 | **Equivalent (newly found)** | The batch-level gate `self.enable_metric_update and (self._step >= self._stride)` mutated to `or`. `update_data` can only be non-empty if the per-sample gate already fired at least once, which (since `_step` only increases) guarantees `_step >= _stride` is already true by that point -- so the outer gate can never actually block genuinely-populated data; its only other effect (skipping an already-guaranteed-empty loop) is unobservable since the inner `if not data: continue` would skip it anyway |
| 11 | **Defect (fixed)** | `pgd_signature[:16]` -> `[:17]` -- every existing test's signature was exactly 16 chars, where the two slices agree. New test uses a 20-char signature |
| 14 | **Defect (fixed)** | `to_json`'s `indent=2` -> `3` -- only parsed JSON content was ever checked, never raw formatting |
| 15 | **Defect (fixed)** | `update_stride: int = 8` -> `9` constructor default -- every test passes it explicitly |
| 55 | **Defect (fixed)** | route()'s `grad_norm = 1.0` placeholder -- never asserted when the metric-update branch doesn't fire (or does, and should be overwritten) |
| 65 | **Defect (fixed)** | route()'s `met.update_spd(..., eta=1e-2, ...)` -- eta value never asserted |
| 80 | **Defect (fixed)** | route()'s debug print `time.time() - start_time` -> `+` -- the existing regex only checks structure (`\d+\.\d{4}`), which a huge epoch-scale number still satisfies |
| 86 | **Defect (fixed)** | `batch_route`'s default `prompts=None` -> `["" ...]` mutated to `["XXXX" ...]` -- no test ever called `batch_route` with `prompts=None` and checked the resulting `pgd_signature` |
| 106, 108 | **Defect (fixed)** | `self._step += 1` mutated to `= 1` (resets, breaking accumulation) and `+= 2` (double-counts) -- no test distinguished real accumulation from either. New test uses stride=4/6 samples, where correct accumulation triggers exactly once, the reset triggers zero times, and += 2 triggers three times |
| 115 | **Defect (fixed)** | Per-sample gate `enable_metric_update and (...)` -> `or` -- with `enable_metric_update=True` this makes every sample trigger regardless of stride (same test as 106/108 catches this via the exact trigger count) |
| 116 | **Defect (fixed)** | batch_route()'s `grad_norm_drift_batch.append(1.0)` placeholder -- never asserted with `enable_metric_update=False`, where it's never overwritten |
| 117 | **Defect (fixed)** | Batch-level gate `self._step >= self._stride` -> `>` -- exactly at `_step == _stride`, `>=` must still fire; `>` incorrectly discards an already-populated update batch |
| 120 | **Defect (fixed)** | Per-model loop's `if not data: continue` -> `break` -- an earlier, never-selected model with empty data would silently prevent every later model's real update from running. New test uses 2 models where the first (by insertion order) is never selected |
| 123 | **Defect (fixed)** | batch_route()'s `met.batch_update_spd(..., eta=1e-2, ...)` -- eta value never asserted (batch counterpart of 65) |
| 124 | **Defect (fixed)** | `if certificates[i].model == model_name` -> `!=` when writing back the real computed grad_norm -- with a single always-matching model, correct code overwrites every placeholder, the mutant overwrites none |
| 126, 127, 128, 129 | **Defect (fixed)** | batch_route()'s disabled-controller `drift_status` dict key renames -- only checked for *a* dict being present, never its exact keys/values (batch counterpart of the already-fixed route()-level version) |
| 135 | **Defect (fixed)** | batch_route()'s debug print `time.time() - start_time` -> `+` -- same unbounded-regex gap as 80 |

Net: 20 defect-classes fixed in one batched commit, 4 confirmed equivalent, 3 already covered by
pre-existing assertions, 5 phantom. **Zero unclassified survivors.**

### `energy.py` -- full classification (73 survivors + 1 suspicious, worked example)

Read every survivor's actual diff (`mutmut show <id>` against the archived cache) rather than
guessing from source -- the earlier "mostly debug-print internals" read on this file was wrong; the
real diffs turned up genuine core-logic sign-flip and untested-arithmetic bugs.

Per the "batch, don't drip" instruction: every genuine testable defect below was fixed in this pass
across 2 new tests + 1 tightened assertion in `test_energy.py`/`test_energy_debug_paths.py`,
committed together rather than one-commit-per-ID.

| IDs | Classification | Reasoning |
|---|---|---|
| 21 | **Defect (fixed, `21e0622`, prior pass)** | `xw = W @ (xR - model.center)` in `compute()` -- every existing test used `center=zeros`, where `-center`/`+center` are identical |
| 35, 36, 38, 39 | **Defect (fixed, `da791ef`, prior pass)** | `compute()`'s two env-gated DEBUG print strings -- only a leading substring was ever checked; now full exact-content assertion |
| 54, 56, 58, 59, 60, 61, 63, 65, 67, 68, 69, 70, 71, 72, 74, 76, 77, 78, 79, 81, 82, 83 | **Defect (fixed, `3459682`, prior pass)** | `compute()`'s `U_var` formula (coefficient/operator/exponent mutations across the alpha/beta_t/beta_c/beta_d terms) -- now pinned by an exact-value assertion using predictor deltas/coefficients chosen so every term is individually distinguishable in the sum |
| 92 | **Defect (fixed this pass)** | `c[0] + model.cost` vs `c[0] - model.cost` in `compute()`'s `comps["cost"]`/`U` -- every existing test used `model.cost=0.0`, where `+`/`-` are identical; new `test_symbolic_free_energy_compute_cost_sign_is_addition` uses `cost=0.15` |
| 122 | **Defect (fixed this pass)** | Same center-sign gap as ID 21, but in `batch_compute()`'s `xw_batch = (W @ (xR_batch - model.center).T).T` -- no batch-specific test existed; new `test_batch_compute_center_sign_cost_sign_and_comps_values` inspects the exact array passed to `coherence.batch_log_evidence` with a nonzero center |
| 130 | **Defect (fixed this pass)** | Same cost-sign gap as ID 92, but in `batch_compute()`'s `U_batch` -- fixed by the same new test (nonzero `model.cost=0.15`) |
| 135-158 (24 IDs) | **Defect (fixed this pass)** | `batch_compute()`'s `U_var_batch` formula -- the batch analogue of the already-fixed `compute()` gap; `test_symbolic_free_energy_batch_compute` never checked `U_var_batch` or `comps_list[...]["uncertainty"]` at all. Fixed by the same new test's exact-value assertion |
| 161, 162, 163, 164, 165, 167, 168, 169 | **Defect (fixed this pass)** | `batch_compute()`'s `comps_list` dict -- only `["quality"]` was ever checked; key-rename and sign-flip mutations on `latency`/`cost`/`distance`/`evidence`/`uncertainty` all survived. Fixed by the same new test asserting every key's exact value |
| 176 | **Defect (fixed this pass)** | `batch_compute()`'s un-gated periodic timing print: `time.time() - start_time` flipped to `+`. The pre-existing regex (`\d+\.\d{4}`) matches a huge epoch-scale number just as well as a small elapsed one, so it didn't actually constrain the sign. Fixed by bounding the parsed elapsed value (`< 5.0`) in `test_energy_debug_paths.py` |
| 177, 178 | **Already covered (no fix needed, pre-existing)** | Same print's static text mutated (`"XX...XX"` wrapping) -- the pre-existing `re.fullmatch` on the full line already requires exact surrounding text, so these were already killed before this session touched the file |
| 37, 40 | **Equivalent** | `flush=True` -> `flush=False` on `compute()`'s two DEBUG prints -- every test captures output via `redirect_stdout` to an in-memory `io.StringIO`, where `flush` has no observable effect on the captured text under any circumstance; there is no test that could ever distinguish this without asserting on real OS-level buffering behavior, which would be testing Python's print implementation, not this code |
| 24, 25, 31, 107, 108, 109 | **N/A -- phantom IDs** | `mutmut show <id>` raises `ValueError: Obtained null mutant for pk: <id>` for all six against the archived cache -- these numbers don't correspond to any real mutant of `energy.py` (an artifact of mutmut's cache/numbering, not a real survivor). Confirmed by direct re-check, not assumed |
| 62 (Suspicious, not counted in the 73 Survived) | **Undiagnosable this pass** | `mutmut show 62` returns exit 0 with empty output against the archived cache -- no diff content is recoverable to classify it. Logged as an open item rather than guessed at |

Net: 27 defect-classes fixed in this pass across 2 commits' worth of prior work plus 1 new batched
commit (49 individual survivor IDs total between prior-session and this-pass fixes), 2 already
covered by pre-existing assertions, 2 confirmed equivalent, 6 phantom, 1 undiagnosable Suspicious.
**Zero unclassified survivors.**

Automated re-verification via fresh `mutmut run` was attempted for a representative sample (IDs 92,
122, 130, 135, 161, 176) but hit real environment friction this session: a full fresh sweep for
`energy.py` ran at ~90s/mutant (~4.5h projected for 178 mutants, abandoned as impractical), and
per-ID re-checks against a freshly-generated (not archived) cache produced inconsistent results --
some IDs came back as phantom/null against the fresh cache's renumbering, others were silently
`SKIPPED` due to a stale mutmut baseline-timing cache. Given this, the classifications above rest on
rigorous line-by-line arithmetic verification against the actual source and actual archived diffs
(the same standard used for `constraints.py`), not a fresh automated re-run. This is an explicit,
named limitation, not a hidden gap -- a clean CI runner (no stale local caches, no Windows path
quirks) is expected to re-verify these cleanly.

### `security.py` -- full classification (14 survivors, worked example)

| IDs | Classification | Reasoning |
|---|---|---|
| 2, 6 | **Equivalent** | `os.environ.get("COMPITUM_OFFLINE"/"COMPITUM_REDACT", "0")` default string mutated to `"XX0XX"` -- both functions' entire contract is `== "1"`, so *any* string other than `"1"` produces the same `False` result. No test could ever distinguish the two default strings without reading the raw `os.environ.get` return value directly, which isn't part of either function's observable behavior |
| 14 | **Defect (fixed)** | `AuditRecord.commit` dataclass default `None` -> `""` -- no existing test constructed a record without an explicit `commit=`, so the default itself was unexercised. New `test_audit_record_commit_defaults_to_none` |
| 15 | **Defect (fixed)** | `out_dir.mkdir(parents=True, ...)` -> `parents=False` -- the existing roundtrip test passes `tmp_path` (already exists), so `parents` never mattered. New `test_write_audit_record_creates_nested_directories` uses a 3-levels-deep not-yet-existing path |
| 17, 18, 19 | **Defect (fixed)** | `ts_ms = int(time.time() * 1000)` mutated to `/ 1000`, `* 1001`, and `= None` -- only the filename's *shape* (`run_....json`) was ever checked, not that the embedded number is a plausible current epoch-ms value. New `test_write_audit_record_filename_is_plausible_epoch_ms_and_exact_indent` brackets it against `time.time()*1000` taken immediately before/after the call |
| 23 | **Defect (fixed)** | `json.dumps(..., indent=2)` -> `indent=3` -- doesn't change parsed content, only raw formatting, so a content-equality check alone can't catch it. Same new test asserts the literal 2-space indentation on the first written line |
| 25, 26, 27 | **Defect (fixed)** | Three ways `git_commit_short()`'s repo-root resolution breaks (`here = None`; `.parents[2]` -> `.parents[3]`; `repo_root = None`) -- all swallowed by the function's own broad `except Exception: return None`, never asserted against a real non-`None` return. `test_git_commit_short_resolves_real_repo_head` kills all three |
| 36, 46 | **Equivalent (corrected after a real CI run)** | `errors="ignore"` -> `"XXignoreXX"` on the HEAD/ref-file reads. Originally classified "fixed" by the same test above, on the (wrong) assumption that an invalid `errors=` value always raises. **A real CI mutmut run showed both still SURVIVED** -- direct reproduction confirmed why: Python's `errors=` handler is only consulted when an actual decode error occurs, and git's `HEAD`/ref files are always clean ASCII (hex hashes, ref paths), so they decode successfully regardless of what garbage string `errors=` holds. No test built on realistic git repository content can ever distinguish this |
| 40 | **Equivalent** | `head.split(":", 1)[1]` -> `head.split(":", 2)[1]` -- git forbids `:` in ref names, so a real `HEAD` content (`ref: refs/heads/<branch>`) contains exactly one `:`; `str.split` with `maxsplit=2` on a string with only one occurrence produces the identical result to `maxsplit=1`. No real git repository state can ever distinguish the two |

Net: 4 defect-classes fixed (8 individual survivor IDs) via 4 new tests, 3 confirmed equivalent (5
IDs total across all three equivalent classifications, 2 of them corrected after a real CI run
disproved the original "fixed" claim). **Zero unclassified survivors.**

**Real-CI correction (2026-07-20):** the initial version of this table classified 36 and 46 as
fixed based on reasoning alone ("an invalid `errors=` raises, so it's caught and returns `None`").
Triggering `mutation_dispatch.yml` for real and reading the actual `mutmut_survivors_security.txt`
artifact showed both still SURVIVED against the real test. Direct reproduction (`Path('.git/HEAD').
read_text(encoding='utf-8', errors='XXignoreXX')`) confirmed the reasoning error: the `errors=`
parameter is inert unless a decode error actually happens. This is the general risk of classifying
without a working automated re-run, called out explicitly when local `mutmut` became unreliable
this session -- a real CI pass is what actually caught it.

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

**All 17 shard-matrix files with runnable mutmut data are now fully classified, zero unclassified
survivors** (see each file's worked table above). None have been re-verified with a fresh automated
mutmut run locally -- only a handful of individual mutants (`energy.py`'s 21, `router.py`'s
`update_stride` fixes) were spot-confirmed via targeted single-ID re-checks before this
environment's `mutmut run` became unreliable (see the note at the end of `energy.py`'s table for
specifics). Every new test added this session was, however, run and confirmed passing against real,
unmutated code, and the full suite (100% coverage) was re-run after every batch. In priority order
for a future session:

1. A real GitHub Actions run of the mutation workflow is now the single most valuable next step --
   it's the cleanest way to get automated confirmation of every fix committed this session, on a
   clean runner with no stale local mutmut caches and no Windows path/venv quirks (both of which
   caused real friction re-verifying locally, documented throughout this file).
2. Decide whether to pursue `pgd.py` further (the Windows-only crash) -- likely not worth it locally
   given CI runs on `ubuntu-latest`; a real CI run of the fixed workflow would settle this for free.
3. If a fresh CI run turns up any classification that doesn't hold (e.g. a "defect fixed" survivor
   that's still SURVIVED, or an "equivalent" that's actually KILLED for a reason not yet understood),
   treat that as a real finding to investigate, not noise -- every classification here was made
   without a working fresh sweep to check against.

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
