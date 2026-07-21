# Invariants

Compitum defends a set of invariants checked by tests and by runtime process signals:

- Lyapunov-like energy drift does not increase under allowed updates.
- Constraint residuals respect `A · x ≤ b` with documented slack.
- Metric coherence: neighborhoods act like neighborhoods in the SPD metric.
- Boundary behavior: tie regions are observable with gap/entropy/uncertainty checks.

See the `tests/invariants/` suite for details and run with `pytest -m invariants`.

## Coverage (selected)

- Geometry (SPD metric)
  - Ray monotonicity: `test_invariants_metric_ray.py`
  - Triangle inequality: `test_invariants_metric_triangle.py`
  - Update descent + SPD: `test_invariants_metric_update.py`
  - SPD eigenvalue bounds: `tests/metric/test_metric_spd_bounds.py`

- Coherence / OOD
  - Monotone outward: `test_invariants_coherence.py`
  - Symmetry (±v): `test_invariants_coherence_symmetry.py`
  - Density–distance coupling: `test_invariants_density_energy_coupling.py`
  - Score directionality (finite diff): `test_invariants_coherence_score_dir.py`
  - Mixture discrimination: `tests/coherence/test_coherence_mixture_discrimination.py`

- Control / Lyapunov
  - Properties (bounds, monotonicity): `test_invariants_control_props.py`
  - Sequences (ΔV non-increase/bounded, recovery): `test_invariants_control_sequences.py`, `test_invariants_control_lyapunov.py`, `test_invariants_control_deltaV_strong.py`, `test_invariants_control_combined_proxy.py`

- Constraints / Duals
  - Feasibility monotone in b: `test_invariants_constraints_monotone.py`
  - Dual sanity/monotone/near-binding: `test_invariants_constraints_duals.py`, `test_invariants_duals_monotone.py`, `test_invariants_duals_near_binding.py`
  - Dual scaling sanity: `tests/invariants/test_invariants_duals_scaling.py`
  - Argmax stability: `test_invariants_solver_argmax.py`

- Router / Certificates
  - Determinism (repeated route): `test_invariants_router_determinism.py`
  - Paraphrase flip budget + explainability: `test_paraphrase_suite.py`, `test_paraphrase_explainability.py`
  - JSON structure/schema: `tests/certificates/*.py`

Run the fast suite:

```
pytest -q tests/invariants
```

## Mutation-Hardening Coverage (Boundary & Edge-Case Invariants)

A separate category from the property-based suite above: exact-value and boundary-condition
invariants found via a real mutmut sweep across the full `src/compitum` shard matrix (16/17 files
run to completion; see `MUTATION_HARDENING_STATUS.md` for per-file scores). Each line below was a
genuine survivor -- an existing test exercised the code path but didn't pin down the exact value or
boundary needed to actually kill a behavioral mutation -- confirmed against real code before the
fix, and in `energy.py`'s case confirmed killed against the real mutant diff.

- `effort_qp.py`: `e_star` resolves to `0.0` (not `1.0`) exactly at the `grad == 0` boundary; q1/t1/c1
  multiplier terms are exercised with non-unity values; `lambda_high`'s `max(0.0, grad)` floor is
  exercised with `0 < grad < 1`, not just `grad > 1` where a `1.0` floor mutation happens to be
  masked — `tests/test_effort_qp.py`
- `capabilities.py`: `Capabilities.deterministic` defaults to `False` when omitted —
  `tests/test_capabilities.py`
- `boundary.py`: `uncertainty` defaults to `0.0` (and correctly resolves `is_boundary=False`) when
  the winning model is missing from `u_sigma`; the real `gap_threshold` default (`0.05`) is
  exercised where a much-larger mutated default would flip the result (every other case either has
  a small gap or a failing sigma condition that masks it); `entropy`'s exact value is pinned (not
  just which side of `entropy_threshold` it lands on); `gap < gap_threshold`,
  `entropy > entropy_threshold`, and `sigma > sigma_threshold` are each exercised at *exact*
  equality (not just clearly inside/outside), where a `<`/`>` vs `<=`/`>=` mutation would otherwise
  survive — `tests/test_boundary.py`
- `predictors.py`: isotonic calibration clips out-of-domain raw values to the boundary's fitted `y`
  rather than extrapolating; each of the 3 internal `GradientBoostingRegressor`s' exact
  `n_estimators`/`random_state` hyperparameters; `fitted` defaults to `False`, not `None` (`not x`
  passes for both) — `tests/test_predictors.py`
- `control.py`: the EMA trust-region branch is neutral (no change) exactly at the 1.5x/0.7x
  thresholds, not just clearly above/below them; `kappa`/`r0` constructor defaults (`0.1`/`1.0`,
  never exercised since every other test passes them explicitly); `eta_cap`'s `+1e-6` epsilon is
  checked at `grad_norm=0.0`, where it's the entire denominator, not `2.0`, where it's too small a
  relative perturbation for the default tolerance to catch — `tests/test_control.py`
- `symbolic.py`: `+`, `*`, `@` operators evaluate correctly, not just `-`/`/`; `SymbolicValue.to_latex`
  is genuinely `@abstractmethod` (direct instantiation raises `TypeError`); `TypeError`/`ValueError`
  messages match their exact text, not just that *some* exception of that type was raised;
  `__matmul__`'s empty `latex_op` and `SymbolicMatrix.T`'s `f"{name}^T"` label are both checked via
  `to_latex()`/`.name`, not just `.evaluate()` — `tests/test_symbolic.py`
- `security.py`: SHA-256 outputs match the exact expected digest (not just length 64);
  `is_offline()`/`redaction_enabled()` default to `False` when unset; `AuditRecord.commit` defaults
  to `None`, not `""`; `write_audit_record` creates missing parent directories and its filename
  embeds a plausible current epoch-ms timestamp with exact 2-space JSON indentation;
  `git_commit_short()` resolves a real, well-formed commit hash from this repo's own `.git` state
  (not silently swallowed to `None`) — `tests/security/test_security_utils.py`
- `constraints.py`: shadow price matches the exact `(Δutility)/1e-5` value, not just its sign;
  multi-constraint infeasible fallback zeros every `lambda_i`; feasibility's `A@x <= b + 1e-10`
  tolerance is exercised at `A@x` exactly equal to `b + 1e-10` (not just `b` itself, where the
  epsilon makes `<=` and `<` agree regardless); a competitor missing from `utilities` still loses
  in the shadow-price calculation specifically (`-inf` default), not just in `m_star` selection (a
  separate line with the same default pattern) — `tests/test_constraints.py`
- `integrations/matbench_adapter.py`: CSV columns map to the correct attribute *values*, not just
  present attributes; `id_column`/`formula_column`/`label_column` default to `None`, not `""`
  (both falsy everywhere they're checked); all 4 "column not found"/"missing columns" error
  messages match their exact full text, not just an unanchored substring (`pytest.raises(match=)`
  is a substring search, so text wrapped around the expected substring still "matches") —
  `tests/integrations/test_matbench_adapter.py`
- `coherence.py`: `WeightedReservoir` clamps nonpositive weights to `1e-6`; `WeightedReservoir`'s and
  `CoherenceFunctional`'s `k` constructor default is `1000`; reservoir replacement doesn't fire when
  the sampled index equals `k` exactly (`j < k`, not `<=`); the fitted KDE's bandwidth matches
  Scott's rule exactly (`n ** (-1/(d+4))`); `log_evidence`/`batch_log_evidence` clip to exactly
  `[-10.0, 10.0]` (checked via a mocked KDE forcing extreme scores, since realistic fits never
  naturally exceed that range) — `tests/test_coherence.py`
- `energy.py`: `comps["uncertainty"]`/`U_var` match the exact variance formula (both `compute()` and
  `batch_compute()`); debug/timing prints match their full exact (or regex-matched, elapsed-time
  bounded) content, not just a leading substring; evidence uses `xR - model.center`, not `+`, in
  both `compute()` and `batch_compute()`; `cost` uses `c + model.cost`, not `-`, in both — every
  prior test used a zero center/zero cost, where the two are indistinguishable; `batch_compute()`'s
  `comps_list` dict is checked key-by-key (not just `"quality"`); `compute()`'s debug-print gate
  (`step % 100 == 0 and env == "1"`, checked before `_step` increments) is exercised at a nonzero
  multiple of 100 (not just `0`, where `%`/`/` and `%100`/`%101` happen to agree) and with the env
  var unset at `_step == 0` (where an `and`->`or` mutation would print regardless) —
  `tests/test_energy.py`, `tests/test_energy_debug_paths.py`,
  `tests/energy/test_symbolic_free_energy.py`
- `router.py`: `update_stride` floors to `1` for `stride <= 0` and defaults to `8` when omitted;
  `router.srmf is router.controller` (legacy alias); disabled-controller `drift_status` matches the
  controller's exact current state, not just key presence, in both `route()` and `batch_route()`;
  `SwitchCertificate.to_json()` truncates `pgd_signature` to exactly 16 chars and indents with
  exactly 2 spaces; `route()`/`batch_route()`'s `grad_norm` placeholder (`1.0`) and `eta` (`1e-2`)
  values are exact; `batch_route()`'s per-sample `_step` accumulates by exactly 1 (not reset, not
  doubled); its batch-level update gate fires at `_step == _stride` exactly (`>=`, not `>`); its
  per-model update loop uses `continue` (not `break`) so an earlier never-selected model can't block
  a later one's real update; `batch_route(prompts=None)` defaults to empty-string prompts, not a
  placeholder; both route()'s and batch_route()'s debug-print elapsed times are bounded, not just
  regex-shaped (a `time.time() + start_time` sign flip still matches `\d+\.\d{4}`) —
  `tests/test_router_simple.py`, `tests/test_router_batch.py`
- `metric.py`: `_update_cholesky()`'s error-recovery delta/prints are exact and the recovered `W` is
  upper-triangular; `distance()`/`batch_distance()`'s `len(whitened_residuals) > rank` boundary is
  exercised at exact equality, and `z = x - mu` (not `+ mu`); the `ValueError` when `W` stays `None`
  matches its exact message; the sigma-squared clamp floors at exactly `0.0`, not `1.0` (forced via
  a mocked indefinite covariance, since a real PSD one never goes negative); `batch_update_spd()`'s
  `d_batch_safe` epsilon, `A_batch`/`grad_L`/`z_norm2_batch`/`lipschitz`/`eta_stab` arithmetic, and
  the gradient-descent direction (`- eta_eff * grad_L`, not `+`) are all pinned to exact values; a
  wildly-skewed-magnitude batch forces the backtracking loop to genuinely engage (the `eta_stab`
  "stability cap" is only an average-based Lipschitz estimate, not worst-case), pinning its `eta_eff
  *= 0.5` / `new_L = self.L - eta_eff * grad_L` arithmetic; `fnorm > 10.0`'s clamp is exercised with
  `fnorm` strictly between 10 and 11 (not just at 10 exactly, where the clamp is a no-op regardless
  of `>`/`>=`); the residual-pruning loop removes from the front (FIFO), not an arbitrary index —
  `tests/test_metric.py`, `tests/test_metric_debug_path.py`
- `pgd.py`: `RegexPromptExtractor.extract_features()`'s ~40 feature-dict entries all match their
  exact computed value (one comprehensive golden-vector test on an engineered prompt), not just a
  loose `>= 1.0`/presence check that a dict-key-rename mutation could pass by coincidence (a missing
  key is silently backfilled with `0.0`, which a loose check can miss when the real value happens to
  also look "present enough"); `syn_0`/`syn_1`/`sem_0`/`sem_1`/`sem_2`'s empty-input fallbacks default
  to `0.0`; the `len(w) > 6` "long word" boundary is exercised at exactly 6 characters; `"class " in
  prompt or "def " in prompt` is exercised with each operand isolated (both, neither, and each alone)
  — `tests/pgd/test_regex_prompt_extractor.py`

Known true equivalent mutants (not gaps):
- `d_best = abs(-comps[...]["distance"])` — appears 3 times (`route()`'s metric-update branch,
  `route()`'s controller branch, `batch_route()`'s per-sample loop) — `abs(-x) == abs(x)` for all
  real `x`, so no test can ever distinguish the two.
- `router.py`'s `batch_route()` batch-level gate `enable_metric_update and (_step >= _stride)`
  mutated to `or` — `update_data` can only be non-empty if the per-sample gate already fired, which
  (since `_step` only increases) guarantees `_step >= _stride` is already true by then; the gate's
  only other possible effect (skipping an already-guaranteed-empty pass) is unobservable since the
  per-model loop's own `if not data: continue` would skip it anyway.
- `flush=True` on `energy.py`'s `compute()` debug prints — every test captures stdout via
  `redirect_stdout` to an in-memory buffer, where `flush` has no observable effect on the captured
  text; distinguishing it would require asserting on real OS-level buffering, not this code's logic.
- The unused default string in `security.py`'s `is_offline()`/`redaction_enabled()`
  (`os.environ.get(..., "0")`) — both functions' entire contract is `== "1"`, so any non-`"1"`
  default produces the same result; the raw default string is never itself observable.
- `security.py`'s `git_commit_short()`: `head.split(":", 1)` vs `split(":", 2)` — git forbids `:` in
  ref names, so a real `HEAD` (`ref: refs/heads/<branch>`) always contains exactly one `:`, making
  `maxsplit=1` and `maxsplit=2` produce identical results against any real repository state.
- `security.py`'s `git_commit_short()`: `errors="ignore"` vs `errors="XXignoreXX"` on the HEAD/ref
  file reads — confirmed equivalent only after a real CI run showed it still surviving despite a
  test asserting a real non-`None` result (see `MUTATION_HARDENING_STATUS.md`'s real-CI correction
  note). Python's `errors=` decode-error handler is only consulted when an actual decode error
  occurs; git's HEAD/ref files are always clean ASCII, so they decode successfully regardless of
  what garbage string `errors=` holds.
- `boundary.py`'s `probs = np.exp(arr - u1)` vs `np.exp(arr + u1)` — the immediately-following
  `probs /= probs.sum()` normalization is invariant to any constant additive shift applied uniformly
  to `arr` before the exponential (softmax shift-invariance), and `u1` is a single scalar shared
  across every element of `arr`. Only distinguishable via float overflow at utility magnitudes far
  outside this system's realistic bounded range.
- `coherence.py`'s `sample_weight=w / w.sum()` vs `w * w.sum()` — sklearn's `KernelDensity.fit`
  internally renormalizes `sample_weight` before use, so any positive scalar rescaling of the same
  weight vector produces identical results up to ~1e-15 floating-point noise (verified empirically,
  not assumed).
- `metric.py`'s `metric_matrix()`: the `name=` labels passed to `SymbolicMatrix`/`SymbolicScalar`
  (`"L"`, `"\delta"`, `"I"`) — the function returns only `.evaluate()`'s numpy array; the labeled
  objects (and their `to_latex()` output) are local and discarded, never exposed to any caller.
- `metric.py`'s `_update_cholesky()` recovery delta's upper clamp (`min(..., 1e-1)` vs `1.1`) —
  entering that `except` block at all requires `self.delta <= 0` (any positive delta alone
  guarantees `L @ L.T + delta*I` is positive-definite, so Cholesky can never fail), and the
  recovery step only adds `1e-3`, so the result can never exceed `0.1` regardless of the mutation.
- `metric.py`'s `surrogate_energy`'s `0.5 *` scaling constant vs `1.5 *` — its return value is only
  ever used in `e1 > e0` comparisons; scaling both sides by the same positive constant never
  changes which is larger, so it can't affect control flow or any other observed value.
- `metric.py`'s `batch_update_spd()`'s `fnorm > 10.0` vs `>= 10.0` — exactly at `fnorm == 10.0`, the
  clamp (`self.L *= 10.0/fnorm`) multiplies by exactly `1.0`, a no-op in IEEE754 regardless of
  whether it fires; the only point where `>` and `>=` disagree is also the only point where the
  clamp can't have any effect.
- `pgd.py`'s `aux_*` feature padding (loop bound, key-name mismatches between the write and the
  `.get()` read) — `aux_*` features are permanently `0.0` by construction (nothing else in
  `extract_features()` ever assigns them a nonzero value), and any genuinely-missing key gets
  backfilled with `0.0` anyway by the "ensure all keys present" safety loop, so no combination of
  these mutations can ever produce an observable difference.
- `pgd.py`'s `prag_*` Banach features' defensive `.get(key, default)` reads — each is read back
  immediately after being unconditionally assigned the *same* value the `.get()` default uses, so
  renaming either the assignment's key or the read's key, or changing the read's default, all fall
  through to a value that coincidentally equals what would have been produced anyway.
