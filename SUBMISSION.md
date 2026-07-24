# Submission: Bellman-Consistent Discrete Shadow-Charge Pricing

**Canonical repository:** `git@github.com:PaulTiffany/compitum.git`
**Canonical branch:** `experiment/fabricpc-trajectory-observer`
**Frozen tag:** `fabricpc-compitum-shadow-pricing-v1`
**Frozen commit:** `617f8979daa921d326301266e55740c0746ab95c`

This document is the entry point to a completed, closed research
program. It is readable without traversing any of the seven underlying
tranche reports. Nothing described here has been merged into `main` or
activated in production Compitum routing.

---

## 1. Contribution

For discrete resource-constrained routing, a local scalar shadow price
is insufficient to price actions that cross multiple marginal-value
regions. We instead charge each action by the continuation value of the
resource it consumes.

```text
C_t(a) = V_{t+1}(B_t, q_{t+1}) - V_{t+1}(B_t - c_t(a), q_{t+1})
```

where `V_{t+1}` is the exact Bellman continuation value, `B_t` is
remaining budget entering step `t`, `c_t(a)` is action `a`'s resource
consumption net of this step's own replenishment, and `q_{t+1}` is the
belief entering the continuation (posterior after this step's own
observation, projected forward one transition — not the belief prior to
observing, and not the raw, unprojected posterior).

Maximizing

```text
u_t(a) - C_t(a)
```

is equivalent, up to an action-independent additive constant
(`V_{t+1}(B_t, q_{t+1})`, the same term for every candidate action), to
full Bellman-Q selection:

```text
Q(a) = u_t(a) + V_{t+1}(B_t - c_t(a), q_{t+1})
```

so `argmax_a [u_t(a) - C_t(a)] == argmax_a Q(a)`. This is proven, not
merely observed: with the exact belief, our implementation's selected
action is required — and verified — to be bit-identical, at every step,
to the literal Bellman-optimal online policy.

## 2. Decisive evidence

**Translation correctness (Gate A-prime):** exact bit-identical
agreement between the discrete shadow-charge policy and the true online
Bellman optimum. Zero mismatches across the original 35-sequence
held-out test set and five independent robustness seeds
(`4242, 1, 2, 3, 100`).

**Economic recovery (tranche 6.5):**

| quantity | value |
| --- | --- |
| exact-belief shadow-charge regret | **0.000** |
| frozen pacing regret | 1.829 |
| scalar-price (linear `lambda * consumption`) regret | 1.943 |
| violations | 0 |

The exact-belief shadow-charge policy recovers the *entire* economic gap
over pacing; the scalar-price translation this work corrects does not
beat pacing at all.

**Belief-sensitive validation (tranche 7):**

| quantity | value |
| --- | --- |
| recoverable gap (pacing regret − exact-belief regret) | 0.371 |
| ridge regret | **0.000** |
| FabricPC (predictive coding) regret | 0.314 |
| FabricPC captured fraction of recoverable gap | 15.4% |
| violations (all arms) | 0 |

Ridge — an ordinary linear regression on the same declared window
features — recovers the exact-belief result exactly, proving the
belief-estimation task is genuinely learnable and that the recoverable
gap is real, not an artifact of the oracle.

## 3. FabricPC finding

> A genuinely trained FabricPC model recovered belief information above
> the naive baseline, but the fixed small topology was substantially
> less accurate than ridge regression on the same declared history and
> did not clear the regret gate.

FabricPC's belief-prediction MSE was approximately 600x worse than
ridge's on the same held-out test features, and its predictive-coding
and same-topology-backprop training runs tied with each other exactly
(0.314 mean regret, zero variance in their paired difference across all
35 test sequences).

We do **not** say FabricPC improved Compitum. We do **not** say
predictive coding failed generally — the tie with backprop under an
identical, small, fixed topology and bounded training budget (three
seeds, no architecture or hyperparameter search) indicates an
architecture/representation bottleneck under the tested design, not a
demonstrated difference between the two learning rules.

## 4. Reproduction

**Environment.** Two virtual environments coexist in the canonical
repository:

- `.venv` — plain Python 3.11, `compitum` only (numpy-based
  `regret_lab` code; no FabricPC/JAX dependency).
- `.venv-fabricpc` — Python 3.11 with FabricPC and JAX installed
  editable, plus `compitum`.

**Pinned FabricPC.** Repository `https://github.com/trueagi-io/FabricPC.git`,
commit `32ae295182ab944b8f084abaf4a40da2c50bab5f` (tag `v0.3.2`), external
checkout, never vendored or patched. Receipt:
`experiments/fabricpc/fabricpc_install_receipt.json`
(`receipt_sha256: 6f33dfc44ac0e69747e2b66a07ec83965cd4d01f45a885c83e29c07861410264`).

**Canonical commands** (run from the repository root, on the tagged
commit):

```bash
# Tranche 6.5's shadow-charge pilot (no FabricPC training required to re-verify Gate A-prime)
.venv-fabricpc/Scripts/python.exe experiments/fabricpc/tranche6_5/run_shadow_charge_pilot.py

# Tranche 7's Gate 0 identifiability check (pure regret_lab, no FabricPC/JAX needed)
.venv/Scripts/python.exe experiments/fabricpc/tranche7/run_gate0_identifiability.py

# Tranche 7's ten-arm pilot (trains FabricPC; ~60 seconds)
.venv-fabricpc/Scripts/python.exe experiments/fabricpc/tranche7/run_ten_arm_pilot.py

# Minimal frozen demonstration (no retraining, no FabricPC/JAX required)
.venv/Scripts/python.exe experiments/fabricpc/submission_demo.py

# Full test suite / type-check / lint
.venv/Scripts/python.exe -m pytest -q -m "not routerbench and not heavy_bench"
.venv/Scripts/python.exe -m mypy -p compitum --ignore-missing-imports --hide-error-context
.venv/Scripts/python.exe -m ruff check .
```

**Expected output.** The pilot scripts write JSON reports to
`experiments/fabricpc/tranche6_5/artifacts/` and
`experiments/fabricpc/tranche7/artifacts/` matching the values quoted in
section 2 above (exact regret values are deterministic given the fixed
seeds declared in each script). The test suite reports 905 passed, 1
skipped (a documented pre-existing Windows subprocess/asyncio issue
unrelated to this work), 3 deselected (routerbench, out of scope);
mypy and ruff report no issues.

**Report and checkpoint hashes** (SHA-256, computed at the frozen
commit): see `handoff/bellman_shadow_pricing/artifact_manifest.json`
for the complete, machine-readable list. FabricPC training-run
checkpoint hashes are recorded per-seed inside
`experiments/fabricpc/tranche7/artifacts/ten_arm_pilot_report.json`
(`fabricpc_training.backprop_runs[].checkpoint_hash`,
`fabricpc_training.pcn_runs[].checkpoint_hash`).

**Exact tagged source revision:** `617f8979daa921d326301266e55740c0746ab95c`,
tag `fabricpc-compitum-shadow-pricing-v1`.

## 5. Artifact map

| artifact | path |
| --- | --- |
| Final cross-tranche synthesis | `experiments/fabricpc/FINAL_SYNTHESIS.md` |
| Tranche 6.5 report (shadow-charge correction) | `experiments/fabricpc/tranche6_5/REPORT.md` |
| Tranche 7 report (belief-sensitive validation) | `experiments/fabricpc/tranche7/REPORT.md` |
| Gate 0 identifiability JSON | `experiments/fabricpc/tranche7/artifacts/gate0_report.json` |
| Ten-arm pilot JSON | `experiments/fabricpc/tranche7/artifacts/ten_arm_pilot_report.json` |
| ADR 0008 (shadow-charge curve) | `docs/adr/0008-bellman-consistent-shadow-price-curve.md` |
| ADR 0009 (belief-sensitive validation) | `docs/adr/0009-belief-sensitive-shadow-charge-validation.md` |
| Core shadow-charge implementation | `src/compitum/regret_lab/belief_action_pricing.py`, `src/compitum/regret_lab/belief_action_pricing_v2.py` |
| Exact online comparator | `src/compitum/regret_lab/belief_online_optimum.py`, `src/compitum/regret_lab/belief_online_optimum_v2.py` |
| Trained FabricPC model | `experiments/fabricpc/tranche6/fabricpc_belief_model.py` |
| Dependency receipt | `experiments/fabricpc/fabricpc_install_receipt.json` |

Portable, code-free provenance capsule (for later extraction into a
standalone package, Sketched, or notebook_compiler without disturbing
this record): `handoff/bellman_shadow_pricing/`.
