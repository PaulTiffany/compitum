# FabricPC × Compitum observation-only tranche: report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`
(`a8de8cbafa5eb00b523f539c340ba81a146aa781`). FabricPC pin: v0.3.2 at
`32ae295182ab944b8f084abaf4a40da2c50bab5f` (external checkout `C:\src\FabricPC`;
not vendored, not patched). Nothing in this tranche affects route selection,
`main`, the `v0.2.0` tag, the frozen `SwitchCertificate` schema, or upstream
FabricPC.

## What was built

- `compitum.trajectory`: dependency-free core (protocol, governed evidence,
  no-op observer, capability/receipt checks, evidence assembly, artifact
  bundles) — Python 3.9, stdlib-only, 100% line+branch covered, mypy-strict
  clean; `import compitum` provably never imports `jax`/`fabricpc`
  (clean-interpreter test).
- Ported Sketched instruments with provenance and tests: orientation audit,
  second-order square/commutator audit, generalized blockwise metric audit.
- `experiments/fabricpc/`: JAX-side probe (lightweight
  `run_inference_with_history` for ordinary observation; full history only
  for bounded paired audits), Compitum-owned install receipt, baseline
  capture, observation-only pilot.

## Pilot results (N=24 synthetic cases, toy router, toy PC graph)

**Core question — do trajectory features add held-out information about a
deferral-need proxy (top-2 utility margin) beyond frozen Compitum features?**

| arm | held-out MSE |
| --- | --- |
| 1: baseline (frozen certificate features) | **0.00756** |
| 2: + FabricPC terminal energy | 0.01064 |
| 3: + trajectory summary features | 0.01008 |
| 4: + shuffled/case-permuted control features | 0.00769 |

**Negative result.** Every FabricPC arm is worse than baseline; genuine
trajectory features (arm 3) perform like the destroyed-information control
(arm 4) modulo noise. On this toy setup, trajectory evidence carries no
incremental information — which is the honest expected null: the observed
graph's clamps see only two bounded embedding coordinates, and the routing
margin is a deterministic function of features the baseline already has.
Caveats: N=23 usable cases, one train/test split, proxy target rather than a
realized outcome label. This pilot validates the pipeline and the
experimental machinery, not the hypothesis; the H1 gate stays **closed**.

**Governed failure path, exercised for real:** the deliberately non-finite
case produced `status=invalid`,
`reason="steps[0]['source'].energy is missing or non-finite"`, a complete
bundle with validation `ok=false` — no crash, no partial success.

**Instruments at the 0.3.2 pin (cross-version reproduction of the Sketched
0.3.1 findings):**

- orientation: 1 gain-breach candidate in 12 transitions, first step only, no
  orientation reversal;
- blockwise: full product-norm breach with the perturbed hidden block
  contractive and latent-block emergence — the transport lesson reproduced
  (a scalar gain would have misclassified this as intrinsic instability);
- second-order: linear graph residue exactly 0 (additive null); sigmoid graph
  max residue `1.7851115632e-3` vs Sketched's recorded `1.78511156e-3` at
  0.3.1 — identical to 9 significant figures under the same seeds.

**Latency:** observe p50 163 ms, p95 234 ms, max 1.48 s (one-off JIT
warmup); frozen routing p50 4.8 ms. Observation-only cost is off the routing
path; any future inline use would multiply routing latency ~30-50x per case
at this graph size and would need explicit budgeting.

**Baseline integrity:** frozen suite in the worktree: 368 passed + the known
pre-existing worktree-only `git_commit_short` failure (documented in
`reports/mutation_clean_verification_security.json` at release time). With
the new module: 417 passed, 100.00% coverage, mypy `--strict` clean, ruff
clean. The no-op observer arm is byte-identical routing (timestamp aside),
asserted by test.

## Provenance notes

- Sketched (`C:\src\sketched`): orientation/second-order sensors and the
  blockwise audit ported with tests and headers recording origin; the
  Principia/Lean authority model and `FabricPCGuard.lean` correspondence
  were deliberately NOT imported. Sketched's historical FabricPC 0.3.1
  receipt is untouched.
- Notebook Compiler (`C:\src\notebook_compiler`): inspected for general
  patterns only (typed artifacts, governed refusals, manifest/checksum
  bundles, claim firewall). All Compitum artifact schemas and code here are
  independent clean-room implementations with Compitum-specific names; no
  source, prose, schemas, fixtures, or branding copied.

## Open items and smallest defensible next step

Unresolved: whether any real routing workload produces trajectory-observable
structure (this pilot cannot answer that); whether the energy trajectory of
a PC graph trained on routing-relevant signals (rather than a fixed toy
graph) correlates with boundary errors; per-node full-state audits at scale.

Smallest defensible next activation-adjacent step (still observation-only):
train a small FabricPC predictor on a real labeled routing dataset
(RouterBench-style outcome labels), regenerate the same four arms with
realized-outcome targets instead of router-derived proxies, and pre-register
the arm-3-must-beat-arm-4 criterion before running. Route-affecting use
remains gated on that evidence.
