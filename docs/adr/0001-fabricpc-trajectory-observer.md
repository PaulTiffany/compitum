# ADR 0001: FabricPC as an optional, observation-only trajectory observer

Status: accepted for the observation tranche only. Branch:
`experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`
(commit `a8de8cbafa5eb00b523f539c340ba81a146aa781`). Nothing in this ADR
authorizes route-affecting integration.

## Authority hierarchy

When sources disagree, precedence is:

1. frozen `v0.2.0` source and tests;
2. schemas and release artifacts attached to that source
   (`reports/release_attestation.json`, `reports/mutation_summary.json`,
   certificate schema tests under `tests/certificates/`);
3. executable notebooks and reproducible reports;
4. the Compitum paper (https://paultiffany.github.io/compitum-paper/);
5. the wiki and explanatory prose.

The wiki contains historical and aspirational API examples. No API is
implemented here because the wiki describes it; every field and formula
referenced below was reconciled against the frozen source. Two examples of
that reconciliation:

- `SwitchCertificate` (src/compitum/router.py) is the strict, frozen route
  certificate: `model`, `utility`, `utility_components`, `constraints`,
  `boundary_analysis`, `drift_status`, `pgd_signature` (16-char truncation in
  `to_json`), `timestamp`, `router_version`. It is not modified by this
  tranche.
- `constraints.shadow_prices` (src/compitum/constraints.py:99) are approximate
  finite-difference boundary diagnostics — literally
  `(utility_competitor - utility_m_star) / 1e-5` under a per-row relaxation
  `b_relaxed[i] += 1e-5`. They are not persistent online dual variables and
  nothing in this tranche pretends otherwise.

## Research hypotheses (falsifiable, ordered)

H1 (this tranche's question, observation-only):

> Do FabricPC inference-trajectory features provide out-of-sample information
> about constrained routing regret, boundary errors, deferral need, or
> impending constraint pressure beyond the information already available to
> Compitum v0.2.0?

H2 (gated on H1; NOT part of this tranche):

> Can a calibrated trajectory signal reduce constrained routing regret or
> improve timely deferral without increasing constraint violations,
> miscalibration, or unacceptable routing latency?

We do not begin by assuming FabricPC improves Compitum. The null hypothesis —
trajectory features add nothing beyond Compitum's own utility components,
uncertainty, boundary gap/entropy, drift state, constraint status, and
FabricPC's terminal energy alone — is the default outcome and is reportable
as a legitimate negative result.

## Nonclaims

- FabricPC does not replace Compitum. Compitum remains responsible for
  feasibility-first selection, Symbolic Free Energy, utilities and calibrated
  predictors, boundary analysis, trust-region and drift control, metric
  adaptation, route selection, deferral/escalation policy, and certificates.
- FabricPC node energy is not Compitum Symbolic Free Energy, is not Compitum
  utility, is not constraint slack, is not `shadow_prices`, and is not a
  future online dual variable. These remain distinct named quantities.
- A local directional gain is not a global Lipschitz constant.
- A negative orientation cosine is not a causal explanation.
- A nonzero mixed finite difference is observable non-additivity only — not a
  Hessian, phase transition, imagination event, or operator-order witness.
- Scalar full-state norm growth is not "instability" until blockwise
  transport and metric choice are audited (the Sketched 72-run sweep showed
  100% of product-norm first-step breaches with 0% hidden-block breaches —
  transport across the node-block interface, not intrinsic expansion).
- Stability is not safety. Local, not global. Surrogate, not first-principles.
  Empirical correspondence must be demonstrated, not narrated.

## Semantic mappings and non-mappings

| Quantity | Owner | Status in this tranche |
| --- | --- | --- |
| FabricPC per-node `energy` | FabricPC | observed, recorded as `node_energy` |
| Compitum Symbolic Free Energy | Compitum | untouched; never conflated with node energy |
| Compitum utility / components | Compitum | untouched |
| constraint slack | Compitum | untouched |
| `constraints.shadow_prices` | Compitum | untouched; finite-difference diagnostics only |
| prospective constraint-pressure estimate | future work | if ever produced, named `trajectory_pressure` / `predicted_constraint_pressure` / `proxy_dual_pressure`; NEVER serialized as `shadow_prices` absent a validated online primal–dual controller |
| directional gain / orientation cosine | trajectory evidence | finite-difference observations with explicit nonclaims embedded in the artifact |
| mixed finite-difference residue | trajectory evidence | non-additivity witness only |

## Version and dependency boundaries

- Compitum core: Python >= 3.9 (frozen `pyproject.toml`). Unchanged.
- FabricPC pin: `https://github.com/trueagi-io/FabricPC` at
  `32ae295182ab944b8f084abaf4a40da2c50bab5f` (release v0.3.2, current `main`
  at pin time). Requires Python >= 3.10 and introduces JAX. Checkout lives
  outside this repository at `C:\src\FabricPC`; it is not vendored, patched,
  or forked.
- FabricPC is therefore strictly optional:
  - no `fabricpc` or `jax` import during ordinary `import compitum`;
  - no FabricPC entry in Compitum's required dependencies;
  - explicit capability detection with a deterministic, governed
    unavailable/refused artifact when the optional dependency is missing;
  - CPU execution on native Windows; no GPU assumptions in baseline tests.
- Sketched's historical receipt (`sketched.fabricpc-install-receipt.v1`,
  FabricPC 0.3.1 at `b6f64adf9314…`) is preserved untouched. This tranche
  writes a new, Compitum-owned receipt for the 0.3.2 pin.
- Trajectory APIs verified present at the pin:
  `fabricpc.utils.dashboarding.inference_tracking.run_inference_with_history`
  (lightweight per-step, per-node metrics: `energy`, `latent_grad_norm`,
  `error_norm`, `z_latent_mean`, `z_latent_std`),
  `run_inference_with_full_history` (full `GraphState` per step; bounded
  audits only), `unstack_inference_history`,
  `summarize_inference_convergence`.

## Data contracts

Core (dependency-free, Python 3.9-compatible, stdlib-only):

- `TrajectoryRequest`: observer inputs (case identifier, seeds, configuration
  mapping, optional embedding/feature payload). Hashable to a canonical
  `config_sha256`.
- `TrajectoryEvidence`: status
  (`observed | unavailable | refused | invalid | failed`), observer identity
  and version, external dependency repository and commit, run/config hash,
  raw-trace artifact reference and hash, terminal summaries, per-step
  summaries, per-node/per-block summaries, energy trajectory, latent-gradient
  trajectory, error trajectory, convergence indicators,
  perturbation/orientation diagnostics where applicable, warnings, explicit
  nonclaims, runtime cost. Non-`observed` statuses carry a structured reason;
  no partially populated "success" objects.

Artifacts (companion to — never a mutation of — the frozen route
certificate):

- schema `compitum.fabricpc-observation/v1`;
- per-run bundle: raw trajectory, trajectory evidence certificate,
  audit/provenance record, experiment manifest, validation summary,
  checksums file (SHA-256 of every member);
- route certificate and trajectory certificate may reference each other by
  hash; `SwitchCertificate` semantics remain byte-identical to v0.2.0.

Failure contract: missing dependency, receipt mismatch (checkout drift),
non-finite trajectory, shape/node-order mismatch, or invalid configuration
each produce a structured governed refusal/failure artifact. No silent
partial success; no uncontrolled crash; no blanket exception swallowing.

## Experiment arms (observation-only pilot)

Identical cases, splits, seeds, model pool, constraints, and baseline state
across arms:

1. frozen Compitum v0.2.0 baseline (no observer / no-op observer —
   behaviorally identical, asserted by test);
2. terminal-only FabricPC evidence (final energy only);
3. FabricPC trajectory-summary evidence (per-step features);
4. shuffled / seed-mismatched trajectory negative control.

Where justified: blockwise vs full-product metrics; additive null vs
nonlinear positive control; PC vs backprop on the same topology; raw vs
compressed feature ablations.

The comparison of record is incremental information beyond Compitum's own
features plus terminal energy — not whether FabricPC predicts anything in
isolation. Strict train/calibration/evaluation separation; no leakage from
route outcomes, future labels, repeated prompts, or seeds.

Reported metrics: constrained routing regret; constraint-violation count and
rate; utility per dollar where applicable; boundary-error prediction;
deferral precision/recall; calibration (Brier-style) error; route-flip rate
in later shadow simulations; p50/p95 observer latency; total routing
overhead; failure/refusal rate; effect sizes with uncertainty. Zero baseline
constraint violations are not traded for average regret improvement.

## Activation gates

Route-affecting use of FabricPC evidence is prohibited in this tranche and
permitted later only if the observation report demonstrates held-out
predictive value beyond existing Compitum + terminal-state features. At that
gate, candidate interventions are ablated separately (calibrated evidence
term; deferral/escalation gate; controller input; predicted
constraint-pressure observer; genuine online primal–dual controller) — never
implemented at once.

## Rollback plan

The experiment always retains a clean path back to `v0.2.0` with no FabricPC
installed:

- all changes live on `experiment/fabricpc-trajectory-observer`; `main` and
  the `v0.2.0` tag are untouched;
- FabricPC lives in an external checkout plus a separate venv; deleting
  `C:\src\FabricPC` and the experiment venv restores a FabricPC-free machine
  state;
- the core `compitum.trajectory` module imports no optional dependency, so
  reverting the branch removes every trace;
- baseline equivalence of the no-op observer is asserted by test, so any
  regression is detectable before activation work begins.

## Provenance of ported instruments

The orientation sensor, second-order (square/commutator) sensor, and
blockwise metric audit are ported/generalized from Sketched
(`C:\src\sketched\verification\tools\`: `fabricpc_orientation_sensor.py`,
`second_order_sensor.py`, `fabricpc_orientation_adapter.py`,
`fabricpc_orientation_block_sweep.py`, `fabricpc_second_order_adapter.py`),
with tests retained and Sketched origin recorded in module headers. Sketched
authority-model assumptions (Principia bindings, `FabricPCGuard.lean`
correspondence) are explicitly NOT imported: no Lean guard is treated as
realized by FabricPC.

Generic engineering patterns (deterministic core vs optional orchestration;
typed intermediate artifacts; raw-audit/consumer-certificate/manifest
separation; governed refusal artifacts; content hashing and bundle
checksums; validation summaries) are independently reimplemented with
Compitum-specific names and schemas. This is a clean-room reimplementation of
general software patterns; no source code, prose, schemas, fixtures, names,
or generated artifacts from the restricted Notebook Compiler repository are
copied.
