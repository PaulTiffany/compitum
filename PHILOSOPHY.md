# Compitum Philosophy

Compitum turns model routing into a principled, auditable control problem. We replace vibes with geometry, constraints, and verifiable artifacts, so choices are explainable today and improvable tomorrow.

## Purpose

- Make routing predictable, safe, and accountable under real constraints.
- Prefer small, composable ideas with clear math over clever stacks that are hard to reason about.
- Ship evidence: every claim connects to tests, certificates, and reports.

## First Principles

- Geometry first
  - Learn an SPD metric `M` so distances, energies, and neighborhoods have meaning.
  - Work in spaces where derivatives are interpretable and stable.

- Constraints are real
  - Cost, latency, region, tools, safety: first‑class, not afterthoughts.
  - Use dual variables (shadow prices) to quantify tradeoffs when constraints bind.

- Determinism by default
  - Same inputs → same route (within tolerance). Randomness is explicit and opt‑in.

- Mechanistic feedback
  - Prefer internal signals (coherence, energy drift) to stochastic “judges” in the tight loop.

- Stability matters
  - Trust‑region updates and bounded change beat brittle cleverness.

- Traceability
  - Tests, JSON/HTML reports, and machine‑readable certificates make decisions auditable.

## Method (what we actually do)

- Metric learning
  - Factorize `M = L^T L` with explicit rank/δ controls; regularize spectra (trace/Frobenius) to avoid collapse.
  - Interpret gradients in the learned basis; keep derivatives smooth and meaningful.

- Trust‑region update
  - Clamp step size under `M`: `||x_{t+1} − x_t||_M ≤ ρ`.
  - Choose a Lyapunov candidate `V` with `ΔV ≤ 0` to prioritize stability.

- Coherence and OOD
  - Use a coherence score (e.g., `∇ log p_M(x)`) and energy drift to flag out‑of‑distribution.
  - Route conservatively when OOD magnitude is high.

- Constraints and duals
  - Solve with Lagrangians; expose shadow prices `λ` (how much value rises if a constraint relaxes).
  - Report residuals `g(m*)` and which constraints are active.

- Calibration and smoothness
  - Use bounded‑Jacobian predictors for quality/latency/cost.
  - Avoid sharp kinks that break determinism and stability.

- Certificates (explainable by construction)
  - Emit inputs, constraints, model choice, `λ`, coherence/energy signals, and sensitivities.
  - Sensitivities via finite differences along principal axes of `M` (LLM‑readable).

## Proof Obligations (claims → tests → artifacts)

- Geometry fidelity → rank/δ ablations → invariants bench + plots.
- Determinism → route‑flip budget under paraphrases → diffable traces + tests.
- Stability → Lyapunov proxy monotonicity → per‑step `ΔV` plots.
- Constraint rationale → shadow prices in certificates → JSON/HTML certificate slices.
- Value under budget → fixed‑WTP + 95% CIs → release tables.

## Discipline Hooks (arXiv alignment)

- cs.LG — robust learning
  - `M = L^T L` with spectral controls; bounded‑Jacobian calibrators; OOD via score gradient.
  - Evidence: rank/δ ablations, invariants/mutation tests, generalization checks.

- cs.CL — explainable routing
  - Deterministic policy with certificates; prompt/tokenization stability; reproducible scripts.
  - Evidence: route‑flip analyses under paraphrases; end‑to‑end reproduction logs.

- cs.SY — control and stability
  - Trust regions, Lyapunov candidates, KKT duals; safety envelopes and constraint tightening.
  - Evidence: stability traces, violation rates, dual variable dynamics.

- stat.ML — estimation and uncertainty
  - Paired tests; bootstrap CIs for fixed‑WTP; multiple comparisons; calibration metrics (ECE/MCE).
  - Evidence: CI tables, per‑baseline win rates, sensitivity to λ.

## Operating Commitments

- Reproducibility
  - Deterministic defaults; pinned deps; profiles with fixed seeds and thread caps.
  - Self‑contained artifacts with SHA‑256 manifests; fetch scripts for large public files.

- Safety
  - No secrets in code/history; no silent network calls.
  - Do not redistribute third‑party datasets; document how to fetch with checksums.

- Minimalism
  - Small, composable parts that pass tests beat clever stacks that don’t.

- Traceability
  - Reports live under `reports/`; certificates are machine‑readable; scripts are one‑command runnable.

## Glossary (LLM‑friendly cues)

- SPD metric `M` — symmetric positive‑definite matrix defining distances and energies.
- Trust‑region radius `ρ` — maximum allowed step under `M` per update.
- Coherence score `∇ log p(x)` — gradient of local density; large magnitude implies OOD.
- Shadow price `λ` — dual variable indicating value of relaxing a constraint.

## One‑Sentence Promise

When gradients, constraints, and coherence disagree, Compitum makes the tradeoffs explicit, bounded, and reproducible.

