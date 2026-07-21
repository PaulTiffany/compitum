---
title: Compliance Overview
description: Data flow, licensing, privacy, security posture, and auditability for compliance teams.
---

# Compliance Overview

Summary

- Compitum is a local routing engine. It operates offline by default, uses locally cached licensed inputs for evaluation, and emits structured, auditable artifacts. No judge-based calls or dataset fetching are performed by default scripts.

Data Flow (Default)

- Inputs: user prompts (free text), model metadata, configuration (including constraints), and evaluation CSVs (when generating evidence).
- Processing: local computation only; no network by default. Utility computed as U = performance − lambda · cost under constraints (A x ≤ b).
- Outputs: local JSON/CSV/HTML in `reports/` and `docs/`. A SHA-256 artifact manifest is generated.

Privacy & Data Handling

- PII: Prompts may contain free text; by policy, users should avoid PII. Compitum does not retain or transmit PII by default.
- Retention: Artifacts are local; retention is under the user’s control. No telemetry is collected.
- DSR/Deletion: Delete local artifacts in `reports/` and cache folders to satisfy internal retention requirements.

Security Posture

- Offline by default: no judge calls; scripts do not auto-fetch datasets.
- Integrity: mutation testing — every release-critical module fully classified, with only 2 documented, accepted defensive survivors remaining across the matrix (see `MUTATION_HARDENING_STATUS.md`); 100% line+branch coverage; lint/type/security checks (ruff/mypy/bandit) are clean; docs build without warnings.
- Supply chain: versions pinned in `pyproject.toml`; optional SBOM and lock snapshot targets (`make sbom`, `make lock`). Packaging excludes `*.mp4` and `*.sqlite`.
- Secrets hygiene: `.gitignore` patterns and optional local secrets scanning (see CONTRIBUTING).

Licensing & Third Parties

- Project license: MIT (see `LICENSE`).
- RouterBench: inputs follow upstream licenses; we do not redistribute proprietary datasets. Evaluation uses local caches.
- Tokenization/backends (tiktoken/tokencost/HF) are optional libraries used locally; no calls are made upstream by default.

Policy Controls via Constraints

- Constraints (A x ≤ b) encode hard limits (e.g., region availability, rate caps). Infeasible routes are rejected by construction.
- The routing certificate exposes `constraints.feasible` and `constraints.shadow_prices` for each decision, enabling audit trails and policy reviews.

Auditability & Evidence

- Routing certificate: structured JSON for every decision (CLI: `--trace`, API: `cert.to_json()`), with utility components, constraints, boundary diagnostics, and drift monitors.
- Evidence pack: per-baseline win rates, frontier gap (with 95% CIs), per-task summaries, and a panel summary.
- Artifact manifest: `reports/artifact_manifest.json` lists key outputs with SHA-256 checksums.

Compliance Checklists (Quick)

- GDPR/Data Minimization: avoid PII in prompts; no network; artifacts local; delete artifacts to satisfy retention.
- SOC2-style hygiene: source control (GitHub), CI quality gates (tests/lint/types/security), artifact checksums, optional SBOM/lock, no auto-network.
- Export/Use: open-source research code (MIT); respect export control laws; no redistribution of proprietary datasets.

Operational Notes

- Offline reproduction: see {doc}`PEER_REVIEW` and {doc}`Artifact-README`.
- Fairness and evaluation: see {doc}`RouterBench-Fairness` and {doc}`Panel-Summary`.
- Certificate schema: see {doc}`Certificate-Schema` (download `assets/certificate.schema.json`).

Contact

- For compliance documentation requests (e.g., SBOM format, license reports, no-network CI logs), please open an issue. We will collaborate without changing scientific pipelines.

