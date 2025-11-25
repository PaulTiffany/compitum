% Data Policy & Security

Scope
- Strong separation of concerns: core library and CI run offline by default; data acquisition is manual and opt-in.
- Reproducible, conservative results: artifacts include uncertainty and attestation with hashes and environment info.

Acquisition
- Prefer CSV/JSON inputs you control. Avoid serialized Python objects (pickle) due to code execution risk.
- Materials Project access is manual: use `MP_API_KEY` in the `materials_audit` workflow or local sessions only.
- RouterBench dataset remains gated; helper script fetches from the canonical source and warns about `.pkl` risk.

CI/CD boundaries
- Default CI does not download external data or call external APIs.
- Manual workflows (`workflow_dispatch`) accept user-provided CSV paths and secrets, and upload artifacts (no Releases).
- Attestation JSON (hashes, env, commit) accompanies outputs for review reproducibility.

Security
- Bandit scans `src/compitum`, `tools`, `examples`, `scripts` in CI.
- We do not unpickle arbitrary files in CI; `.pkl` files are advisory only and never committed.
- Secrets are scoped to manual jobs and not echoed in logs; outputs exclude secrets.

Recommended layout
- `data/` for local inputs and products (ignored by Git). Suggested subfolders:
  - `data/external/` for third-party CSVs
  - `data/samples/` for small examples (checked in)
- Avoid committing third-party data; instead, reference provenance in attestation or a small README next to the file.

Reproducibility
- Use the offline Matbench workflow to calibrate and evaluate from a repo CSV path, producing:
  - Calibration JSON, Regret CSV/JSON (+ groups/budget optional), Baseline CSV/JSON, Layers CSV/JSON, and Attestation JSON.

