---
title: Certificate Schema
description: Minimal schema and field descriptions for Compitum's routing certificate JSON.
---

# Certificate Schema

The routing certificate is a structured JSON object emitted by the CLI (`--trace`) and available via the API (`cert.to_json()`). It is designed for mechanistic auditing and ablations.

Fields

- model: string — name of selected model
- utility: number — U = alpha·quality - beta_t·latency - beta_c·(cost + model.cost) - beta_d·distance + beta_s·evidence
- utility_components: object — {quality, latency, cost, distance, evidence, uncertainty}
- constraints: object — {status: string, violations: array[string], feasible: boolean, shadow_prices: object[string→number]}
- boundary: object — {winner: string, runner_up: string, utility_gap: number, entropy: number, uncertainty: number, is_boundary: boolean}
- drift: object — {trust_radius: number, drift_ema: number, drift_integral: number, lyapunov_function: number}
- pgd_signature: string — first 16 hex chars of a sha256 digest of the prompt (tamper-evident, not reversible)
- timestamp: number — raw Unix epoch seconds (`time.time()`), not an ISO-8601 string
- router_version: string — the installed `compitum` package version

Example (verified live run, `compitum route --prompt "..." --trace`)

```json
{
  "model": "auto",
  "utility": -0.617274,
  "utility_components": {"quality": 0.598477, "latency": -0.841229, "cost": -2.290030, "distance": -2.299428, "evidence": 0.0, "uncertainty": 0.117909},
  "constraints": {"status": "optimal", "violations": [], "feasible": true, "shadow_prices": {"lambda_0": 0.0, "lambda_1": 0.0, "lambda_2": 0.0, "lambda_3": 0.0}},
  "boundary": {"winner": "auto", "runner_up": "fast", "utility_gap": 0.020410, "entropy": 1.098564, "uncertainty": 0.117909, "is_boundary": false},
  "drift": {"trust_radius": 1.088503, "drift_ema": 0.229943, "drift_integral": 2.184457, "lyapunov_function": 4.771852},
  "pgd_signature": "eff8b649c985f166",
  "timestamp": 1784341869.07,
  "router_version": "0.2.0"
}
```

Minimal JSON Schema (informative) — download: `assets/certificate.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CompitumRoutingCertificate",
  "type": "object",
  "required": ["model", "utility", "utility_components", "constraints", "boundary", "drift"],
  "properties": {
    "model": {"type": "string"},
    "utility": {"type": "number"},
    "utility_components": {
      "type": "object",
      "properties": {
        "quality": {"type": "number"},
        "latency": {"type": "number"},
        "cost": {"type": "number"},
        "distance": {"type": "number"},
        "evidence": {"type": "number"},
        "uncertainty": {"type": "number"}
      },
      "additionalProperties": {"type": "number"}
    },
    "constraints": {
      "type": "object",
      "required": ["feasible", "shadow_prices"],
      "properties": {
        "status": {"type": "string"},
        "violations": {"type": "array", "items": {"type": "string"}},
        "feasible": {"type": "boolean"},
        "shadow_prices": {
          "type": "object",
          "additionalProperties": {"type": "number"}
        }
      }
    },
    "boundary": {
      "type": "object",
      "properties": {
        "winner": {"type": "string"},
        "runner_up": {"type": "string"},
        "utility_gap": {"type": "number"},
        "entropy": {"type": "number"},
        "uncertainty": {"type": "number"},
        "is_boundary": {"type": "boolean"}
      },
      "additionalProperties": false
    },
    "drift": {
      "type": "object",
      "properties": {
        "trust_radius": {"type": "number"},
        "drift_ema": {"type": "number"},
        "drift_integral": {"type": "number"},
        "lyapunov_function": {"type": "number"}
      },
      "additionalProperties": false
    },
    "pgd_signature": {"type": "string"},
    "timestamp": {"type": "number"},
    "router_version": {"type": "string"}
  },
  "additionalProperties": false
}
```

This embedded copy now matches the downloadable `docs/_extra/assets/certificate.schema.json` file exactly (that file was already correct -- this markdown page's prose/example had drifted out of sync with it).

Notes

- Keys may include additional utility components in future versions; consumers should treat unknown utility component keys as additive numeric terms.
- Shadow prices (constraints.shadow_prices) are an approximate local diagnostic: for each constraint i, we relax b[i] by a small epsilon (1e-5), check if any competitor becomes both feasible and higher utility than the chosen model, and if so record (U_competitor - U_star)/epsilon as `lambda_i`. Zeros can occur for near-binding constraints that don’t flip the argmax at that epsilon.
- Values depend on the epsilon scale and constraint units; compare ranks/magnitude qualitatively, not absolutely across deployments.

