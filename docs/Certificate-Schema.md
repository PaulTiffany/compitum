---
title: Certificate Schema
description: Minimal schema and field descriptions for Compitum's routing certificate JSON.
---

# Certificate Schema

The routing certificate is a structured JSON object emitted by the CLI (`--trace`) and available via the API (`cert.to_json()`). It is designed for mechanistic auditing and ablations.

Fields

- model: string — name of selected model
- utility: number — total utility U = quality - (lambda × cost) plus any additional terms included by energy
- utility_components: object — additive components, typically {quality, latency, cost}
- constraints: object — {feasible: boolean, shadow_prices: object[string→number]}
- boundary: object — {gap: number, entropy: number, sigma: number}
- drift: object — {trust_radius: number, ema: number, integral: number}

Example

```json
{
  "model": "fast",
  "utility": 0.423,
  "utility_components": {"quality": 0.61, "latency": -0.07, "cost": -0.12},
  "constraints": {"feasible": true, "shadow_prices": {"lambda_0": 0.0, "lambda_1": 0.13, "lambda_2": 0.0}},
  "boundary": {"gap": 0.03, "entropy": 0.58, "sigma": 0.11},
  "drift": {"trust_radius": 0.8, "ema": 0.76, "integral": 0.12}
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
        "cost": {"type": "number"}
      },
      "additionalProperties": {"type": "number"}
    },
    "constraints": {
      "type": "object",
      "required": ["feasible", "shadow_prices"],
      "properties": {
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
        "gap": {"type": "number"},
        "entropy": {"type": "number"},
        "sigma": {"type": "number"}
      },
      "additionalProperties": false
    },
    "drift": {
      "type": "object",
      "properties": {
        "trust_radius": {"type": "number"},
        "ema": {"type": "number"},
        "integral": {"type": "number"}
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

Notes

- Keys may include additional utility components in future versions; consumers should treat unknown utility component keys as additive numeric terms.
- Shadow prices (constraints.shadow_prices) are an approximate local diagnostic: for each constraint i, we relax b[i] by a small epsilon (1e-5), check if any competitor becomes both feasible and higher utility than the chosen model, and if so record (U_competitor - U_star)/epsilon as `lambda_i`. Zeros can occur for near-binding constraints that don’t flip the argmax at that epsilon.
- Values depend on the epsilon scale and constraint units; compare ranks/magnitude qualitatively, not absolutely across deployments.

