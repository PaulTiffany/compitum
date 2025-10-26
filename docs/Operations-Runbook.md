---
title: Operations Runbook
description: Practical guidance for logging, monitoring, alerting, and run/rollback procedures using Compitum’s mechanistic signals.
---

# Operations Runbook

Overview

- Compitum emits structured certificates per decision (CLI `--trace`, API `cert.to_json()`). These fields are designed to map directly to logs and metrics for SRE/ops.

Logging (Structured)

- Log per decision the following fields (JSON):
  - `model`, `utility`, `utility_components.quality`, `utility_components.cost`
  - `constraints.feasible`, `constraints.shadow_prices`
  - `boundary_analysis.gap`, `boundary_analysis.entropy`, `boundary_analysis.sigma`
  - `drift_status.trust_radius`, `drift_status.ema`
- Example: see `examples/cert_to_logging.py`.

Metrics (Suggested)

- Gauges/histograms:
  - Utility (U), quality, cost
  - Gap, entropy, sigma
  - Trust radius, EMA
- Counters:
  - Feasible/infeasible decisions
  - Deferrals (if policy triggers on ambiguity)
- Derived:
  - “At frontier” rate (gap ~ 0)
  - Active-constraint count (nonzero shadow prices)

Alerts (Initial Thresholds)

- Constraint compliance < 99.9% over 5–15 min window
- Prolonged high ambiguity:
  - Gap < 0.02 and Entropy > 0.8 for > 1% of decisions in 15 min
- Drift tightness:
  - Trust radius persistently low (e.g., < 0.2) beyond N decisions

Dashboards (Minimal)

- Efficiency: U, quality, cost (p50/p90)
- Ambiguity: gap, entropy (p50/p90), at-frontier rate
- Constraints: feasible rate, active constraint count, top shadow prices
- Drift: trust radius and EMA trend

Run Procedures

- Standard run:
  - Use fixed configs (defaults, constraints)
  - Log certificate JSON for each decision
  - Export metrics from logs via your pipeline (e.g., ELK/OTel)
- Rollback:
  - Revert to previous frozen config (tagged release)
  - Reduce update stride or tighten trust radius if instability appears

Knobs (Tuning)

- `lambda` (WTP): cost sensitivity
- Metric params: `D`, `rank`, `delta` (stability)
- Boundary thresholds: `gap_threshold`, `entropy_threshold`, `sigma_threshold`
- Update cadence: `update_stride`

SRE Tests (Smoke)

- Route a fixed prompt set and assert:
  - No infeasible certificates
  - U, gap, entropy within expected bands
  - Logs parse as valid JSON; metrics exporter sees fields

References

- {doc}`Certificate-Schema`
- {doc}`PEER_REVIEW` (Routing Certificate)
- {doc}`Panel-Summary`
- `examples/cert_to_logging.py`

