---
title: Performance Notes
description: How to run and interpret Compitum’s built-in benchmarks, capacity knobs, and practical tips.
---

# Performance Notes

Benchmarks

- Pytest benchmarks are included and run in CI. Profiles:
  - SMOKE/DEV/FULL examples in README (Windows). See “Benchmark Profiles”.
  - Key tests: energy drift, constraint violation rate, SPD metric bounds, throughput/latency.

How to run (quick)

```bat
pytest -q -k "test_energy_drift or test_constraint_violation_rate or test_spd_det_and_trust_radius_bounds"
```

Interpreting Metrics

- Throughput/latency: `test_router_throughput_and_latency`
- Stability: `test_spd_det_and_trust_radius_bounds`, `test_energy_drift`
- Constraint health: `test_constraint_violation_rate`
- Utility efficiency: `test_iso_utility_savings_vs_fixed_best`

Capacity Knobs

- Feature/metric complexity: `D`, `rank`, `delta`
- Coherence: `k` (neighbors)
- Update cadence: `update_stride`
- Boundary thresholds: `gap_threshold`, `entropy_threshold`, `sigma_threshold`

Tips

- Set `OMP_NUM_THREADS=1` (and similar) for stable microbenchmarks (see README “A Note on Threading”).
- Profile only the portion you change; avoid conflating IO and CPU time.

References

- README (Benchmark Profiles)
- {doc}`PEER_REVIEW`
- {doc}`Panel-Summary`

