# Compitum Benchmarks

These suites quantify *why Compitum is better* in terms of decision quality, efficiency, stability, and governance.
They run as standard `pytest` tests and integrate with `pytest-benchmark`.

## Quick start

```bash
# from repo root
pytest benchmarks -q

# benchmark tables only
pytest benchmarks -m benchmark --benchmark-only

# store JSON for dashboards
pytest benchmarks --benchmark-json=artifacts/benchmarks.json --benchmark-only
```

## What gets measured

- **Regret & Uplift**: mean regret vs. oracle, uplift over random/fixed routers.
- **Pareto Efficiency**: utility per $ / per ms, and ε-Pareto frontier coverage.
- **Constraint Compliance**: violation rate across sampled tasks.
- **Metric & Energy Health**: SPD det > ε, monotone energy (tolerant), trust-radius bounds.
- **Throughput & Latency**: ops/sec and latency quantiles for `router.route`.

> All suites are duck-typed: they will use your real `CompitumRouter` if present.
> If not, they fallback to the light mock fixtures used in `tests/test_orchestration_performance.py`.
