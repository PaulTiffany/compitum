---
title: CLI
---

# CLI

## Route

```bash
compitum route --prompt "..." [--constraints configs/constraints_us_default.yaml] \
               [--defaults configs/router_defaults.yaml] [--verbose | --trace]
```

- `--verbose`: prints the full routing certificate JSON.
- `--trace`: prints the full certificate with process signals (constraints, boundary, drift, utility components).

The concise output includes only `model` and `U`.

## Annotated Trace Examples

Example 1: Boundary case with high entropy and low gap

```bash
compitum route --prompt "Prove AM-GM for two variables." --trace
```

Interpretation guide:
- `utility_components`: quality, latency, and cost contributions before constraints
- `constraints`: `feasible`, approximate shadow prices (diagnostic)
- `boundary`: `gap`, `entropy`, `sigma` diagnostics
- `drift`: trust radius and EMA flags

Example 2: Clear winner with strong utility gap

```bash
compitum route --prompt "Summarize the plot of Pride and Prejudice." --trace
```

Notes:
- Use `--constraints`/`--defaults` to point at custom configs.
- Add `--seed` to match synthetic demo fitting deterministically.
