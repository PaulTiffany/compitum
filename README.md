# compitum

A production-ready, geometrically-aware AI router with SPD metric learning, constraint-aware
selection (shadow prices), metric-aware KDE coherence, and Lyapunov-stable online updates.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick demo

```bash
compitum route --prompt "Prove the binomial identity using generating functions."
```

## Run tests

```bash
pytest
```

See `configs/` and `examples/` for constraints and a synthetic benchmark.

## Testing Strategy

The project maintains a rigorous, deterministic testing program.

*   **CI Profile (default):** `pytest` runs with `HYPOTHESIS_PROFILE=ci`. This uses a fixed random seed and a moderate number of examples (`max_examples=100`) for fast, repeatable builds.
*   **Mutation Profile:** For mutation testing with `cosmic-ray`, a dedicated `HYPOTHESIS_PROFILE=mutation` is used via a wrapper script. This allows for a different number of examples to balance thoroughness and speed.
*   **Invariants Suite:** A dedicated property-based test suite in `tests/invariants/` validates the core mathematical and operational invariants of the system. These tests are marked with `@pytest.mark.invariants`.

To run the full verification suite, including mutation testing:
```bat
set HYPOTHESIS_PROFILE=ci && ruff check . && mypy src tests && pytest --maxfail=1 --cov=compitum --cov-branch --cov-report=term-missing && del /q session.sqlite 2>nul && cosmic-ray init --force cosmic-ray.toml session.sqlite && cosmic-ray exec cosmic-ray.toml session.sqlite && cr-report session.sqlite
```

## Export Control

This project is open-source research code (MIT). Use is subject to U.S. export laws and sanctions compliance. Do not use if you are a sanctioned person/region.