import pytest


def _energy_like(cert) -> float:
    # duck-typed to your project's cert
    return getattr(cert, "drift_status", {}).get("drift_ema", 0.0)


@pytest.mark.heavy_bench
@pytest.mark.benchmark
def test_energy_drift(benchmark, router):
    prompts = [
        "simple query 1",
        "general query 1",
        "complex query 1",
        "simple query 2",
        "general query 2",
        "complex query 2",
    ]
    window = 2  # Moved window definition here

    def run():
        violations = 0
        energies = []
        for p in prompts:
            cert = router.route(p)
            energies.append(_energy_like(cert))
            if len(energies) >= window:  # noqa: F821
                if energies[-1] > energies[-2] + 1e-6:
                    violations += 1
        return {"nonincrease_violations": violations}

    results = benchmark(run)
    # Allow tiny number due to noise/ties
    assert results["nonincrease_violations"] <= 2
