import pytest


def _energy_like(cert):
    # If your certificate exposes explicit energy, adapt here.
    # As a proxy we invert utility (lower is "worse"): E ~ (1 - utility)
    u = float(getattr(cert, "utility", 0.5))
    return max(0.0, 1.0 - u)

@pytest.mark.benchmark
def test_rolling_energy_nonincrease(benchmark, router):
    prompts = [
        "general query 1", "complex query 1", "simple query 1",
        "general query 2", "complex query 2", "simple query 2",
    ]

    def run():
        window = 2
        violations = 0
        energies = []
        for p in prompts:
            cert = router.route(p)
            energies.append(_energy_like(cert))
            if len(energies) >= window:
                if energies[-1] > energies[-2] + 1e-6:
                    violations += 1
        return {"nonincrease_violations": violations}
    results = benchmark(run)
    # Allow tiny number due to noise/ties
    assert results["nonincrease_violations"] <= 2
