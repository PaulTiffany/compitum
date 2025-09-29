import pytest

# Import the data generation function from the new module
from benchmarks.iso_utility_data_generator import get_iso_utility_data_for_summary


@pytest.mark.benchmark
def test_iso_utility_savings_vs_fixed_best(benchmark, router, fixed_best_router):
    # The actual data generation logic is now in get_iso_utility_data_for_summary
    out = get_iso_utility_data_for_summary(router, fixed_best_router)

    # Guarantees:
    # - Cost: non-negative savings (Compitum shouldn't cost more than "always pick best")
    # - Latency: allow tiny negative due to routing-time jitter in mocks (<= 0.5%)
    assert out["tau_0.6"]["savings_cost_pct"] >= 0.0
    assert out["tau_0.6"]["savings_e2e_pct"] >= -0.5

    # To avoid PytestBenchmarkWarning, we can benchmark a dummy function.
    benchmark(lambda: None)
