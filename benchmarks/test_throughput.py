import time

import pytest


@pytest.mark.benchmark
def test_router_throughput_and_latency(benchmark, router):
    prompt = "general query throughput"
    reps = 128

    def run_once():
        start = time.perf_counter()
        for _ in range(reps):
            router.route(prompt)
        end = time.perf_counter()
        total = end - start
        return {"ops_per_sec": reps / max(total, 1e-9), "p50_latency_ms": (total / reps) * 1000.0}

    results = benchmark(run_once)
    assert results["ops_per_sec"] > 10  # very lenient, adjust upward in CI
    assert results["p50_latency_ms"] < 50.0  # lenient; adjust with real data
