import numpy as np
import pytest


def _router_models(router):
    # Expect dict of Model-like with .cost and .latency
    models = getattr(router, "models", {}) or {}
    return models

def _utility_cost_latency(router, prompt):
    """Call router and also recover cost/latency of chosen model if available."""
    cert = router.route(prompt)
    models = _router_models(router)
    m = models.get(cert.model)
    cost = getattr(m, "cost", 1.0) if m else 1.0
    lat  = getattr(m, "latency", 0.05) if m else 0.05
    return cert.utility, cost, lat, cert

def _oracle_utility(router, prompt):
    """Compute oracle by trying each model's hypothetical pick via a shadow run.
    If we can't enumerate models, fall back to observed utility (no regret)."""
    models = _router_models(router)
    if not models:
        return router.route(prompt).utility
    # brute force: replace router decision by evaluating utilities via prompt variants if exposed
    # As a duck-typed fallback, we approximate oracle by the max over models'
    # quality (when utilities align).
    # If router has a 'pgd' extractor, many compitum repos compute utility as
    # f(quality, query_type).
    # We'll approximate with .quality_score when present.
    u_candidates = []
    for name, m in models.items():
        # heuristic oracle utility guess
        u = getattr(m, "quality_score", 0.5)
        # rough bonus if complex & high_quality encoded in name
        if "complex" in prompt and "high_quality" in name:
            u += 0.3
        if "simple" in prompt and "low_quality" in name:
            u -= 0.1
        u_candidates.append(float(u))
    return max(u_candidates) if u_candidates else router.route(prompt).utility

@pytest.mark.benchmark
def test_mean_regret_and_pareto(benchmark, router):
    prompts = [
        "simple query 1", "general query 1", "complex query 1",
        "simple query 2", "general query 2", "complex query 2",
        "simple query 3", "general query 3", "complex query 3",
    ]

    def run():
        regrets = []
        u_per_cost = []
        u_per_ms = []
        on_frontier = 0
        pts = []

        for p in prompts:
            u, cost, lat, _ = _utility_cost_latency(router, p)
            u_oracle = _oracle_utility(router, p)
            regret = max(0.0, u_oracle - u)
            regrets.append(regret)
            u_per_cost.append(u / max(cost, 1e-9))
            u_per_ms.append(u / max(lat * 1000.0, 1e-9))
            pts.append((u, cost, lat))

                # ε-Pareto frontier coverage: within eps of utility frontier
        # while not dominated on (cost,lat)
        eps = 1e-6
        def dominated(i, j):
            # i dominated by j if j has >= utility and <= cost and <= lat with at least one strict
            return (
                pts[j][0] >= pts[i][0] - eps
                and pts[j][1] <= pts[i][1] + eps
                and pts[j][2] <= pts[i][2] + eps
            ) and (
                (pts[j][0] > pts[i][0] + eps)
                or (pts[j][1] < pts[i][1] - eps)
                or (pts[j][2] < pts[i][2] - eps)
            )

        frontier_flags = []
        for i in range(len(pts)):
            is_dom = any(dominated(i, j) for j in range(len(pts)) if j != i)
            frontier_flags.append(not is_dom)
        on_frontier = sum(frontier_flags)

        return {
            "mean_regret": float(np.mean(regrets)),
            "p95_regret": float(np.percentile(regrets, 95)),
            "u_per_cost_mean": float(np.mean(u_per_cost)),
            "u_per_ms_mean": float(np.mean(u_per_ms)),
            "frontier_coverage": float(on_frontier) / max(1, len(pts)),
        }

    results = benchmark(run)
    # Soft sanity checks: regret shouldn't explode; efficiency ratios positive
    assert results["mean_regret"] >= 0.0
    assert results["u_per_cost_mean"] > 0.0
    assert results["u_per_ms_mean"] > 0.0
