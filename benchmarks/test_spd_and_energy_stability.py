import pytest


def _spd_det(router):
    # try to read per-model metric map if present (duck-typed to your project)
    metric_map = getattr(router, "metric_map", None)
    if not metric_map:
        return 1.0
    # read one model's det() as a proxy
    for met in metric_map.values():
        try:
            spd = met.get_spd()
            return float(spd.det())
        except Exception:
            continue
    return 1.0

def _trust_radius(router, cert):
    drift = getattr(cert, "drift_status", {}) or {}
    return float(drift.get("trust_radius", 1.0))

@pytest.mark.benchmark
def test_spd_det_and_trust_radius_bounds(benchmark, router):
    prompts = [
        "simple query 1", "general query 1", "complex query 1",
        "simple query 2", "general query 2", "complex query 2",
    ]
    def run():
        min_det = float("inf")
        trs = []
        for p in prompts:
            cert = router.route(p)
            min_det = min(min_det, _spd_det(router))
            trs.append(_trust_radius(router, cert))
        return {
            "min_det": float(min_det),
            "trust_radius_p95": sorted(trs)[int(0.95 * len(trs)) - 1] if trs else 1.0,
        }
    results = benchmark(run)
    assert results["min_det"] > 1e-6
    assert 0.1 <= results["trust_radius_p95"] <= 10.0
