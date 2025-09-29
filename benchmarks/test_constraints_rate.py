import pytest


def _is_violation(cert)->bool:
    c = getattr(cert, "constraints", {}) or {}
    feasible = c.get("feasible", True)
    return not bool(feasible)

def _route(router, prompt:str):
    return router.route(prompt)

@pytest.mark.benchmark
def test_constraint_violation_rate(benchmark, router):
    prompts = [
        "simple query 1", "general query 1", "complex query 1",
        "simple query 2", "general query 2", "complex query 2",
        "simple query 3", "general query 3", "complex query 3",
    ]
    def run():
        violations = 0
        for p in prompts:
            cert = _route(router, p)
            if _is_violation(cert):
                violations += 1
        return {"violation_rate": violations / max(1, len(prompts))}
    results = benchmark(run)
    # Compitum target: zero violations under default constraints
    assert 0.0 <= results["violation_rate"] <= 0.05
