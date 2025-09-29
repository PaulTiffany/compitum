import numpy as np

from compitum.metric import SymbolicManifoldMetric


def main() -> None:
    rng = np.random.default_rng(0)
    D = 35
    M = SymbolicManifoldMetric(D, 8)
    # two clusters: math-like vs code-like
    math_center = rng.normal(0, 1, size=D)
    code_center = rng.normal(0, 1, size=D)
    code_center[:5] += 2.0
    X_math = rng.normal(0, 0.6, size=(500, D)) + math_center
    X_code = rng.normal(0, 0.6, size=(500, D)) + code_center
    dm = np.mean([M.distance(x, math_center)[0] for x in X_math])
    dc = np.mean([M.distance(x, code_center)[0] for x in X_code])
    print({"avg_d_math": float(dm), "avg_d_code": float(dc)})
if __name__ == "__main__":
    main()
