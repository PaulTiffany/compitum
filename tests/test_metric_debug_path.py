import os
import io
from contextlib import redirect_stdout

import numpy as np

from compitum.metric import SymbolicManifoldMetric


def test_metric_distance_debug_prints_when_env_set():
    os.environ["COMPITUM_DEBUG_METRIC"] = "1"
    try:
        metric = SymbolicManifoldMetric(D=2, rank=1, delta=1e-3)
        x = np.array([0.1, -0.2])
        mu = np.zeros(2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            d, sigma = metric.distance(x, mu)
        out = buf.getvalue()
        assert "!!! DISTANCE METHOD CALLED !!!" in out
        assert isinstance(d, float) and isinstance(sigma, float)
    finally:
        os.environ.pop("COMPITUM_DEBUG_METRIC", None)
