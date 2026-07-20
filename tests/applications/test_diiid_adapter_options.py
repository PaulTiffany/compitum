from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compitum.applications.fusion.diiid_adapter import load_shot_csv


def test_adapter_minimal_columns_no_crash(tmp_path: Path):
    t = np.arange(10, dtype=float)
    q = np.full_like(t, 1.2)
    df = pd.DataFrame({"time_ms": t, "q_min": q})
    p = tmp_path / "shot_min.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p, state_dim=4)
    assert data.crash_index is None
    assert data.state.shape == (10, 4)
    # q_min should be mapped to state[:,2]
    assert np.allclose(data.state[:, 2], q)


def test_adapter_missing_required_column_raises(tmp_path: Path):
    df = pd.DataFrame({"time_ms": np.arange(5, dtype=float)})  # no q_min
    p = tmp_path / "shot_bad.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match="Missing required column 'q_min'"):
        load_shot_csv(p, state_dim=4)
