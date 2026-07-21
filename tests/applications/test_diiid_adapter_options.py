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

    with pytest.raises(ValueError) as exc_info:
        load_shot_csv(p, state_dim=4)
    # An unanchored `match=` substring search still passes if the message is
    # wrapped (e.g. "XX...XX"), since the original text remains a substring --
    # assert the exact message to actually pin it.
    assert str(exc_info.value) == f"Missing required column 'q_min' in {p}"


def test_adapter_default_state_dim_is_8(tmp_path: Path):
    """`state_dim`'s default was never exercised -- every other test passes
    it explicitly."""
    t = np.arange(5, dtype=float)
    df = pd.DataFrame({"time_ms": t, "q_min": np.full_like(t, 1.2)})
    p = tmp_path / "shot_default.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p)
    assert data.state.shape == (5, 8)


def test_adapter_column_mapping_is_exact(tmp_path: Path):
    """Te_core/ne/q_min must land in state columns 0/1/2 respectively --
    using distinct, nonzero per-column values so a column swap, a wrong
    index, or a renamed lookup key are all observable."""
    t = np.arange(4, dtype=float)
    te_core = np.array([10.0, 11.0, 12.0, 13.0])
    ne = np.array([20.0, 21.0, 22.0, 23.0])
    q_min = np.array([30.0, 31.0, 32.0, 33.0])
    df = pd.DataFrame({"time_ms": t, "Te_core": te_core, "ne": ne, "q_min": q_min})
    p = tmp_path / "shot_full.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p, state_dim=4)
    assert np.allclose(data.state[:, 0], te_core)
    assert np.allclose(data.state[:, 1], ne)
    assert np.allclose(data.state[:, 2], q_min)


def test_adapter_missing_optional_columns_zero_filled(tmp_path: Path):
    t = np.arange(5, dtype=float)
    df = pd.DataFrame({"time_ms": t, "q_min": np.full_like(t, 1.2)})
    p = tmp_path / "shot_minimal.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p, state_dim=4)
    assert np.allclose(data.state[:, 0], 0.0)
    assert np.allclose(data.state[:, 1], 0.0)


def test_adapter_crash_threshold_is_strictly_below_one(tmp_path: Path):
    """`q_min < 1.0` was never exercised at exact equality -- a sample at
    exactly 1.0 must not itself count as a crash."""
    t = np.arange(4, dtype=float)
    q_min = np.array([1.5, 1.0, 1.0, 0.9])  # only index 3 is a real crash
    df = pd.DataFrame({"time_ms": t, "q_min": q_min})
    p = tmp_path / "shot_boundary.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p, state_dim=4)
    assert data.crash_index == 3


def test_adapter_crash_index_is_first_not_later(tmp_path: Path):
    """With multiple samples below threshold, `crash_index` must be the
    *first* one."""
    t = np.arange(5, dtype=float)
    q_min = np.array([1.5, 0.8, 0.5, 0.3, 0.1])  # first crash at index 1
    df = pd.DataFrame({"time_ms": t, "q_min": q_min})
    p = tmp_path / "shot_multi_crash.csv"
    df.to_csv(p, index=False)

    data = load_shot_csv(p, state_dim=4)
    assert data.crash_index == 1
