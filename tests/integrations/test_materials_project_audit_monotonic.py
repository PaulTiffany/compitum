from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hypothesis import given, strategies as st

from compitum.integrations.materials_project_audit import map_material_to_srmf


@given(
    band_gap1=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    band_gap2=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    density=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    nsites=st.integers(min_value=1, max_value=200),
    fe=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_kappa_monotone_decreasing_in_band_gap(
    band_gap1: float, band_gap2: float, density: float, nsites: int, fe: float
) -> None:
    # Fix other fields; vary band_gap and check drift and kappa decrease when band_gap increases
    if np.isclose(band_gap1, band_gap2):
        return
    low, high = (band_gap1, band_gap2) if band_gap1 < band_gap2 else (band_gap2, band_gap1)
    d1 = SimpleNamespace(band_gap=low, density=density, nsites=nsites, formation_energy_per_atom=fe)
    d2 = SimpleNamespace(
        band_gap=high, density=density, nsites=nsites, formation_energy_per_atom=fe
    )
    s1 = map_material_to_srmf(d1)
    s2 = map_material_to_srmf(d2)
    # Drift is inversely related to band_gap (with epsilon), so it must decrease
    assert s1.drift >= s2.drift
    # Denominator in kappa is equal; thus kappa must also decrease
    k1 = s1.drift / (1.0 + s1.constraint + s1.bias)
    k2 = s2.drift / (1.0 + s2.constraint + s2.bias)
    assert k1 >= k2
