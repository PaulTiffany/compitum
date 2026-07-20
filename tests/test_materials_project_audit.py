from types import SimpleNamespace
import sys

import pandas as pd

from compitum.integrations.materials_project_audit import SRMFState, map_material_to_srmf


def test_current_phase_bias_dominant():
    # bias > drift and bias > constraint -> falls through both "if" checks to "bias"
    s = SRMFState(drift=0.1, constraint=0.1, bias=5.0)
    assert s.current_phase() == "bias"


def test_current_phase_tie_falls_to_bias():
    # drift == constraint (neither strictly greater) -> also falls through to "bias"
    s = SRMFState(drift=2.0, constraint=2.0, bias=0.0)
    assert s.current_phase() == "bias"


def test_map_material_to_srmf_handles_none():
    doc = SimpleNamespace(band_gap=None, density=None, nsites=None, formation_energy_per_atom=None)
    s = map_material_to_srmf(doc)
    assert s.drift > 0
    assert s.constraint >= 0
    assert s.bias == 0.0
    assert s.current_phase() in {"drift", "constraint", "bias"}


def test_audit_monkeypatched(monkeypatch):
    docs = [
        SimpleNamespace(
            material_id="mp-1",
            formula_pretty="LaNiO3",
            band_gap=0.1,
            density=7.2,
            nsites=5,
            formation_energy_per_atom=-1.2,
        ),
        SimpleNamespace(
            material_id="mp-2",
            formula_pretty="La2NiO4",
            band_gap=2.5,
            density=6.5,
            nsites=10,
            formation_energy_per_atom=-0.8,
        ),
    ]

    class FakeSummary:
        def search(self, **kwargs):
            return docs

    class FakeMaterials:
        def __init__(self):
            self.summary = FakeSummary()

    class FakeMPR:
        def __init__(self, api_key):
            self.materials = FakeMaterials()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    # Provide fake mp_api module
    fake_client = SimpleNamespace(MPRester=FakeMPR)
    monkeypatch.setitem(sys.modules, "mp_api", SimpleNamespace(client=fake_client))
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_client)

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold("dummy", {"elements": ["La", "Ni", "O"], "nelements": 3})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(
        ["material_id", "formula", "srmf_phase", "curvature_kappa", "stability_leak", "prediction"]
    ).issubset(df.columns)
