from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from compitum.integrations.materials_project_audit import (
    SRMFState,
    _curvature_kappa,
    _lyapunov_leak,
    map_material_to_srmf,
)


def test_current_phase_bias_dominant():
    # bias > drift and bias > constraint -> falls through both "if" checks to "bias"
    s = SRMFState(drift=0.1, constraint=0.1, bias=5.0)
    assert s.current_phase() == "bias"


def test_current_phase_tie_falls_to_bias():
    # drift == constraint (neither strictly greater) -> also falls through to "bias"
    s = SRMFState(drift=2.0, constraint=2.0, bias=0.0)
    assert s.current_phase() == "bias"


def test_current_phase_drift_dominant():
    """No existing test ever asserted `current_phase()` actually returns
    "drift" -- only a loose `in {"drift", "constraint", "bias"}` check
    elsewhere, which can't distinguish the real label from e.g. a typo'd
    one."""
    s = SRMFState(drift=5.0, constraint=1.0, bias=1.0)
    assert s.current_phase() == "drift"


def test_current_phase_constraint_dominant():
    """No existing test ever exercised the "constraint" branch at all."""
    s = SRMFState(drift=1.0, constraint=5.0, bias=1.0)
    assert s.current_phase() == "constraint"


def test_current_phase_drift_equals_bias_falls_through():
    """`self.drift > self.bias` was never exercised at exact equality --
    with `drift > constraint` but `drift == bias`, the phase must fall all
    the way through to "bias", not "drift"."""
    s = SRMFState(drift=5.0, constraint=1.0, bias=5.0)
    assert s.current_phase() == "bias"


def test_current_phase_constraint_equals_bias_falls_through():
    """Same boundary as above, mirrored for the "constraint" branch's own
    `self.constraint > self.bias` check."""
    s = SRMFState(drift=1.0, constraint=5.0, bias=5.0)
    assert s.current_phase() == "bias"


def test_map_material_to_srmf_handles_none():
    doc = SimpleNamespace(band_gap=None, density=None, nsites=None, formation_energy_per_atom=None)
    s = map_material_to_srmf(doc)
    assert s.drift > 0
    assert s.constraint >= 0
    assert s.bias == 0.0
    assert s.current_phase() in {"drift", "constraint", "bias"}


def test_map_material_to_srmf_exact_values_with_real_attributes():
    """The all-None test above only checks loose bounds -- it can't catch a
    wrong getattr() key (e.g. "band_gap" mangled) or a wrong fallback
    default (e.g. nsites's `or 1` changed to `or 2`), since a missing
    attribute falls back to the SAME default either way. With real,
    non-None attribute values, drift/constraint/bias must match the exact
    documented formulas: drift = 1/(band_gap+0.01), constraint =
    density*ln(nsites), bias = |formation_energy_per_atom|."""
    doc = SimpleNamespace(
        band_gap=0.1, density=7.2, nsites=5, formation_energy_per_atom=-1.2
    )
    s = map_material_to_srmf(doc)
    assert np.isclose(s.drift, 1.0 / (0.1 + 0.01))
    assert np.isclose(s.constraint, 7.2 * np.log(5))
    assert s.bias == 1.2
    assert s.current_phase() == "constraint"


def test_curvature_kappa_exact_value():
    """`_curvature_kappa`'s output was never checked for an exact value --
    only indirectly, via a loose column-existence check on the DataFrame it
    feeds into. Pins the denominator (1.0 + constraint + bias) and the
    division itself, catching a wrong constant, a sign flip on either term,
    or division replaced with multiplication."""
    s = SRMFState(drift=2.0, constraint=3.0, bias=1.0)
    assert np.isclose(_curvature_kappa(s), 2.0 / (1.0 + 3.0 + 1.0))


def test_lyapunov_leak_reads_the_right_status_key():
    """`_lyapunov_leak` constructs a fresh `LyapunovController` and reads
    `status["lyapunov_function"]` after one `update()` call. The
    controller's `kappa`/`r0`/`integral_gain` construction arguments and the
    `grad_norm` passed to `update()` do NOT affect `lyapunov_function()`
    (confirmed directly: it depends only on `drift_integral`, which only
    accumulates `d_star`) -- so mutating those constants is genuinely
    equivalent for this function's output; no test could ever distinguish
    them via the returned leak value. Mangling the status dict KEY itself
    (`"lyapunov_function"` -> a typo'd key) IS observable here, since the
    real key is always present and a wrong key silently falls through to
    the `0.0` default. The `0.0` DEFAULT value itself, however, is
    unreachable dead code -- the key is never actually missing in practice
    -- so mutating the default (e.g. to `1.0`) is also genuinely equivalent
    and is not, and cannot be, covered by any test (documented, not fixed,
    matching this codebase's other accepted-defensive-default cases). A
    fresh controller's `drift_integral` after one update is `0.95 * d_star`
    (accumulate then decay), so the exact leak is
    `(0.95 * state.drift) ** 2`."""
    s = SRMFState(drift=5.0, constraint=0.0, bias=0.0)
    leak = _lyapunov_leak(s)
    assert np.isclose(leak, (0.95 * 5.0) ** 2)
    assert leak != 0.0


def test_map_material_to_srmf_none_band_gap_and_nsites_defaults_are_exact():
    """The all-None test only checks loose bounds, which can't distinguish
    band_gap's `or 0.0` default from `or 1.0`, nor nsites's `or 1` default
    (and the separate `max(nsites, 1)` floor) from a wrong value -- both
    changes still leave drift>0 and constraint>=0. Isolate them with
    density fixed at a real, nonzero value (so the nsites-related mutations
    show up in `constraint` instead of being masked by density's own
    default) while band_gap and nsites are both None."""
    doc = SimpleNamespace(
        band_gap=None, density=2.0, nsites=None, formation_energy_per_atom=-1.0
    )
    s = map_material_to_srmf(doc)
    assert np.isclose(s.drift, 1.0 / (0.0 + 0.01))  # band_gap default must be 0.0, not 1.0
    assert s.constraint == 0.0  # nsites default 1 -> log(1) == 0, not log(2) != 0
    assert s.bias == 1.0


def test_map_material_to_srmf_none_density_default_is_exact():
    """Mirrors the test above for density's `or 0.0` default specifically --
    nsites is a real, nonzero value here so a wrong density default (0.0 vs
    1.0) shows up as a nonzero constraint instead of being masked."""
    doc = SimpleNamespace(
        band_gap=1.0, density=None, nsites=5, formation_energy_per_atom=None
    )
    s = map_material_to_srmf(doc)
    assert s.constraint == 0.0  # density default must be 0.0, not 1.0 (which would give density*ln(5) != 0)


def _patch_mp_api(monkeypatch, docs):
    """Wire a fake mp_api.client.MPRester that returns `docs` from a search,
    matching what audit_the_manifold expects. Shared by every test below
    that needs to drive audit_the_manifold without a real API key."""

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

    fake_client = SimpleNamespace(MPRester=FakeMPR)
    monkeypatch.setitem(sys.modules, "mp_api", SimpleNamespace(client=fake_client))
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_client)


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
    _patch_mp_api(monkeypatch, docs)

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold("dummy", {"elements": ["La", "Ni", "O"], "nelements": 3})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(
        ["material_id", "formula", "srmf_phase", "curvature_kappa", "stability_leak", "prediction"]
    ).issubset(df.columns)


def test_audit_monkeypatched_exact_row_values(monkeypatch):
    """The test above only checks column NAMES exist, never the VALUES in
    them -- a wrong getattr() key for material_id/formula_pretty, or a
    mangled default, would silently write "" or a wrong string into those
    columns and go unnoticed. Pins the exact values for both fixture docs,
    computed independently via the same underlying functions this session
    verified precisely (map_material_to_srmf, _curvature_kappa,
    _lyapunov_leak)."""
    doc1 = SimpleNamespace(
        material_id="mp-1",
        formula_pretty="LaNiO3",
        band_gap=0.1,
        density=7.2,
        nsites=5,
        formation_energy_per_atom=-1.2,
    )
    doc2 = SimpleNamespace(
        material_id="mp-2",
        formula_pretty="La2NiO4",
        band_gap=2.5,
        density=6.5,
        nsites=10,
        formation_energy_per_atom=-0.8,
    )
    _patch_mp_api(monkeypatch, [doc1, doc2])

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold("dummy", {"elements": ["La", "Ni", "O"], "nelements": 3})

    row1, row2 = df.iloc[0], df.iloc[1]
    assert row1["material_id"] == "mp-1"
    assert row1["formula"] == "LaNiO3"
    assert row1["srmf_phase"] == "constraint"
    assert np.isclose(row1["curvature_kappa"], 0.6593371119702864)
    assert np.isclose(row1["stability_leak"], 74.58677685950414)
    assert row1["prediction"] == "non_candidate"

    assert row2["material_id"] == "mp-2"
    assert row2["formula"] == "La2NiO4"
    assert row2["srmf_phase"] == "constraint"
    assert np.isclose(row2["curvature_kappa"], 0.023761618241701924)
    assert np.isclose(row2["stability_leak"], 0.14325169441754892)
    assert row2["prediction"] == "non_candidate"


def test_audit_prediction_candidate_at_exact_kappa_threshold_boundary(monkeypatch):
    """`(kappa >= kappa_threshold) and (leak <= leak_threshold)` -- at an
    exact kappa/threshold tie, `>=` must still classify as candidate (a `>`
    mutant would not). leak_threshold is set generous so only the kappa
    boundary is under test."""
    doc = SimpleNamespace(
        material_id="mp-1",
        formula_pretty="LaNiO3",
        band_gap=0.1,
        density=7.2,
        nsites=5,
        formation_energy_per_atom=-1.2,
    )
    _patch_mp_api(monkeypatch, [doc])

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold(
        "dummy",
        {"elements": ["La", "Ni", "O"], "nelements": 3},
        kappa_threshold=0.6593371119702864,  # == this doc's exact kappa
        leak_threshold=1000.0,  # trivially satisfied
    )
    assert df.iloc[0]["prediction"] == "candidate"


def test_audit_prediction_candidate_at_exact_leak_threshold_boundary(monkeypatch):
    """Mirrors the kappa boundary test above for `leak <= leak_threshold` --
    at an exact tie, `<=` must still classify as candidate (a `<` mutant
    would not). kappa_threshold is set to 0 so only the leak boundary is
    under test."""
    doc = SimpleNamespace(
        material_id="mp-1",
        formula_pretty="LaNiO3",
        band_gap=0.1,
        density=7.2,
        nsites=5,
        formation_energy_per_atom=-1.2,
    )
    _patch_mp_api(monkeypatch, [doc])

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold(
        "dummy",
        {"elements": ["La", "Ni", "O"], "nelements": 3},
        kappa_threshold=0.0,  # trivially satisfied
        leak_threshold=74.58677685950414,  # == this doc's exact leak
    )
    assert df.iloc[0]["prediction"] == "candidate"


def test_audit_prediction_requires_both_conditions_not_either(monkeypatch):
    """`(kappa >= k) and (leak <= l)` mutated to `or` -- or to `is_cand =
    None` outright -- would classify this doc as a candidate since its
    kappa condition alone is trivially satisfied (threshold 0.0), even
    though its leak is far above a strict leak_threshold. Only `and`
    correctly requires both."""
    doc = SimpleNamespace(
        material_id="mp-1",
        formula_pretty="LaNiO3",
        band_gap=0.1,
        density=7.2,
        nsites=5,
        formation_energy_per_atom=-1.2,
    )
    _patch_mp_api(monkeypatch, [doc])

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold(
        "dummy",
        {"elements": ["La", "Ni", "O"], "nelements": 3},
        kappa_threshold=0.0,  # trivially satisfied
        leak_threshold=1.0,  # this doc's leak (74.59) fails this
    )
    assert df.iloc[0]["prediction"] == "non_candidate"


def test_audit_the_manifold_default_thresholds_are_exact():
    """kappa_threshold=0.5 and leak_threshold=0.1's default values were never
    checked directly. They can't be checked end-to-end through a realistic
    doc either: kappa = drift/(1+constraint+bias) can never exceed drift
    itself (constraint and bias are both always >= 0 by construction), and
    leak = (0.95*drift)**2 <= 0.1 forces drift <= ~0.333 -- meaning any doc
    whose leak passes the default leak_threshold already has kappa capped
    below 0.333, well under the default kappa_threshold=0.5 regardless of
    whether that default is 0.5 or the mutant's 1.5. No doc-driven scenario
    can make the default's exact value observable in the final prediction,
    so the default itself is inspected directly."""
    import inspect

    from compitum.integrations.materials_project_audit import audit_the_manifold

    sig = inspect.signature(audit_the_manifold)
    assert sig.parameters["kappa_threshold"].default == 0.5
    assert sig.parameters["leak_threshold"].default == 0.1


def test_audit_monkeypatched_missing_material_id_and_formula_use_empty_default(monkeypatch):
    """`getattr(doc, "material_id", "")` and `getattr(doc, "formula_pretty",
    "")` were never exercised on a doc that's actually missing those
    attributes -- every other test's fixture docs always set them, so the
    default value itself (as opposed to the getattr key) was never
    observed. A doc without material_id/formula_pretty at all must produce
    empty strings in those columns, not some other placeholder."""
    doc = SimpleNamespace(
        band_gap=0.1, density=7.2, nsites=5, formation_energy_per_atom=-1.2
    )
    assert not hasattr(doc, "material_id")
    assert not hasattr(doc, "formula_pretty")
    _patch_mp_api(monkeypatch, [doc])

    from compitum.integrations.materials_project_audit import audit_the_manifold

    df = audit_the_manifold("dummy", {"elements": ["La", "Ni", "O"], "nelements": 3})
    assert df.iloc[0]["material_id"] == ""
    assert df.iloc[0]["formula"] == ""
