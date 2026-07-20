from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from compitum.integrations.matbench_adapter import CSVMatbenchAdapter


def _write_csv(path: Path, with_labels: bool = False) -> None:
    rows: List[dict] = [
        {
            "band_gap": 0.1,
            "density": 7.2,
            "nsites": 5,
            "formation_energy_per_atom": -1.2,
            "mid": "mp-1",
            "formula": "LaNiO3",
            "label_candidate": 1 if with_labels else 0,
        },
        {
            "band_gap": 2.5,
            "density": 6.5,
            "nsites": 10,
            "formation_energy_per_atom": -0.8,
            "mid": "mp-2",
            "formula": "La2NiO4",
            "label_candidate": 0 if with_labels else 0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_csv_adapter_iter_docs(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv, with_labels=True)
    ad = CSVMatbenchAdapter(
        path=str(csv), id_column="mid", formula_column="formula", label_column="label_candidate"
    )
    docs = list(ad.iter_docs())
    assert len(docs) == 2
    d0 = docs[0]
    # hasattr alone only proves the key exists, not that the right CSV column
    # ended up on the right attribute -- assert exact values too.
    assert d0.band_gap == 0.1
    assert d0.density == 7.2
    assert d0.nsites == 5
    assert d0.formation_energy_per_atom == -1.2
    assert d0.material_id == "mp-1"
    assert d0.formula_pretty == "LaNiO3"
    assert d0.label_candidate == 1


def test_fields_with_all_optional_columns(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv, with_labels=True)
    ad = CSVMatbenchAdapter(
        path=str(csv), id_column="mid", formula_column="formula", label_column="label_candidate"
    )
    assert ad.fields() == [
        "band_gap",
        "density",
        "nsites",
        "formation_energy_per_atom",
        "material_id",
        "formula_pretty",
        "label_candidate",
    ]


def test_fields_and_iter_docs_without_optional_columns(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv, with_labels=True)
    ad = CSVMatbenchAdapter(path=str(csv))
    assert ad.fields() == ["band_gap", "density", "nsites", "formation_energy_per_atom"]
    d0 = next(iter(ad.iter_docs()))
    assert not hasattr(d0, "material_id")
    assert not hasattr(d0, "formula_pretty")
    assert not hasattr(d0, "label_candidate")


def test_missing_required_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    pd.DataFrame([{"band_gap": 0.1, "density": 7.2, "nsites": 5}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        CSVMatbenchAdapter(path=str(csv))


def test_missing_id_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    with pytest.raises(ValueError, match="id_column 'nope' not found"):
        CSVMatbenchAdapter(path=str(csv), id_column="nope")


def test_missing_formula_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    with pytest.raises(ValueError, match="formula_column 'nope' not found"):
        CSVMatbenchAdapter(path=str(csv), formula_column="nope")


def test_missing_label_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    with pytest.raises(ValueError, match="label_column 'nope' not found"):
        CSVMatbenchAdapter(path=str(csv), label_column="nope")
