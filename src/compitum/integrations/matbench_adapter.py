from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, List, Optional

import pandas as pd


class AbstractMatbenchAdapter:
    """Abstract adapter that yields material-like docs for SRMF mapping.

    Concrete subclasses should implement `iter_docs()` to yield objects exposing
    attributes: band_gap, density, nsites, formation_energy_per_atom, and optionally
    material_id, formula_pretty, and an optional label column if present in source data.
    """

    def iter_docs(self) -> Iterable[Any]:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    def fields(self) -> List[str]:  # pragma: no cover - abstract by convention
        raise NotImplementedError


@dataclass
class CSVMatbenchAdapter(AbstractMatbenchAdapter):
    """CSV-backed adapter for offline Matbench-style data.

    Required columns: band_gap, density, nsites, formation_energy_per_atom.
    Optional: id/formula columns can be provided to map to material_id and formula_pretty.
    A label column, if present, is passed through as an attribute on yielded docs.
    """

    path: str
    id_column: Optional[str] = None
    formula_column: Optional[str] = None
    label_column: Optional[str] = None

    def __post_init__(self) -> None:
        self._df = pd.read_csv(self.path)
        self._validate()

    def _validate(self) -> None:
        required = [
            "band_gap",
            "density",
            "nsites",
            "formation_energy_per_atom",
        ]
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Optional columns are fine; if provided, ensure present
        if self.id_column and self.id_column not in self._df.columns:
            raise ValueError(f"id_column '{self.id_column}' not found in CSV")
        if self.formula_column and self.formula_column not in self._df.columns:
            raise ValueError(f"formula_column '{self.formula_column}' not found in CSV")
        if self.label_column and self.label_column not in self._df.columns:
            raise ValueError(f"label_column '{self.label_column}' not found in CSV")

    def iter_docs(self) -> Iterator[Any]:
        for _, row in self._df.iterrows():
            payload = {
                "band_gap": row["band_gap"],
                "density": row["density"],
                "nsites": row["nsites"],
                "formation_energy_per_atom": row["formation_energy_per_atom"],
            }
            if self.id_column:
                payload["material_id"] = row[self.id_column]
            if self.formula_column:
                payload["formula_pretty"] = row[self.formula_column]
            if self.label_column:
                payload[self.label_column] = row[self.label_column]
            yield SimpleNamespace(**payload)

    def fields(self) -> List[str]:
        cols = [
            "band_gap",
            "density",
            "nsites",
            "formation_energy_per_atom",
        ]
        if self.id_column:
            cols.append("material_id")
        if self.formula_column:
            cols.append("formula_pretty")
        if self.label_column:
            cols.append(self.label_column)
        return cols
