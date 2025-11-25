#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote CSV: {out} ({len(rows)} rows)")


def _offline_mock(n: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        band_gap = float(rng.uniform(0.0, 3.0))
        density = float(rng.uniform(4.0, 9.0))
        nsites = int(rng.integers(2, 20))
        fe = float(rng.normal(-1.0, 0.5))
        y_true = 2.0 - band_gap + 0.1 * density + 0.01 * nsites - 0.1 * abs(fe)
        rows.append(
            dict(
                band_gap=band_gap,
                density=density,
                nsites=nsites,
                formation_energy_per_atom=fe,
                y_true=y_true,
                material_id=f"mp-{i}",
                formula=f"X{i}Y",
            )
        )
    return rows


def _from_mp(elements: List[str], nelements: Optional[int], objective: str, *, api_key: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as e:
        raise SystemExit("mp_api not installed; pip install mp_api or use --offline-mock") from e
    fields = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "density",
        "nsites",
        "formation_energy_per_atom",
    ]
    crit: Dict[str, Any] = {"elements": elements}
    if nelements and nelements > 0:
        crit["nelements"] = int(nelements)
    rows: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:  # pragma: no cover
        docs: Iterable[Any] = mpr.materials.summary.search(**crit, fields=fields)
        for doc in docs:
            d = SimpleNamespace(**{f: getattr(doc, f, None) for f in fields})
            # Ensure numeric fallbacks
            bg = float(getattr(d, "band_gap", 0.0) or 0.0)
            dens = float(getattr(d, "density", 0.0) or 0.0)
            ns = int(getattr(d, "nsites", 0) or 0)
            fe = getattr(d, "formation_energy_per_atom", None)
            fe = 0.0 if fe is None else float(fe)
            y = None
            if objective == "band_gap":
                y = bg
            elif objective == "-formation_energy":
                y = -abs(fe)
            else:
                y = bg
            rows.append(
                dict(
                    band_gap=bg,
                    density=dens,
                    nsites=ns,
                    formation_energy_per_atom=fe,
                    y_true=float(y),
                    material_id=getattr(d, "material_id", ""),
                    formula=getattr(d, "formula_pretty", ""),
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a Matbench-like CSV (SRMF features + objective)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--from-mp", action="store_true", help="Export from Materials Project search (requires MP_API_KEY)")
    ap.add_argument("--elements", nargs="*", default=["La", "Ni", "O"], help="Elements for MP search")
    ap.add_argument("--nelements", type=int, default=3, help="Number of elements in MP search")
    ap.add_argument("--objective", type=str, default="band_gap", choices=["band_gap", "-formation_energy"], help="Objective to set as y_true")
    ap.add_argument("--limit", type=int, default=500, help="Max rows from MP search")
    ap.add_argument("--offline-mock", action="store_true", help="Write a synthetic CSV for testing/demo")
    ap.add_argument("--out", type=Path, default=Path("data/matbench_task.csv"))
    args = ap.parse_args()

    if args.offline_mock:
        rows = _offline_mock(n=200)
        _write_csv(rows, args.out)
        return 0

    if args.from_mp:
        key = os.environ.get("MP_API_KEY")
        if not key:
            raise SystemExit("MP_API_KEY not set; use --offline-mock or export the variable")
        rows = _from_mp(args.elements, args.nelements, args.objective, api_key=key, limit=args.limit)
        if not rows:
            raise SystemExit("No rows returned from MP; adjust criteria")
        _write_csv(rows, args.out)
        return 0

    # Default to offline mock if no source specified
    rows = _offline_mock(n=200)
    _write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
