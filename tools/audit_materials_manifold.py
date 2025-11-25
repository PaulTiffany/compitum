#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from compitum.integrations.materials_project_audit import (
    audit_the_manifold,
    map_material_to_srmf,
    _curvature_kappa,
    _lyapunov_leak,
)


def _parse_criteria(args: argparse.Namespace) -> Dict[str, Any]:
    if args.criteria:
        try:
            return json.loads(args.criteria)
        except json.JSONDecodeError as e:  # pragma: no cover
            raise SystemExit(f"Invalid JSON for --criteria: {e}")
    crit: Dict[str, Any] = {}
    if args.elements:
        crit["elements"] = list(args.elements)
    if args.nelements is not None:
        crit["nelements"] = int(args.nelements)
    if not crit:
        crit = {"elements": ["La", "Ni", "O"], "nelements": 3}
    return crit


def _mock_docs() -> Sequence[Any]:
    return (
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
    )


def _evaluate_docs(
    docs: Iterable[Any],
    *,
    kappa_threshold: float,
    leak_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        state = map_material_to_srmf(doc)
        kappa = _curvature_kappa(state)
        leak = _lyapunov_leak(state)
        is_cand = (kappa >= float(kappa_threshold)) and (leak <= float(leak_threshold))
        pred = "candidate" if is_cand else "non_candidate"
        rows.append(
            dict(
                material_id=getattr(doc, "material_id", ""),
                formula=getattr(doc, "formula_pretty", ""),
                srmf_phase=state.current_phase(),
                curvature_kappa=float(kappa),
                stability_leak=float(leak),
                prediction=pred,
            )
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit Materials Project manifold and export CSV (simulated-friendly)"
    )
    ap.add_argument(
        "--criteria",
        type=str,
        default=None,
        help='Search criteria as JSON (e.g. {"elements":["La","Ni","O"],"nelements":3})',
    )
    ap.add_argument(
        "--elements",
        type=str,
        nargs="*",
        default=None,
        help="Elements list to include in criteria (alternative to --criteria)",
    )
    ap.add_argument(
        "--nelements",
        type=int,
        default=None,
        help="nelements value for criteria (alternative to --criteria)",
    )
    ap.add_argument("--kappa-threshold", type=float, default=0.5)
    ap.add_argument("--leak-threshold", type=float, default=0.1)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/materials_manifold_audit.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--offline-mock",
        action="store_true",
        help="Run without mp_api using synthetic docs (for CI/local testing)",
    )
    args = ap.parse_args()

    crit = _parse_criteria(args)
    df: pd.DataFrame
    if args.offline_mock:
        df = _evaluate_docs(
            _mock_docs(),
            kappa_threshold=args.kappa_threshold,
            leak_threshold=args.leak_threshold,
        )
    else:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise SystemExit("MP_API_KEY not set; use --offline-mock for synthetic run.")
        df = audit_the_manifold(
            api_key,
            crit,
            kappa_threshold=args.kappa_threshold,
            leak_threshold=args.leak_threshold,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote audit: {args.out} ({len(df)} rows)")
    if not df.empty:
        print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

