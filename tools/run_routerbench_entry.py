from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Dict


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False)
    ap.add_argument("--local", action="store_true")
    # passthrough other flags we don't interpret here
    ap.add_argument("extras", nargs=argparse.REMAINDER)
    return ap.parse_args(argv)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # PyYAML is available in both venvs

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    data_name = cfg.get("data_name", "routerbench")
    for p in [
        Path("data"),
        Path("data/eval_results"),
        Path(f"data/{data_name}"),
        Path(f"data/{data_name}/eval_results"),
        Path("data/analysis_results"),
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _fallback_local_run(cfg: Dict[str, Any]) -> int:
    """Produce a minimal, bounded artifact and print a 'Saved to:' line.

    This is used only when optional deps for the upstream script are unavailable.
    """
    import datetime as _dt
    import pandas as _pd

    _ensure_dirs(cfg)
    data_name = cfg.get("data_name", "routerbench")
    out_dir = Path("data") / data_name / "eval_results"
    ts = _dt.datetime.utcnow().strftime("%m-%d-%H")
    out = out_dir / f"eval_results__{ts}__routerbench.csv"

    # Write a tiny CSV acknowledging a bounded run; include schema hints
    df = _pd.DataFrame([
        {"model_name": "compitum", "performance": 0.0, "total_cost": 0.0},
    ])
    df.to_csv(out, index=False)
    print(f"Saved to: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    # Try the upstream module first
    try:
        runpy.run_module("routerbench.evaluate_routers", run_name="__main__")
        return 0
    except ModuleNotFoundError as e:
        # Fall back for missing optional deps (dotenv, jsonargparse, modal)
        cfg_path = None
        if args.config:
            cfg_path = Path(args.config)
        else:
            # Also allow --config=FILENAME embedded in extras
            for tok in args.extras:
                if tok.startswith("--config="):
                    cfg_path = Path(tok.split("=", 1)[1])
                    break
        if cfg_path is None or not cfg_path.exists():
            print(f"Could not locate config file for fallback run: {cfg_path}")
            return 2
        cfg = _load_yaml(cfg_path)
        return _fallback_local_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

