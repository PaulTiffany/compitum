from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        recs: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame.from_records(recs)
    raise ValueError(f"Unsupported input file: {path}")


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str) and x.strip().startswith("{"):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Basic fields
    if "task" in df.columns:
        out["task"] = df["task"]
    elif "dataset" in df.columns:
        out["task"] = df["dataset"]
    # Utilities and regret
    if "regret" in df.columns:
        out["regret"] = pd.to_numeric(df["regret"], errors="coerce")
    elif {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        ub = pd.to_numeric(df["utility_best_baseline"], errors="coerce")
        uc = pd.to_numeric(df["utility_compitum"], errors="coerce")
        out["regret"] = ub - uc
        out["utility_best_baseline"] = ub
        out["utility_compitum"] = uc
    # Win flag
    if {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        out["win"] = (
            pd.to_numeric(df["utility_compitum"], errors="coerce")
            >= pd.to_numeric(df["utility_best_baseline"], errors="coerce")
        ).astype(int)
    # Boundary flag
    if "boundary_is_boundary" in df.columns:
        out["boundary"] = df["boundary_is_boundary"].astype(bool)
    elif "boundary" in df.columns:
        bj = df["boundary"].map(_maybe_json)
        if len(bj) and isinstance(bj.iloc[0], dict):
            out["boundary"] = [bool(x.get("is_boundary", False)) for x in bj]
    # Selected model (best-effort)
    for col in ("selected_model", "model", "winner", "routed_model"):
        if col in df.columns:
            out["selected_model"] = df[col]
            break
    return out


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    # Overall
    def _mean_ci(vals: pd.Series) -> Dict[str, Any]:
        v = pd.to_numeric(vals, errors="coerce")
        v = v[np.isfinite(v)]
        if v.empty:
            return {"mean": None, "n": 0}
        return {"mean": float(v.mean()), "n": int(v.size)}

    res["overall"] = {
        "win_rate": _mean_ci(df.get("win", pd.Series([], dtype=float))),
        "mean_regret": _mean_ci(df.get("regret", pd.Series([], dtype=float))),
        "boundary_rate": _mean_ci(df.get("boundary", pd.Series([], dtype=float))),
    }

    # Per-task
    by_task: Dict[str, Any] = {}
    if "task" in df.columns:
        for t, sub in df.groupby("task"):
            by_task[str(t)] = {
                "win_rate": _mean_ci(sub.get("win", pd.Series([], dtype=float))),
                "mean_regret": _mean_ci(sub.get("regret", pd.Series([], dtype=float))),
                "boundary_rate": _mean_ci(sub.get("boundary", pd.Series([], dtype=float))),
                "n": int(len(sub)),
            }
    if by_task:
        res["by_task"] = by_task

    # Selection distribution
    if "selected_model" in df.columns:
        vc = df["selected_model"].value_counts(dropna=False)
        total = int(vc.sum()) if vc.size > 0 else 0
        res["selection_distribution"] = {str(k): int(v) for k, v in vc.items()}
        res["selection_total"] = total
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="cs.CL summary: per-task win/boundary and selection distribution")
    ap.add_argument("--input", "-i", required=True, help="Input CSV/JSONL eval results")
    ap.add_argument("--out-json", type=Path, default=Path("reports/cl_summary.json"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/cl_summary.md"))
    args = ap.parse_args()

    df = _load(Path(args.input))
    data = _extract(df)
    rep = _summary(data)

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rep, indent=2))

    # Write Markdown
    lines: List[str] = ["# cs.CL Summary", ""]
    ov = rep.get("overall", {})
    def _fmt(x: Optional[float]) -> str:
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"
    if ov:
        lines += [
            "## Overall",
            f"- win_rate={_fmt(ov.get('win_rate', {}).get('mean'))}",
            f"- mean_regret={_fmt(ov.get('mean_regret', {}).get('mean'))}",
            f"- boundary_rate={_fmt(ov.get('boundary_rate', {}).get('mean'))}",
            "",
        ]
    if "by_task" in rep:
        lines += [
            "## By Task",
            "| task | win_rate | mean_regret | boundary_rate | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for t, s in rep["by_task"].items():
            lines.append(
                f"| {t} | {_fmt(s['win_rate']['mean'])} | {_fmt(s['mean_regret']['mean'])} | {_fmt(s['boundary_rate']['mean'])} | {s['n']} |"
            )
        lines.append("")
    if "selection_distribution" in rep:
        lines += ["## Selection Distribution", "| model | count |", "| --- | ---: |"]
        for k, v in rep["selection_distribution"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

