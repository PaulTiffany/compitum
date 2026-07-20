from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Lambda column
    if "lambda" not in out.columns:
        # Try alternatives
        for alt in ("wtp", "WTP", "lambda_value"):
            if alt in out.columns:
                out["lambda"] = pd.to_numeric(out[alt], errors="coerce")
                break
    # Utilities/regret
    if "regret" not in out.columns:
        if {"utility_best_baseline", "utility_compitum"}.issubset(out.columns):
            out["regret"] = pd.to_numeric(
                out["utility_best_baseline"], errors="coerce"
            ) - pd.to_numeric(out["utility_compitum"], errors="coerce")
    # Win flag
    if "win" not in out.columns and {"utility_best_baseline", "utility_compitum"}.issubset(
        out.columns
    ):
        out["win"] = (
            pd.to_numeric(out["utility_compitum"], errors="coerce")
            >= pd.to_numeric(out["utility_best_baseline"], errors="coerce")
        ).astype(int)
    # Feasible
    if "feasible" not in out.columns and "constraints" in out.columns:
        parsed = out["constraints"].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x
        )
        if len(parsed) and isinstance(parsed.iloc[0], dict):
            out["feasible"] = [bool(c.get("feasible", False)) for c in parsed]
    return out


def _percentile_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(12345)
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    n = vals.size
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(vals[idx]))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def _summarize(df: pd.DataFrame, alpha: float, B: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Overall
    reg = pd.to_numeric(df.get("regret", pd.Series([], dtype=float)), errors="coerce")
    win = pd.to_numeric(df.get("win", pd.Series([], dtype=float)), errors="coerce")
    feas = (
        df.get("feasible", pd.Series([], dtype=bool)).astype(bool)
        if "feasible" in df.columns
        else None
    )

    def _metric(vals: pd.Series) -> Dict[str, Any]:
        vals = pd.to_numeric(vals, errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            return {"mean": None, "ci": (None, None), "n": 0}
        lo, hi = _percentile_ci(vals.to_numpy(), alpha=alpha, B=B)
        return {"mean": float(vals.mean()), "ci": (lo, hi), "n": int(vals.size)}

    out["overall"] = {
        "mean_regret": _metric(reg),
        "win_rate": _metric(win),
        "feasible_rate": (
            {
                "mean": float(feas.mean()),
                "ci": _percentile_ci(feas.astype(float).to_numpy(), alpha, B),
                "n": int(feas.size),
            }
            if feas is not None and feas.size > 0
            else {"mean": None, "ci": (None, None), "n": 0}
        ),
    }

    # Per task if available
    by_task: Dict[str, Any] = {}
    task_col = None
    for c in ("task", "dataset", "benchmark", "panel_task"):
        if c in df.columns:
            task_col = c
            break
    if task_col:
        for t, sub in df.groupby(task_col):
            reg_t = pd.to_numeric(sub.get("regret", pd.Series([], dtype=float)), errors="coerce")
            win_t = pd.to_numeric(sub.get("win", pd.Series([], dtype=float)), errors="coerce")
            feas_t = (
                sub.get("feasible", pd.Series([], dtype=bool)).astype(bool)
                if "feasible" in sub.columns
                else None
            )
            by_task[str(t)] = {
                "mean_regret": _metric(reg_t),
                "win_rate": _metric(win_t),
                "feasible_rate": (
                    {
                        "mean": float(feas_t.mean()),
                        "ci": _percentile_ci(feas_t.astype(float).to_numpy(), alpha, B),
                        "n": int(feas_t.size),
                    }
                    if feas_t is not None and feas_t.size > 0
                    else {"mean": None, "ci": (None, None), "n": 0}
                ),
                "n": int(len(sub)),
            }
    if by_task:
        out["by_task"] = by_task

    return out


def _filter_lambda(df: pd.DataFrame, lambdas: Optional[List[float]]) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, pd.DataFrame] = {}
    if lambdas is None:
        # Try to infer small set of unique values
        if "lambda" in df.columns:
            uniq = sorted(set(pd.to_numeric(df["lambda"], errors="coerce")))
            uniq = [x for x in uniq if np.isfinite(x)]
            if len(uniq) > 8:
                uniq = uniq[:8]
            for lam in uniq:
                parts[str(lam)] = df[pd.to_numeric(df["lambda"], errors="coerce") == lam]
        else:
            parts["all"] = df
        return parts
    # Explicit lambdas
    for lam in lambdas:
        if "lambda" in df.columns:
            parts[str(lam)] = df[pd.to_numeric(df["lambda"], errors="coerce") == lam]
        else:
            parts[str(lam)] = df.copy()
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="cs.LG summary: paired bootstrap for regret/win/compliance"
    )
    ap.add_argument(
        "--input", "-i", action="append", required=True, help="Input CSV/JSONL eval results"
    )
    ap.add_argument(
        "--lambdas", type=float, nargs="*", default=None, help="Lambda slices (e.g., 0.1 1.0)"
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="1 - CI level (default 0.05)")
    ap.add_argument(
        "--bootstrap", type=int, default=1000, help="Bootstrap resamples (default 1000)"
    )
    ap.add_argument("--out-json", type=Path, default=Path("reports/lg_summary.json"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/lg_summary.md"))
    args = ap.parse_args()

    frames = [_ensure_columns(_load(Path(p))) for p in args.input]
    df = pd.concat(frames, ignore_index=True)

    by_slice = _filter_lambda(df, args.lambdas)
    report: Dict[str, Any] = {"alpha": args.alpha, "bootstrap": args.bootstrap, "slices": {}}
    for name, sub in by_slice.items():
        report["slices"][name] = _summarize(sub, alpha=args.alpha, B=args.bootstrap)

    # Ensure output paths
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    # Write JSON
    args.out_json.write_text(json.dumps(report, indent=2))

    # Write Markdown table (overall per-slice)
    lines: List[str] = ["# cs.LG Summary", ""]
    lines.append(
        "| slice | mean_regret | 95% CI | win_rate | 95% CI | feasible_rate | 95% CI | n |"
    )
    lines.append("| --- | ---: | --- | ---: | --- | ---: | --- | ---: |")
    for sname, data in report["slices"].items():
        ov = data["overall"]
        mr = ov["mean_regret"]
        wr = ov["win_rate"]
        fr = ov.get("feasible_rate", {"mean": None, "ci": (None, None)})
        n = mr.get("n") or wr.get("n") or 0

        def _fmt(x: Optional[float]) -> str:
            return (
                "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"
            )

        lines.append(
            f"| {sname} | {_fmt(mr['mean'])} | [{_fmt(mr['ci'][0])}, {_fmt(mr['ci'][1])}] | "
            f"{_fmt(wr['mean'])} | [{_fmt(wr['ci'][0])}, {_fmt(wr['ci'][1])}] | "
            f"{_fmt(fr.get('mean'))} | [{_fmt(fr.get('ci')[0] if fr.get('ci') else None)}, {_fmt(fr.get('ci')[1] if fr.get('ci') else None)}] | {n} |"
        )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
