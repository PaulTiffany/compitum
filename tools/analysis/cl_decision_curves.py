from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings as _warn

# Silence benign numpy warning when all components for a row are NaN
_warn.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame.from_records(rows)
    raise ValueError(f"Unsupported input file: {path}")


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str) and x.strip().startswith("{"):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # regret
    if "regret" in df.columns:
        out["regret"] = pd.to_numeric(df["regret"], errors="coerce").clip(lower=0.0)
    elif {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        ub = pd.to_numeric(df["utility_best_baseline"], errors="coerce")
        uc = pd.to_numeric(df["utility_compitum"], errors="coerce")
        out["regret"] = (ub - uc).clip(lower=0.0)
    else:
        out["regret"] = np.nan
    # boundary scores: entropy, gap, uncertainty
    # direct columns or nested JSON under boundary/boundary_analysis
    entropy = None
    gap = None
    uncert = None
    for col in ("entropy", "boundary_entropy"):
        if col in df.columns:
            entropy = pd.to_numeric(df[col], errors="coerce")
            break
    for col in ("utility_gap", "boundary_gap"):
        if col in df.columns:
            gap = pd.to_numeric(df[col], errors="coerce")
            break
    if "uncertainty" in df.columns:
        uncert = pd.to_numeric(df["uncertainty"], errors="coerce")
    if entropy is None or gap is None:
        # try parsing JSON
        for jcol in ("boundary_analysis", "boundary"):
            if jcol in df.columns:
                parsed = df[jcol].map(_maybe_json)
                if len(parsed) and isinstance(parsed.iloc[0], dict):
                    if entropy is None:
                        entropy = pd.to_numeric([x.get("entropy", np.nan) for x in parsed], errors="coerce")
                    if gap is None:
                        gap = pd.to_numeric([x.get("utility_gap", np.nan) for x in parsed], errors="coerce")
                    if uncert is None and "uncertainty" in parsed.iloc[0]:
                        uncert = pd.to_numeric([x.get("uncertainty", np.nan) for x in parsed], errors="coerce")
                break
    out["entropy"] = entropy if entropy is not None else np.nan
    out["gap"] = gap if gap is not None else np.nan
    out["uncertainty"] = uncert if uncert is not None else np.nan
    # boundary boolean if available
    if "boundary_is_boundary" in df.columns:
        out["boundary_flag"] = df["boundary_is_boundary"].astype(bool)
    elif "boundary" in df.columns:
        parsed = df["boundary"].map(_maybe_json)
        if len(parsed) and isinstance(parsed.iloc[0], dict):
            out["boundary_flag"] = [bool(x.get("is_boundary", False)) for x in parsed]
    # task label if present
    for col in ("task", "dataset", "benchmark"):
        if col in df.columns:
            out["task"] = df[col]
            break
    return out


def _rank_standardize(s: pd.Series, invert: bool = False) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    m = v.notna()
    ranks = pd.Series(np.nan, index=s.index, dtype=float)
    if m.sum() == 0:
        return ranks
    order = v[m].rank(method="average", pct=True)
    if invert:
        order = 1.0 - order
    ranks.loc[m] = order
    return ranks


def compute_ambiguity_score(df: pd.DataFrame) -> pd.Series:
    # High when entropy high, uncertainty high, gap low
    e = _rank_standardize(df.get("entropy", pd.Series([], dtype=float)), invert=False)
    u = _rank_standardize(df.get("uncertainty", pd.Series([], dtype=float)), invert=False)
    g = _rank_standardize(df.get("gap", pd.Series([], dtype=float)), invert=True)
    # average available components
    arr = np.vstack([
        e.fillna(np.nan).to_numpy(),
        u.fillna(np.nan).to_numpy(),
        g.fillna(np.nan).to_numpy(),
    ])
    # Guard against all-NaN columns leading to empty slice warnings;
    # np.nanmean already ignores NaNs, but when every component is NaN for a row
    # it may emit a warning. Compute with nanmean and silence warning by prechecking.
    if arr.size == 0:
        score = np.full(df.shape[0], np.nan)
    else:
        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter("ignore", category=RuntimeWarning)
            score = np.nanmean(arr, axis=0)
    return pd.Series(score, index=df.index)


def deferral_curve(regret: pd.Series, score: pd.Series, qs: List[float]) -> Dict[str, Any]:
    # Upper-bound counterfactual: deferring top-q by score to best baseline reduces regret to 0 for those items
    m = regret.notna() & score.notna()
    r = regret[m].to_numpy()
    s = score[m].to_numpy()
    order = np.argsort(-s)  # descending ambiguity
    r_sorted = r[order]
    n = r_sorted.size
    prefix_sum = np.cumsum(r_sorted)
    total = float(prefix_sum[-1]) if n > 0 else 0.0
    result = {"q": [], "mean_regret": [], "mean_regret_upperbound_with_deferral": []}
    for q in qs:
        k = int(round(q * n))
        k = min(max(k, 0), n)
        # Defer top k: remove their regret (upper bound improvement)
        residual = total - (prefix_sum[k - 1] if k > 0 else 0.0)
        mean_r = total / n if n > 0 else np.nan
        mean_r_def = residual / n if n > 0 else np.nan
        result["q"].append(q)
        result["mean_regret"].append(mean_r)
        result["mean_regret_upperbound_with_deferral"].append(mean_r_def)
    return result


def boundary_auc_ap(boundary_flag: Optional[pd.Series], score: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if boundary_flag is None or boundary_flag.isna().all() or score.isna().all():
        return out
    y = boundary_flag.astype(int)
    s = score.fillna(score.mean())
    try:
        auc = roc_auc_score(y, s)
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(y, s)
    except Exception:
        ap = np.nan
    out.update({"available": True, "auroc": float(auc), "average_precision": float(ap)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="cs.CL decision curves: ambiguity score deferral and boundary AUC/AP")
    ap.add_argument("--input", "-i", required=True, help="Eval CSV/JSONL with regret and boundary info")
    ap.add_argument("--out-json", type=Path, default=Path("reports/cl_decision_curves.json"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/cl_decision_curves.md"))
    ap.add_argument("--out-png", type=Path, default=Path("reports/cl_decision_curve.png"))
    ap.add_argument("--quantiles", type=str, default="0,0.05,0.1,0.2,0.3,0.4,0.5", help="Comma-separated deferral fractions")
    args = ap.parse_args()

    df = _load_table(Path(args.input))
    data = _extract_fields(df)
    score = compute_ambiguity_score(data)
    qs = [float(x) for x in args.quantiles.split(",") if x]
    curves = deferral_curve(data["regret"], score, qs)
    aucap = boundary_auc_ap(data.get("boundary_flag"), score)

    # Write JSON
    payload = {"deferral_curve": curves, "boundary_auc_ap": aucap}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    # Markdown summary
    lines: List[str] = ["# cs.CL Decision Curves", ""]
    lines.append("## Deferral (Upper-Bound) on Ambiguity Score")
    lines.append("| q_defer | mean_regret | mean_regret_with_deferral (upper bound) |")
    lines.append("| ---: | ---: | ---: |")
    for q, r0, r1 in zip(curves["q"], curves["mean_regret"], curves["mean_regret_upperbound_with_deferral"]):
        def _fmt(x: float) -> str:
            return "n/a" if not np.isfinite(x) else f"{x:.4f}"
        lines.append(f"| {q:.2f} | {_fmt(r0)} | {_fmt(r1)} |")
    lines.append("")
    if aucap.get("available"):
        lines.append("## Boundary Prediction (Ambiguity Score)")
        lines.append(f"- AUROC={aucap['auroc']:.3f}  AP={aucap['average_precision']:.3f}")
        lines.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curves["q"], curves["mean_regret"], label="observed", marker="o")
        ax.plot(curves["q"], curves["mean_regret_upperbound_with_deferral"], label="with deferral (upper bound)", marker="o")
        ax.set_xlabel("Deferral fraction by ambiguity score")
        ax.set_ylabel("Mean regret")
        ax.grid(True, alpha=0.3)
        ax.legend()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
