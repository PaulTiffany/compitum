from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _best_baseline_per_eval(b: pd.DataFrame, w: float) -> pd.DataFrame:
    df = b.copy()
    df["utility"] = df["performance"] - w * df["total_cost"]
    return df.loc[df.groupby("eval_name")["utility"].idxmax()]


def summarize(
    comp_csv: Path, base_csv: Path, wtps: List[float]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    cdf = pd.read_csv(comp_csv)
    bdf = pd.read_csv(base_csv)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    # Baselines of interest (strings contained in model_name)
    groups = {
        "knn": lambda s: "knn" in s.lower(),
        "mlp": lambda s: "mlp" in s.lower(),
        "cascading": lambda s: "cascading" in s.lower(),
    }
    for w in wtps:
        w_key = f"wtp={w:.2f}"
        out[w_key] = {}
        # Best baseline per eval (for regret computation)
        best_b = _best_baseline_per_eval(bdf, w)
        # Per-eval compitum stats
        agg: List[Tuple[float, float, float]] = []
        for ev in sorted(set(cdf["eval_name"].astype(str))):
            cb = cdf[cdf["eval_name"] == ev]
            bb = best_b[best_b["eval_name"] == ev]
            if cb.empty or bb.empty:
                continue
            c_perf = float(cb["performance"].mean())
            c_cost = float(cb["total_cost"].mean())
            c_util = c_perf - w * c_cost
            best_util = float(bb["utility"].values[0])
            regret = max(0.0, best_util - c_util)
            win = 1.0 if c_util >= best_util - 1e-12 else 0.0
            agg.append(
                (regret, win, c_cost - float(bb["total_cost"].values[0]) if win >= 0.5 else np.nan)
            )
        if agg:
            arr = np.array(agg)
            out[w_key]["compitum"] = {
                "mean_regret": float(np.nanmean(arr[:, 0])),
                "win_rate": float(np.nanmean(arr[:, 1])),
                "avg_cost_delta_on_wins": float(np.nanmean(arr[arr[:, 1] >= 0.5][:, 2]))
                if (arr[:, 1] >= 0.5).any()
                else float("nan"),
            }
        # Baseline regret summaries (vs the best baseline)
        for name, pred in groups.items():
            sub = bdf[bdf["model_name"].astype(str).apply(pred)].copy()
            if sub.empty:
                continue
            sub["utility"] = sub["performance"] - w * sub["total_cost"]
            stats: List[float] = []
            for ev in sorted(set(sub["eval_name"].astype(str))):
                s = sub[sub["eval_name"] == ev]
                if s.empty:
                    continue
                best_util = float(best_b[best_b["eval_name"] == ev]["utility"].values[0])
                mean_util = float(s["utility"].mean())
                stats.append(max(0.0, best_util - mean_util))
            if stats:
                out[w_key][name] = {
                    "mean_regret": float(np.mean(stats)),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ablations vs baselines at fixed WTPs")
    ap.add_argument("--compitum", required=True, help="Compitum per-eval CSV (possibly combined)")
    ap.add_argument(
        "--baselines",
        required=True,
        help="RouterBench baselines CSV (eval_results__*__rb_clean.csv)",
    )
    ap.add_argument("--wtps", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--out-json", type=str, default="reports/ablation_summary.json")
    ap.add_argument("--out-md", type=str, default="reports/ablation_summary.md")
    args = ap.parse_args()

    res = summarize(Path(args.compitum), Path(args.baselines), args.wtps)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    Path(args.out_json).write_text(_json.dumps(res, indent=2), encoding="utf-8")

    lines = [
        "# Ablation Summary (Fixed WTP)",
        "",
        "| WTP | Model | Mean Regret |",
        "|---:|:------|-----------:|",
    ]
    for w in sorted(res.keys()):
        for model, stats in res[w].items():
            lines.append(f"| {w} | {model} | {stats.get('mean_regret', float('nan')):.6f} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {args.out_json}\nWrote: {args.out_md}")


if __name__ == "__main__":
    main()
