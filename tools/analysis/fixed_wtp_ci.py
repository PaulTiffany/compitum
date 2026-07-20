from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_fixed_wtp_ci(
    df: pd.DataFrame, w: float, n_boot: int = 1000, seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    evals = sorted(set(df["eval_name"].unique()))
    if not evals:
        return {}
    # Restrict to baselines (non-compitum)
    bdf = df[df["model_name"] != "compitum"].copy()
    cdf = df[df["model_name"] == "compitum"].copy()
    per_eval = []
    for ev in evals:
        b = bdf[bdf["eval_name"] == ev]
        c = cdf[cdf["eval_name"] == ev]
        if b.empty or c.empty:
            continue
        c_perf = float(c["performance"].mean())
        c_cost = float(c["total_cost"].mean())
        c_util = c_perf - w * c_cost
        b_util = (b["performance"] - w * b["total_cost"]).astype(float)
        idx = int(b_util.idxmax())
        best_util = float(b_util.loc[idx])
        best_cost = float(b.loc[idx, "total_cost"])  # type: ignore[index]
        regret = max(0.0, best_util - c_util)
        win = 1.0 if c_util >= best_util - 1e-12 else 0.0
        cost_delta_on_win = (c_cost - best_cost) if win >= 0.5 else np.nan
        per_eval.append((regret, win, cost_delta_on_win))
    if not per_eval:
        return {}
    per_eval = np.array(
        [(_r, _w, (_cd if _cd == _cd else np.nan)) for (_r, _w, _cd) in per_eval], dtype=float
    )
    mean_regrets: List[float] = []
    win_rates: List[float] = []
    avg_cost_deltas: List[float] = []
    m = len(per_eval)
    for _ in range(n_boot):
        idxs = rng.integers(0, m, size=(m,))
        sample = per_eval[idxs]
        mean_regrets.append(float(np.nanmean(sample[:, 0])))
        win_rates.append(float(np.nanmean(sample[:, 1])))
        wmask = sample[:, 1] >= 0.5
        if wmask.any():
            avg_cost_deltas.append(float(np.nanmean(sample[wmask, 2])))
        else:
            avg_cost_deltas.append(float("nan"))

    def _ci(a: List[float]) -> Tuple[float, float, float]:
        arr = np.array([x for x in a if x == x], dtype=float)
        if arr.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        mu = float(np.mean(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        return (mu, lo, hi)

    return {
        "mean_regret": _ci(mean_regrets),
        "win_rate": _ci(win_rates),
        "avg_cost_delta_on_wins": _ci(avg_cost_deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute fixed-WTP bootstrap CIs from compitum CSV")
    ap.add_argument("--input", required=True, help="Path to compitum per-eval CSV")
    ap.add_argument("--wtps", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default="reports/fixed_wtp_summary.json")
    ap.add_argument("--out-md", type=str, default="reports/fixed_wtp_summary.md")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    results: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    for w in args.wtps:
        ci = compute_fixed_wtp_ci(df, w, n_boot=args.bootstrap, seed=args.seed)
        if ci:
            results[w] = ci

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(pd.Series(results).to_json(indent=2), encoding="utf-8")

    # Markdown summary
    def _fmt_ci(mu: float, lo: float, hi: float, *, pct: bool = False) -> str:
        def _f(x: float) -> str:
            if x != x:
                return "-"
            return f"{x * 100:.1f}%" if pct else f"{x:.6f}"

        return f"{_f(mu)} [{_f(lo)}, {_f(hi)}]"

    lines = [
        "# Fixed-WTP Analysis (95% CI)",
        "",
        "| WTP | Mean Regret | Win Rate | Avg Cost Delta (wins) |",
        "|---:|---:|---:|---:|",
    ]
    for w in sorted(results.keys()):
        mr = results[w]["mean_regret"]
        wr = results[w]["win_rate"]
        cd = results[w]["avg_cost_delta_on_wins"]
        lines.append(f"| {w:.2f} | {_fmt_ci(*mr)} | {_fmt_ci(*wr, pct=True)} | {_fmt_ci(*cd)} |")
    if all(results[w]["win_rate"][0] == 0.0 for w in results):
        lines.append("")
        lines.append(
            "_Note: No per-eval wins observed at these WTP slices; cost deltas on wins are undefined. See per-baseline win-rate and panel summaries for context._"
        )
        lines.append("")
        lines.append(
            "_Note: No per-eval wins observed at these WTP slices; cost deltas on wins are undefined. See per-baseline win-rate and panel summaries for context._"
        )

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {args.out_json}\nWrote: {args.out_md}")


if __name__ == "__main__":
    main()
