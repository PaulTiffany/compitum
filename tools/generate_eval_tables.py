from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def all_all_csvs(limit: int = 8) -> List[Path]:
    files = sorted(
        glob.glob("data/rb_clean/eval_results/eval_results-eval-all-*-val_split.csv"),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return [Path(p) for p in files[:limit]]


def latest_baseline_csv() -> Optional[Path]:
    files = sorted(
        glob.glob("data/rb_clean/eval_results/eval_results__*__rb_clean.csv"),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(files[0]) if files else None


def compute_per_baseline_winrate_from_two(
    df_base: pd.DataFrame, df_comp: pd.DataFrame, wtps: Iterable[float]
) -> str:
    # Work on copies to avoid chained assignment warnings
    df_base = df_base.copy()
    df_comp = df_comp.copy()
    rows = []
    # normalize
    for col in ("willingness_to_pay", "performance", "total_cost"):
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors="coerce")
        if col in df_comp.columns:
            df_comp[col] = pd.to_numeric(df_comp[col], errors="coerce")
    for w in wtps:
        B = df_base[(df_base["willingness_to_pay"] == w) & (df_base["model_name"] != "oracle")]
        C = df_comp[
            (df_comp["willingness_to_pay"] == w) & (df_comp["model_name"].astype(str) == "compitum")
        ]
        if B.empty or C.empty:
            continue
        # merge on eval_name
        M = B.merge(
            C[["eval_name", "performance", "total_cost"]].rename(
                columns={"performance": "perf_comp", "total_cost": "cost_comp"}
            ),
            on="eval_name",
            how="inner",
        )
        if M.empty:
            continue
        M["U_comp"] = M["perf_comp"] - w * M["cost_comp"]
        M["U_base"] = M["performance"] - w * M["total_cost"]
        for name, g in M.groupby("model_name"):
            if str(name).lower().startswith("compitum") or str(name).lower() == "oracle":
                continue
            n = int(len(g))
            if n == 0:
                continue
            win_rate = float((g["U_comp"] >= g["U_base"]).mean())
            rows.append({"baseline": str(name), "wtp": w, "win_rate": win_rate, "n": n})
    if not rows:
        return "No comparable per-eval rows found.\n"
    out = ["| Baseline | WTP | Win Rate | N |", "|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["baseline"], x["wtp"])):
        out.append(f"| {r['baseline']} | {r['wtp']:.2f} | {r['win_rate'] * 100:.1f}% | {r['n']} |")
    return "\n".join(out) + "\n"


def compute_per_baseline_winrate(df: pd.DataFrame, wtps: Iterable[float]) -> str:
    # model_name column has 'compitum' and LLM names; exclude 'oracle'
    rows = []
    for w in wtps:
        sub = df[df["willingness_to_pay"] == w]
        if sub.empty:
            continue
        comp = sub[sub["model_name"].astype(str) == "compitum"][
            ["eval_name", "performance", "total_cost"]
        ].rename(columns={"performance": "perf_comp", "total_cost": "cost_comp"})
        if comp.empty:
            continue
        # All baselines: any model_name not compitum/oracle
        baselines = sub[~sub["model_name"].isin(["compitum", "oracle"])][
            ["eval_name", "model_name", "performance", "total_cost"]
        ].copy()
        if baselines.empty:
            continue
        merged = baselines.merge(comp, on="eval_name", how="inner")
        if merged.empty:
            continue
        merged["U_comp"] = merged["perf_comp"] - w * merged["cost_comp"]
        merged["U_base"] = merged["performance"] - w * merged["total_cost"]
        grp = merged.groupby("model_name")
        for name, g in grp:
            n = int(len(g))
            if n == 0:
                continue
            win_rate = float((g["U_comp"] >= g["U_base"]).mean())
            rows.append({"baseline": name, "wtp": w, "win_rate": win_rate, "n": n})

    if not rows:
        # Fall back to panel-level comparison: mean(perf) - w*mean(cost)
        out = [
            "No comparable per-eval rows found. Panel-level utility comparison:",
            "",
            "| Baseline | WTP | U_comp | U_base | Win? |",
            "|---|---:|---:|---:|:--:|",
        ]
        panel = df[~df["model_name"].isin(["oracle"])]
        for w in wtps:
            sub = panel[panel["willingness_to_pay"] == w]
            if sub.empty:
                continue
            comp = sub[sub["model_name"] == "compitum"]
            if comp.empty:
                continue
            u_comp = float(comp["performance"].mean() - w * comp["total_cost"].mean())
            for name, g in sub.groupby("model_name"):
                if name in ("compitum",):
                    continue
                u_base = float(g["performance"].mean() - w * g["total_cost"].mean())
                win = "✅" if u_comp >= u_base else "✗"
                out.append(f"| {name} | {w:.2f} | {u_comp:.6f} | {u_base:.6f} | {win} |")
        # Sanitize any non-ASCII win markers that may appear in some environments
        out = [line.replace("�o.", "Y").replace("�o-", "N") for line in out]
        return "\n".join(out) + "\n"
    # Build Markdown
    out = ["| Baseline | WTP | Win Rate | N |", "|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["baseline"], x["wtp"])):
        out.append(f"| {r['baseline']} | {r['wtp']:.2f} | {r['win_rate'] * 100:.1f}% | {r['n']} |")
    return "\n".join(out) + "\n"


def compute_frontier_gap(
    df: pd.DataFrame, wtps: Iterable[float], *, bootstrap: int = 0, ci: float = 0.95
) -> str:
    lines: list[str] = []
    for w in wtps:
        sub = df[df["willingness_to_pay"] == w]
        if sub.empty:
            continue
        sub = sub[sub["model_name"] != "oracle"].copy()
        if sub.empty:
            continue
        sub["U"] = sub["performance"] - w * sub["total_cost"]
        comp = sub[sub["model_name"] == "compitum"][["eval_name", "U"]].rename(
            columns={"U": "U_comp"}
        )
        best = sub.groupby("eval_name")["U"].max().reset_index().rename(columns={"U": "U_best"})
        merged = comp.merge(best, on="eval_name", how="inner")
        if merged.empty:
            continue
        merged["gap"] = merged["U_best"] - merged["U_comp"]
        mean_gap = float(merged["gap"].mean())
        at_frontier = float((merged["gap"].abs() < 1e-9).mean())
        n = int(len(merged))
        if bootstrap and n > 1:
            samples: List[float] = []
            for _ in range(bootstrap):
                resample = merged.sample(n=n, replace=True)
                samples.append(float(resample["gap"].mean()))
            lo_q = (1 - ci) / 2
            hi_q = 1 - lo_q
            s = pd.Series(samples)
            lo = float(s.quantile(lo_q))
            hi = float(s.quantile(hi_q))
            lines.append(
                f"| {w:.2f} | {mean_gap:.6f} [{lo:.6f}, {hi:.6f}] | {at_frontier * 100:.1f}% | {n} |"
            )
        else:
            lines.append(f"| {w:.2f} | {mean_gap:.6f} | {at_frontier * 100:.1f}% | {n} |")
    if not lines:
        return "No frontier data available.\n"
    header = "| WTP | Avg Gap to Frontier {} | At Frontier | N |\n|---:|---:|---:|---:|\n".format(
        "[95% CI]" if bootstrap else ""
    )
    return header + "\n".join(lines) + "\n"


def compute_results_by_task(df: pd.DataFrame, wtps: Iterable[float]) -> str:
    lines: list[str] = ["# Results by Task", ""]
    for w in wtps:
        sub = df[df["willingness_to_pay"] == w].copy()
        if sub.empty:
            continue
        sub = sub[sub["model_name"] != "oracle"].copy()
        if sub.empty:
            continue
        sub["U"] = sub["performance"] - w * sub["total_cost"]
        tasks = []
        for task, g in sub.groupby("eval_name"):
            g = g.copy()
            u_comp_series = g.loc[g["model_name"] == "compitum", "U"]
            if u_comp_series.empty:
                continue
            u_comp = float(u_comp_series.mean())
            u_best_base_series = g.loc[~g["model_name"].isin(["compitum", "oracle"]), "U"]
            if u_best_base_series.empty:
                continue
            u_best = float(u_best_base_series.max())
            regret = u_best - u_comp
            # micro-average win rate over baseline rows for this task
            baseline_rows = g[~g["model_name"].isin(["compitum", "oracle"])].copy()
            if baseline_rows.empty:
                continue
            win_rate = float((u_comp >= baseline_rows["U"]).mean())
            n = int(len(baseline_rows))
            tasks.append({"task": str(task), "regret": regret, "win_rate": win_rate, "n": n})
        if not tasks:
            continue
        lines.append(f"## WTP = {w:.2f}")
        lines.append("")
        lines.append("| Task | Mean Regret | Win Rate | N |")
        lines.append("|---|---:|---:|---:|")
        for t in sorted(tasks, key=lambda x: x["task"]):
            lines.append(
                f"| {t['task']} | {t['regret']:.6f} | {t['win_rate'] * 100:.1f}% | {t['n']} |"
            )
        lines.append("")
    if len(lines) <= 2:
        return "# Results by Task\n\nNo per-task data available.\n"
    return "\n".join(lines)


def compute_panel_summary(df: pd.DataFrame, wtps: Iterable[float]) -> str:
    lines: list[str] = ["---", "title: Panel Summary", "---", "", "# Panel Summary", ""]
    try:
        present_wtps = sorted(
            {
                float(x)
                for x in pd.to_numeric(df.get("willingness_to_pay"), errors="coerce")
                .dropna()
                .unique()
            }
        )
    except Exception:
        present_wtps = []
    # tasks counted where compitum present
    comp_rows = df[df["model_name"].astype(str) == "compitum"]
    tasks = sorted(comp_rows["eval_name"].dropna().unique())
    n_tasks = len(tasks)
    models = sorted(set(df["model_name"].astype(str)) - {"oracle"})
    lines.append(f"- Tasks (with compitum present): {n_tasks}")
    lines.append(f"- Models (excluding oracle): {len(models)}")
    if present_wtps:
        lines.append(f"- WTP slices present: {', '.join(f'{w:.2f}' for w in present_wtps)}")
    # rows per WTP (compitum only)
    for w in wtps:
        sub = comp_rows[comp_rows["willingness_to_pay"] == w]
        if not sub.empty:
            lines.append(f"- Eval units at WTP={w:.2f}: {len(sub)} (compitum)")
    lines.append("")
    lines.append("Notes")
    lines.append("")
    lines.append("- Seeds are fixed in scripts; see docs/PEER_REVIEW.md (Environment, Seeds).")
    lines.append("- Panel is bounded; see docs/RouterBench-Summary.md for composition details.")
    return "\n".join(lines) + "\n"


def main() -> int:
    csvs = all_all_csvs()
    if not csvs:
        print("No multi-task CSV found.")
        return 1
    # Load and combine recent CSVs to include both baselines and compitum outputs
    frames = []
    for p in csvs:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        print("Failed to load any eval-all CSVs.")
        return 1
    df_comp_all = pd.concat(frames, ignore_index=True).drop_duplicates()
    # Normalize types
    df_comp_all["willingness_to_pay"] = pd.to_numeric(
        df_comp_all.get("willingness_to_pay"), errors="coerce"
    )
    wtps = [0.1, 1.0]

    per_base_md = ["# Per-Baseline Win Rate (Standalone)", ""]
    # Try per-eval utility win-rates by merging baseline and compitum CSVs
    base_csv = latest_baseline_csv()
    per_base_text: str
    if base_csv is not None:
        try:
            df_base = pd.read_csv(base_csv)
            df_comp = df_comp_all[df_comp_all["model_name"].astype(str) == "compitum"]
            per_base_text = compute_per_baseline_winrate_from_two(df_base, df_comp, wtps)
        except Exception:
            per_base_text = compute_per_baseline_winrate(df_comp_all, wtps)
    else:
        per_base_text = compute_per_baseline_winrate(df_comp_all, wtps)
    per_base_md.append(per_base_text)
    # Combine baseline + compitum for frontier computation
    if base_csv is not None:
        try:
            df_base2 = pd.read_csv(base_csv)
            df_all = pd.concat(
                [df_base2, df_comp_all], ignore_index=True, sort=False
            ).drop_duplicates()
        except Exception:
            df_all = df_comp_all
    else:
        df_all = df_comp_all
    df_all["willingness_to_pay"] = pd.to_numeric(df_all.get("willingness_to_pay"), errors="coerce")
    frontier_md = ["# Frontier Gap (Standalone)", ""]
    frontier_md.append(compute_frontier_gap(df_all, wtps, bootstrap=500, ci=0.95))

    # Panel summary
    panel_md = compute_panel_summary(df_all, wtps)

    # Results by Task
    rbt_md = compute_results_by_task(df_all, wtps)

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "Per-Baseline-WinRate.md").write_text("\n".join(per_base_md), encoding="utf-8")
    (docs / "Frontier-Gap.md").write_text("\n".join(frontier_md), encoding="utf-8")
    (docs / "Panel-Summary.md").write_text(panel_md, encoding="utf-8")
    (docs / "Results-By-Task.md").write_text(rbt_md, encoding="utf-8")
    print(
        json.dumps(
            {
                "csvs": [str(p) for p in csvs],
                "baseline_csv": str(base_csv) if base_csv else None,
                "wtps": wtps,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
