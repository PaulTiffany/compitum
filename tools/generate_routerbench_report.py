import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _find_latest_eval_pickle(root: Path) -> Optional[Path]:
    candidates = sorted(
        root.glob("eval_results__*__routerbench.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _find_latest_eval_csv(root: Path) -> Optional[Path]:
    candidates = sorted(
        root.glob("eval_results__*__routerbench.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _safe_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if col in df.columns:
        try:
            return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
        except Exception:
            return None
    return None


def summarize_collection(collection: Any) -> Dict[str, Dict[str, float]]:
    """Summarize per-router metrics from a RouterBench EvaluationResultCollection.

    Returns per-router metrics including accuracy (if present), mean cost/latency if available,
    and an oracle-match accuracy (proxy for regret).
    """
    out: Dict[str, Dict[str, float]] = {}
    for result in getattr(collection, "evaluation_results", []):
        name = getattr(result, "router_type", getattr(result, "router_name", "unknown"))
        df: Optional[pd.DataFrame] = getattr(result, "per_prompt_results", None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        metrics: Dict[str, float] = {}

        # Accuracy, if present (RouterBench renames to 'performance' sometimes)
        if "accuracy" in df.columns:
            metrics["accuracy_mean"] = float(pd.to_numeric(df["accuracy"], errors="coerce").mean())
        elif "performance" in df.columns:
            metrics["accuracy_mean"] = float(pd.to_numeric(df["performance"], errors="coerce").mean())

        # Oracle agreement (proxy for regret). Higher = lower regret.
        if {"chosen_model", "oracle_chosen_model"}.issubset(df.columns):
            oracle_match = (df["chosen_model"].astype(str) == df["oracle_chosen_model"].astype(str)).mean()
            metrics["oracle_match"] = float(oracle_match)

        # Try to infer cost/latency if present
        for col, key in (("cost", "cost_mean"), ("latency", "latency_mean"), ("e2e_ms", "e2e_mean_ms")):
            v = _safe_mean(df, col)
            if v is not None:
                metrics[key] = float(v)

        out[str(name)] = metrics
    return out


def _pick_compitum_key(keys: list[str]) -> Optional[str]:
    for k in keys:
        if "compitum" in k.lower():
            return k
    return None


def generate_markdown(summary: Dict[str, Dict[str, float]]) -> str:
    lines = []
    lines.append("# Compitum RouterBench Evaluation Summary")
    lines.append("")
    lines.append("This report compares Compitum against baseline routers on a bounded evaluation set.")
    lines.append("Higher oracle_match indicates lower regret relative to the oracle assignment.")
    lines.append("")

    # Emphasize Compitum and common baselines
    routers = sorted(summary.keys())
    priority = [r for r in routers if "compitum" in r.lower()] + [
        r for r in routers if "compitum" not in r.lower()
    ]

    lines.append("## Metrics")
    for name in priority:
        metrics = summary[name]
        if not metrics:
            continue
        lines.append(f"- {name}")
        for k in sorted(metrics.keys()):
            lines.append(f"  - {k}: {metrics[k]:.4f}")
    lines.append("")

    # If we have Compitum and at least one baseline, show deltas
    comp_key = _pick_compitum_key(list(summary.keys()))
    if comp_key:
        ck = comp_key
        cmet = summary[ck]
        lines.append("## Where Compitum Wins")
        for b in [k for k in summary.keys() if k != ck]:
            bmet = summary[b]
            if not bmet:
                continue
            # Prefer true regret proxy if available
            if "oracle_match" in cmet and "oracle_match" in bmet:
                delta = cmet["oracle_match"] - bmet["oracle_match"]
                lines.append(f"- Oracle-match vs {b}: {delta:+.4f}")
            if "cost_mean" in cmet and "cost_mean" in bmet:
                delta = cmet["cost_mean"] - bmet["cost_mean"]
                lines.append(f"- Cost mean vs {b}: {delta:+.4f}")
            if "e2e_mean_ms" in cmet and "e2e_mean_ms" in bmet:
                delta = cmet["e2e_mean_ms"] - bmet["e2e_mean_ms"]
                lines.append(f"- End-to-end latency ms vs {b}: {delta:+.4f}")
            # Fallback to accuracy gap
            if "accuracy_mean" in cmet and "accuracy_mean" in bmet:
                delta = cmet["accuracy_mean"] - bmet["accuracy_mean"]
                lines.append(f"- Accuracy mean vs {b}: {delta:+.4f}")
        lines.append("")

        # Regret (accuracy gap to oracle) if oracle present
        if "oracle" in summary and "accuracy_mean" in summary["oracle"] and "accuracy_mean" in cmet:
            lines.append("### Regret (accuracy gap to oracle)")
            oacc = summary["oracle"]["accuracy_mean"]
            comp_gap = oacc - cmet["accuracy_mean"]
            lines.append(f"- Compitum: {comp_gap:+.4f}")
            for b in [k for k in summary.keys() if k not in (ck, "oracle")]:
                if "accuracy_mean" in summary[b]:
                    bgap = oacc - summary[b]["accuracy_mean"]
                    lines.append(f"- {b}: {bgap:+.4f}")
            lines.append("")

    lines.append("## Determinism")
    lines.append(
        "Compitum routing is deterministic given fixed models and parameters, reducing variance and"
    )
    lines.append(
        "improving reproducibility compared to routers relying on stochastic LLM calls for decisions."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    # Ensure we can import local packages for unpickling
    sys.path.insert(0, str(project_root / "src"))
    rb_pkg_dir = project_root / "src" / "routerbench"
    if str(rb_pkg_dir) not in sys.path:
        sys.path.insert(0, str(rb_pkg_dir))

    # Optional CLI arg: explicit results file (CSV or PKL)
    explicit: Optional[Path] = None
    if len(sys.argv) > 1:
        explicit = Path(sys.argv[1]).resolve()

    # Locate results
    eval_dir = project_root / "data" / "routerbench" / "eval_results"
    # Prefer pickle for rich per-prompt metrics; fallback to CSV aggregate.
    latest = None
    if explicit is None:
        latest = _find_latest_eval_pickle(eval_dir)
    summary: Dict[str, Dict[str, float]] = {}
    if (explicit and explicit.suffix.lower() == ".pkl") or latest is not None:
        try:
            import pickle

            pkl_path = explicit if (explicit and explicit.suffix.lower() == ".pkl") else latest
            assert pkl_path is not None
            with open(pkl_path, "rb") as f:
                collection = pickle.load(f)
            summary = summarize_collection(collection)
        except Exception as e:
            print(f"Warning: Failed to load pickle: {e}")
            summary = {}

    if not summary:
        csv_path = None
        if explicit and explicit.suffix.lower() == ".csv":
            csv_path = explicit
        else:
            csv_path = _find_latest_eval_csv(eval_dir)
        if csv_path is None:
            print(f"No eval results found in {eval_dir}")
            return 2
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to load CSV {csv_path}: {e}")
            return 3
        if df.empty:
            print(f"CSV file {csv_path} is empty")
            return 4
        # Aggregate by model_name
        group = df.groupby("model_name")
        for name, g in group:
            metrics: Dict[str, float] = {}
            if "performance" in g.columns:
                metrics["accuracy_mean"] = float(pd.to_numeric(g["performance"], errors="coerce").mean())
            if "mean_regret" in g.columns:
                # Lower mean_regret is better; convert to an 'oracle_match' proxy = 1 - regret if bounded
                mr = float(pd.to_numeric(g["mean_regret"], errors="coerce").mean())
                metrics["oracle_match"] = float(max(0.0, min(1.0, 1.0 - mr)))
            if "total_cost" in g.columns:
                metrics["cost_mean"] = float(pd.to_numeric(g["total_cost"], errors="coerce").mean())
            summary[str(name)] = metrics
        print(f"Summarized from CSV: {csv_path}")

    md = generate_markdown(summary)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "routerbench_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
