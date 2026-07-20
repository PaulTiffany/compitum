from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import pandas as pd


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str) and x.strip().startswith("{"):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _load_table(path: Path) -> pd.DataFrame:
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
                    pass
        return pd.DataFrame.from_records(recs)
    raise ValueError(f"Unsupported file: {path}")


def _extract_ur(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Uncertainty
    if "uncertainty" in df.columns:
        out["uncertainty"] = pd.to_numeric(df["uncertainty"], errors="coerce")
    elif "utility_components" in df.columns:
        comp = df["utility_components"].map(_maybe_json)
        if len(comp) and isinstance(comp.iloc[0], dict):
            out["uncertainty"] = pd.to_numeric(
                [c.get("uncertainty", np.nan) for c in comp], errors="coerce"
            )
    # Regret
    if "regret" in df.columns:
        out["regret"] = pd.to_numeric(df["regret"], errors="coerce").abs()
    elif {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        ub = pd.to_numeric(df["utility_best_baseline"], errors="coerce")
        uc = pd.to_numeric(df["utility_compitum"], errors="coerce")
        out["regret"] = (ub - uc).abs()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reliability curve: uncertainty vs. |regret| bins")
    ap.add_argument("--input", "-i", required=True, help="CSV/JSONL table of per-item results")
    ap.add_argument("--bins", type=int, default=10, help="Number of uncertainty bins (default 10)")
    ap.add_argument("--out-csv", type=Path, default=Path("reports/reliability_curve.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/reliability_curve.md"))
    ap.add_argument("--out-png", type=Path, default=Path("reports/reliability_curve.png"))
    args = ap.parse_args()

    df = _load_table(Path(args.input))
    ur = _extract_ur(df)
    # Gracefully handle missing uncertainty column
    if "uncertainty" not in ur.columns or ur["uncertainty"].isna().all():
        # Write minimal placeholders and exit without error
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "bin": ["n/a"],
                "mean_uncertainty": [np.nan],
                "mean_abs_regret": [np.nan],
                "count": [0],
            }
        ).to_csv(args.out_csv, index=False)
        args.out_md.write_text(
            "# Reliability Curve\n\nuncertainty not available; skipping.\n", encoding="utf-8"
        )
        # Skip plotting
        return
    m = ~(ur["uncertainty"].isna() | ur["regret"].isna())
    ur = ur[m]
    if ur.empty:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "bin": ["n/a"],
                "mean_uncertainty": [np.nan],
                "mean_abs_regret": [np.nan],
                "count": [0],
            }
        ).to_csv(args.out_csv, index=False)
        args.out_md.write_text(
            "# Reliability Curve\n\nno overlapping uncertainty/retarget rows; skipping.\n",
            encoding="utf-8",
        )
        return

    # Bin by uncertainty quantiles
    q = np.linspace(0.0, 1.0, args.bins + 1)
    edges = np.quantile(ur["uncertainty"], q)
    # Ensure monotone non-decreasing edges
    edges = np.unique(edges)
    # If too few unique edges, fall back to equal-width
    if len(edges) < 3:
        umin, umax = ur["uncertainty"].min(), ur["uncertainty"].max()
        edges = np.linspace(umin, umax, min(args.bins, 5) + 1)
    labels = [f"[{edges[i]:.3g}, {edges[i + 1]:.3g})" for i in range(len(edges) - 1)]
    bins = pd.cut(ur["uncertainty"], bins=edges, include_lowest=True, right=False, labels=labels)

    tbl = (
        ur.assign(bin=bins)
        .groupby("bin", observed=True)
        .agg(
            mean_uncertainty=("uncertainty", "mean"),
            mean_abs_regret=("regret", "mean"),
            count=("regret", "size"),
        )
        .reset_index()
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out_csv, index=False)

    # Markdown summary
    lines = [
        "# Reliability Curve",
        "",
        f"Input: {args.input}",
        "",
        "| bin | mean_uncertainty | mean_abs_regret | count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in tbl.iterrows():
        lines.append(
            f"| {r['bin']} | {r['mean_uncertainty']:.4f} | {r['mean_abs_regret']:.4f} | {int(r['count'])} |"
        )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    # Optional plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tbl["mean_uncertainty"], tbl["mean_abs_regret"], marker="o")
        ax.set_xlabel("Mean uncertainty (bin)")
        ax.set_ylabel("Mean |regret| (bin)")
        ax.grid(True, alpha=0.3)
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        plt.close(fig)
    except Exception:
        # Matplotlib not available or plotting failed; skip silently
        pass


if __name__ == "__main__":
    main()
