from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
    return recs


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.DataFrame.from_records(_load_jsonl(path))
    raise ValueError(f"Unsupported file type: {path}")


def _extract_from_certs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # timestamp
    for col in ("timestamp", "time", "ts"):
        if col in df.columns:
            out["timestamp"] = pd.to_numeric(df[col], errors="coerce")
            break
    # drift/trust
    drift = df.get("drift") or df.get("drift_status")
    if drift is not None:
        parsed = drift.map(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x)
        if len(parsed) and isinstance(parsed.iloc[0], dict):
            out["trust_radius"] = pd.to_numeric([d.get("trust_radius", np.nan) for d in parsed], errors="coerce")
            out["drift_ema"] = pd.to_numeric([d.get("drift_ema", np.nan) for d in parsed], errors="coerce")
            out["drift_integral"] = pd.to_numeric([d.get("drift_integral", np.nan) for d in parsed], errors="coerce")
    # keys for joining
    if "pgd_signature" in df.columns:
        out["pgd_signature"] = df["pgd_signature"]
    return out


def _extract_eval(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # regret
    if "regret" in df.columns:
        out["regret"] = pd.to_numeric(df["regret"], errors="coerce")
    elif {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        ub = pd.to_numeric(df["utility_best_baseline"], errors="coerce")
        uc = pd.to_numeric(df["utility_compitum"], errors="coerce")
        out["regret"] = ub - uc
    # timestamp
    for col in ("timestamp", "time", "ts"):
        if col in df.columns:
            out["timestamp"] = pd.to_numeric(df[col], errors="coerce")
            break
    # key
    if "pgd_signature" in df.columns:
        out["pgd_signature"] = df["pgd_signature"]
    return out


def _align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    base = a.copy()
    if "pgd_signature" in a.columns and "pgd_signature" in b.columns:
        return base.merge(b, on="pgd_signature", how="outer", suffixes=("_cert", "_eval"))
    # fallback: concat by index
    return pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)


def compute_trust_events(df: pd.DataFrame, eps: float = 1e-9) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "trust_radius" not in df.columns:
        return out
    r = pd.to_numeric(df["trust_radius"], errors="coerce")
    r = r.dropna()
    if len(r) < 2:
        return out
    dr = r.diff()
    shrink = int((dr < -eps).sum())
    expand = int((dr > eps).sum())
    steady = int((dr.abs() <= eps).sum())
    out.update(
        {
            "available": True,
            "count": int(len(r)),
            "shrink_events": shrink,
            "expand_events": expand,
            "steady_steps": steady,
            "mean_r": float(r.mean()),
            "median_r": float(r.median()),
            "min_r": float(r.min()),
            "max_r": float(r.max()),
        }
    )
    return out


def compute_event_regret_correlation(df: pd.DataFrame, window: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "trust_radius" not in df.columns or "regret" not in df.columns:
        return out
    d = df[["timestamp", "trust_radius", "regret"]].copy()
    if "timestamp" in d.columns and not d["timestamp"].isna().all():
        d = d.sort_values("timestamp")
    d["delta_r"] = d["trust_radius"].diff(periods=1)
    d["delta_regret_future"] = d["regret"].diff(periods=-window) * -1.0
    m = (~d["delta_r"].isna()) & (~d["delta_regret_future"].isna())
    if m.sum() < 5:
        return out
    rho, _ = spearmanr(-d.loc[m, "delta_r"], d.loc[m, "delta_regret_future"])  # shrink vs improvement
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    out.update({"available": True, "spearman": rho})
    return out


def write_outputs(metrics: Dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    lines = [
        "# Control KPIs",
        "",
    ]
    te = metrics.get("trust_events", {})
    if te.get("available"):
        lines += [
            "## Trust-radius events",
            f"- count={te['count']} shrink={te['shrink_events']} expand={te['expand_events']} steady={te['steady_steps']}",
            f"- r: mean={te['mean_r']:.3f} median={te['median_r']:.3f} min={te['min_r']:.3f} max={te['max_r']:.3f}",
            "",
        ]
    corr = metrics.get("shrink_improve_corr", {})
    if corr.get("available"):
        lines += [
            "## Correlation (shrink vs future improvement)",
            f"- Spearman ρ={corr['spearman']:.3f}",
            "",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Control KPIs from certificates and eval outputs")
    ap.add_argument("--certs", required=True, help="Certificates JSONL or CSV (with drift/trust_radius and timestamp)")
    ap.add_argument("--eval", required=True, help="Evaluation CSV/JSONL containing regret and optional timestamp")
    ap.add_argument("--out-json", type=Path, default=Path("reports/control_kpis.json"))
    ap.add_argument("--out-md", type=Path, default=Path("reports/control_kpis.md"))
    ap.add_argument("--window", type=int, default=1, help="Steps ahead for improvement correlation")
    args = ap.parse_args()

    certs_df = _load_table(Path(args.certs))
    eval_df = _load_table(Path(args.eval))
    a = _extract_from_certs(certs_df)
    b = _extract_eval(eval_df)
    table = _align(a, b)

    metrics: Dict[str, Any] = {}
    metrics["trust_events"] = compute_trust_events(table)
    metrics["shrink_improve_corr"] = compute_event_regret_correlation(table, window=args.window)

    write_outputs(metrics, args.out_json, args.out_md)


if __name__ == "__main__":
    main()

