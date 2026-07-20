from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr


def _maybe_parse_json(val: Any) -> Any:
    if isinstance(val, str) and val and val.lstrip().startswith("{"):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        # Expect one JSON object per line
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        df = pd.DataFrame.from_records(records)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return df


def _extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort extraction of needed fields from common layouts.

    Tries to populate columns: regret, boundary_is_boundary, uncertainty,
    feasible, violations, trust_radius, timestamp.
    """
    out = pd.DataFrame(index=df.index)

    # Regret: direct or derivable
    if "regret" in df.columns:
        out["regret"] = pd.to_numeric(df["regret"], errors="coerce")
    elif {"utility_best_baseline", "utility_compitum"}.issubset(df.columns):
        out["regret"] = pd.to_numeric(df["utility_best_baseline"], errors="coerce") - pd.to_numeric(
            df["utility_compitum"], errors="coerce"
        )

    # Boundary: direct flag or nested JSON in 'boundary' column
    boundary_flag = None
    if "boundary_is_boundary" in df.columns:
        boundary_flag = df["boundary_is_boundary"].astype(bool)
    elif "boundary" in df.columns:
        parsed = df["boundary"].map(_maybe_parse_json)
        if isinstance(parsed.iloc[0], dict):
            boundary_flag = pd.Series(
                [bool(x.get("is_boundary", False)) for x in parsed], index=df.index
            )
    elif "boundary_analysis" in df.columns:
        parsed = df["boundary_analysis"].map(_maybe_parse_json)
        if isinstance(parsed.iloc[0], dict):
            boundary_flag = pd.Series(
                [bool(x.get("is_boundary", False)) for x in parsed], index=df.index
            )
    if boundary_flag is not None:
        out["boundary_is_boundary"] = boundary_flag

    # Uncertainty: direct or from 'utility_components'
    if "uncertainty" in df.columns:
        out["uncertainty"] = pd.to_numeric(df["uncertainty"], errors="coerce")
    elif "utility_components" in df.columns:
        parsed = df["utility_components"].map(_maybe_parse_json)
        if isinstance(parsed.iloc[0], dict):
            out["uncertainty"] = pd.to_numeric(
                [x.get("uncertainty", np.nan) for x in parsed], errors="coerce"
            )

    # Feasible/violations: from 'constraints'
    feasible = None
    if "feasible" in df.columns:
        feasible = df["feasible"].astype(bool)
    elif "constraints" in df.columns:
        parsed_c = df["constraints"].map(_maybe_parse_json)
        if isinstance(parsed_c.iloc[0], dict):
            feasible = pd.Series([bool(x.get("feasible", False)) for x in parsed_c], index=df.index)
            out["violations"] = [
                len(x.get("violations", []))
                if isinstance(x.get("violations", None), list)
                else np.nan
                for x in parsed_c
            ]
    if feasible is not None:
        out["feasible"] = feasible

    # Drift/trust-radius: from 'drift'
    if "trust_radius" in df.columns:
        out["trust_radius"] = pd.to_numeric(df["trust_radius"], errors="coerce")
    elif "drift" in df.columns:
        parsed_d = df["drift"].map(_maybe_parse_json)
        if isinstance(parsed_d.iloc[0], dict):
            out["trust_radius"] = pd.to_numeric(
                [x.get("trust_radius", np.nan) for x in parsed_d], errors="coerce"
            )

    # Timestamp if present
    for col in ("timestamp", "time", "ts"):
        if col in df.columns:
            out["timestamp"] = pd.to_numeric(df[col], errors="coerce")
            break

    # Optional join key
    if "pgd_signature" in df.columns:
        out["pgd_signature"] = df["pgd_signature"]

    return out


def _join_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    # Progressive outer-join on available keys
    base = frames[0]
    for nxt in frames[1:]:
        key = None
        # Prefer explicit key for alignment if present
        for k in ("pgd_signature",):
            if k in base.columns and k in nxt.columns:
                key = k
                break
        if key is not None:
            base = base.merge(nxt, on=key, how="outer")
        else:
            # Fallback: align by index length (best-effort)
            base = pd.concat([base.reset_index(drop=True), nxt.reset_index(drop=True)], axis=1)
    return base


def compute_deferral_quality(
    df: pd.DataFrame, topq: Optional[float], tau: Optional[float]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "regret" not in df.columns or "boundary_is_boundary" not in df.columns:
        return out
    reg = pd.to_numeric(df["regret"], errors="coerce")
    bnd = df["boundary_is_boundary"].astype(bool)
    if reg.isna().all() or bnd.isna().all():
        return out
    if topq is not None:
        q = np.quantile(reg.dropna(), 1.0 - topq)
        y = (reg >= q).astype(int)
    elif tau is not None:
        y = (reg >= tau).astype(int)
    else:
        q = np.quantile(reg.dropna(), 0.9)
        y = (reg >= q).astype(int)
    y_score = bnd.astype(int)
    try:
        auroc = roc_auc_score(y, y_score)
    except Exception:
        auroc = np.nan
    try:
        ap = average_precision_score(y, y_score)
    except Exception:
        ap = np.nan
    out.update({"available": True, "auroc": float(auroc), "average_precision": float(ap)})
    # Normalize score as AP (fallback to AUROC if AP nan)
    score = ap if np.isfinite(ap) else auroc
    out["score"] = float(score) if np.isfinite(score) else None
    return out


def compute_calibration(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "uncertainty" not in df.columns or "regret" not in df.columns:
        return out
    u = pd.to_numeric(df["uncertainty"], errors="coerce")
    r = pd.to_numeric(df["regret"], errors="coerce").abs()
    m = ~(u.isna() | r.isna())
    if m.sum() < 10:
        return out
    rho, _ = spearmanr(u[m], r[m])
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    out.update({"available": True, "spearman": rho, "score": max(0.0, min(1.0, rho))})
    return out


def compute_stability(df: pd.DataFrame, window: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "trust_radius" not in df.columns or "regret" not in df.columns:
        return out
    d = df[["timestamp", "trust_radius", "regret"]].copy()
    if "timestamp" in d.columns and not d["timestamp"].isna().all():
        d = d.sort_values("timestamp")
    d["delta_r"] = d["trust_radius"].diff(periods=1)
    d["delta_regret"] = d["regret"].diff(periods=-window) * -1.0  # future decrease positive
    mask = (~d["delta_r"].isna()) & (~d["delta_regret"].isna())
    if mask.sum() < 10:
        return out
    # Correlation: negative delta_r (shrink) associated with positive future delta_regret (improvement)
    rho, _ = spearmanr(-d.loc[mask, "delta_r"], d.loc[mask, "delta_regret"])
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    out.update({"available": True, "spearman": rho, "score": max(0.0, min(1.0, rho))})
    return out


def compute_compliance(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if "feasible" in df.columns:
        feas = df["feasible"].astype(bool)
        rate = float(feas.mean())
        out.update({"available": True, "feasible_rate": rate, "score": rate})
        return out
    if "violations" in df.columns:
        v = pd.to_numeric(df["violations"], errors="coerce")
        ok = float((v.fillna(0.0) <= 0).mean())
        out.update({"available": True, "feasible_rate": ok, "score": ok})
        return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compitum Control-of-Error Index (CEI) report")
    p.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="Input CSV/JSONL with per-item results",
    )
    p.add_argument("--out-json", type=Path, default=Path("reports/cei_report.json"))
    p.add_argument("--out-md", type=Path, default=Path("reports/cei_report.md"))
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--topq", type=float, default=0.1, help="Top quantile for high-regret labeling (e.g., 0.1)"
    )
    group.add_argument(
        "--tau", type=float, default=None, help="Absolute regret threshold for high-regret label"
    )
    p.add_argument(
        "--stability-window", type=int, default=1, help="Steps ahead to evaluate regret change"
    )
    args = p.parse_args()

    frames = []
    for raw in args.input:
        df = _load_table(Path(raw))
        frames.append(_extract_fields(df))
    table = _join_frames(frames)

    metrics: Dict[str, Any] = {}
    dq = compute_deferral_quality(table, topq=args.topq, tau=args.tau)
    cal = compute_calibration(table)
    stab = compute_stability(table, window=args.stability_window)
    comp = compute_compliance(table)

    metrics["deferral_quality"] = dq
    metrics["calibration"] = cal
    metrics["stability"] = stab
    metrics["compliance"] = comp

    # Aggregate CEI over available component scores
    scores = [
        m.get("score")
        for m in (dq, cal, stab, comp)
        if m.get("available") and m.get("score") is not None
    ]
    cei = float(np.mean(scores)) if scores else None
    metrics["cei"] = cei

    # Ensure output dir exists
    for outp in (args.out_json, args.out_md):
        outp.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    args.out_json.write_text(json.dumps(metrics, indent=2))

    # Write Markdown summary
    lines = [
        "# Control-of-Error Index (CEI)",
        "",
        f"CEI: {cei if cei is not None else 'n/a'}",
        "",
        "## Components",
    ]

    def _fmt(name: str, m: Dict[str, Any]) -> str:
        if not m.get("available"):
            return f"- {name}: n/a"
        # Include key metrics if present
        extras = []
        for k in ("average_precision", "auroc", "spearman", "feasible_rate"):
            if k in m and m[k] is not None and np.isfinite(m[k]):
                extras.append(f"{k}={m[k]:.3f}")
        extra = (" (" + ", ".join(extras) + ")") if extras else ""
        score = m.get("score")
        score_s = (
            f"{score:.3f}" if isinstance(score, (float, int)) and np.isfinite(score) else "n/a"
        )
        return f"- {name}: score={score_s}{extra}"

    lines.append(_fmt("Deferral quality", dq))
    lines.append(_fmt("Calibration", cal))
    lines.append(_fmt("Stability", stab))
    lines.append(_fmt("Compliance", comp))
    args.out_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
