from __future__ import annotations

from pathlib import Path
import json
import argparse
from typing import Dict, List, Optional
import sys

# Allow running as a script without package context
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reporting.report_builder import build_html_report, build_metrics_summary, MetricsSummary


def find_latest(pattern: str) -> Optional[Path]:
    paths = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return paths[-1] if paths else None


def main() -> None:
    project_root = ROOT
    reports_dir = project_root / "reports"

    ap = argparse.ArgumentParser(description="Build deterministic release report HTML from pinned artifacts")
    ap.add_argument("--out", type=str, default=str(reports_dir / "report_release.html"), help="Output HTML path")
    ap.add_argument("--pins", type=str, default=str(reports_dir / "report_release_pins.json"), help="Pins JSON path")
    ap.add_argument("--compitum-csv", type=str, default=None, help="Explicit compitum CSV path (overrides pins)")
    ap.add_argument("--rb-csv", type=str, nargs="*", default=None, help="Explicit RB CSV paths (overrides pins)")
    ap.add_argument("--wtp-grid", type=str, default=None, help="Comma-separated WTP grid (e.g., '0.0001,0.001,0.01,0.1,1.0')")
    ap.add_argument("--wtp-selection", type=str, choices=["best","fixed"], default=None, help="Use best-of-grid or fixed WTP")
    ap.add_argument("--wtp", type=float, default=None, help="Fixed WTP to use when --wtp-selection=fixed")
    ap.add_argument("--from-manifest", action="store_true", help="Fallback: derive artifacts from artifact_manifest.json if pins absent")
    ap.add_argument("--write-pins", action="store_true", help="Write pins file capturing the artifacts and WTP settings used")
    args = ap.parse_args()

    out_html = Path(args.out)
    pins_path = Path(args.pins)
    in_json = reports_dir / "report_release.json"

    # Load prior run_meta if present (to avoid overwriting determinism logs)
    run_meta: Optional[Dict] = None
    test_summary: Dict[str, str] = {"stdout": "(skipped)", "returncode": "N/A"}
    if in_json.exists():
        try:
            data = json.loads(in_json.read_text(encoding="utf-8"))
            run_meta = data.get("run_meta")
            if isinstance(run_meta, dict):
                t = run_meta.get("tests") or {}
                # Mirror the shape used by ci_orchestrator when embedding
                test_summary = {k: t.get(k) for k in ("stdout", "returncode", "stderr", "timed_out", "duration_sec") if k in t}
        except Exception:
            pass

    # Load pins (if any)
    pins = None
    if pins_path.exists():
        try:
            pins = json.loads(pins_path.read_text(encoding="utf-8"))
        except Exception:
            pins = None

    # Resolve compitum_file and rb_files deterministically
    compitum_file: Optional[Path] = None
    rb_files: List[Path] = []

    # 1) Explicit CLI overrides
    if args.compitum_csv:
        compitum_file = Path(args.compitum_csv)
    if args.rb_csv:
        rb_files = [Path(p) for p in args.rb_csv]

    # 2) Pins file, if present
    if not compitum_file and pins and pins.get("compitum_csv"):
        compitum_file = Path(pins["compitum_csv"])  # absolute or relative allowed
    if not rb_files and pins and pins.get("rb_csvs"):
        rb_files = [Path(p) for p in pins.get("rb_csvs", [])]

    # 3) Manifest fallback
    if args.from_manifest and (not compitum_file or not rb_files):
        manifest_path = reports_dir / "artifact_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not rb_files:
                    cand = [Path(e["path"]) for e in manifest if isinstance(e, dict) and str(e.get("path","")) .endswith("__rb_clean.csv")]
                    if cand:
                        rb_files = [sorted(cand, key=lambda p: p.stat().st_mtime if p.exists() else 0)[-1]]
                if not compitum_file:
                    cand = [Path(e["path"]) for e in manifest if isinstance(e, dict) and "eval_results-eval-" in str(e.get("path","")) and str(e.get("path","")) .endswith("-val_split.csv")]
                    if cand:
                        compitum_file = sorted(cand, key=lambda p: p.stat().st_mtime if p.exists() else 0)[-1]
            except Exception:
                pass

    # 4) Final fallback to latest on disk
    if not rb_files:
        latest_collection = find_latest("data/rb_clean/eval_results/eval_results__*__rb_clean.csv")
        if latest_collection:
            rb_files.append(latest_collection)
    if not compitum_file:
        compitum_file = find_latest("data/rb_clean/eval_results/eval_results-eval-*-val_split.csv")

    # WTP selection strategy
    wgrid: Optional[List[float]] = None
    wsel: str = "best"
    wfixed: Optional[float] = None
    if args.wtp_grid:
        try:
            wgrid = [float(x.strip()) for x in args.wtp_grid.split(',') if x.strip()]
        except Exception:
            wgrid = None
    if args.wtp_selection:
        wsel = args.wtp_selection
    if args.wtp is not None:
        wfixed = float(args.wtp)
    # From pins if not set by CLI
    if pins:
        if wgrid is None and isinstance(pins.get("wtp_grid"), list):
            try:
                wgrid = [float(x) for x in pins.get("wtp_grid", [])]
            except Exception:
                wgrid = None
        if args.wtp_selection is None and pins.get("wtp_selection") in ("best","fixed"):
            wsel = pins.get("wtp_selection")
        if wfixed is None and isinstance(pins.get("wtp"), (int,float)):
            wfixed = float(pins.get("wtp"))

    # Defaults if still not specified
    if wgrid is None:
        wgrid = [0.0001, 0.001, 0.01, 0.1, 1.0]
    if wsel == "fixed" and wfixed is None:
        wfixed = 1.0

    # Compute metrics deterministically
    metrics: Optional[MetricsSummary] = None
    if compitum_file and Path(compitum_file).exists():
        try:
            if wsel == "best":
                metrics = build_metrics_summary(Path(compitum_file), wtp=1.0, wtp_list=wgrid)
            else:
                metrics = build_metrics_summary(Path(compitum_file), wtp=wfixed, wtp_list=[wfixed])
        except Exception:
            metrics = None

    # Optionally write pins to lock determinism for future builds
    if args.write_pins:
        pins_out = {
            "compitum_csv": str(compitum_file) if compitum_file else None,
            "rb_csvs": [str(p) for p in rb_files],
            "wtp_selection": wsel,
            "wtp_grid": wgrid,
            "wtp": wfixed if wsel == "fixed" else None,
        }
        pins_path.write_text(json.dumps(pins_out, indent=2), encoding="utf-8")

    build_html_report(Path(out_html), test_summary, rb_files, Path(compitum_file) if compitum_file else None, metrics, run_meta)
    print(f"Wrote sanitized release report to: {out_html}")


if __name__ == "__main__":
    main()
