from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tools.reporting.report_builder import (
    MetricsSummary,
    build_html_report,
    build_metrics_summary,
)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, timeout: Optional[int] = None):
    start = datetime.utcnow()
    import time as _t
    t0 = _t.perf_counter()
    meta = {"cmd": cmd, "cwd": str(cwd) if cwd else str(Path().resolve()), "timeout_sec": timeout}
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        dt = _t.perf_counter() - t0
        return {
            **meta,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "started_at": start.isoformat() + "Z",
            "duration_sec": round(dt, 3),
        }
    except subprocess.TimeoutExpired as e:
        dt = _t.perf_counter() - t0
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        return {
            **meta,
            "returncode": None,
            "stdout": out or "",
            "stderr": (err or "") + f"\n[orchestrator] Timed out after {timeout}s",
            "timed_out": True,
            "started_at": start.isoformat() + "Z",
            "duration_sec": round(dt, 3),
        }


def find_latest(pattern: str) -> Optional[Path]:
    paths = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return paths[-1] if paths else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run tests, benchmarks, and build a report")
    ap.add_argument("--tests", action="store_true", help="Run pytest unit tests")
    ap.add_argument("--bench-routerbench", action="store_true", help="Run RouterBench baseline")
    ap.add_argument("--bench-compitum", action="store_true", help="Run Compitum evaluation")
    ap.add_argument("--config", type=str, default="data/rb_clean/evaluate_routers.yaml", help="Eval config path")
    ap.add_argument("--tokenizer-backend", type=str, default="tiktoken", choices=["tiktoken","tokencost","hf"], help="Tokenizer backend for RB runs")
    ap.add_argument("--wtp", type=float, default=1.0, help="Willingness to pay for Compitum summary")
    ap.add_argument("--wtp-list", type=str, default=None, help="Comma-separated WTP grid for Compitum (e.g. '0.0001,0.001,0.01,0.1,1.0')")
    ap.add_argument("--report-out", type=str, default=None, help="HTML report output path")
    ap.add_argument("--max-evals", type=int, default=0, help="Optional cap on number of eval rows (head)")
    ap.add_argument("--timeout-tests", type=int, default=900, help="Timeout (s) for unit tests")
    ap.add_argument("--timeout-rb", type=int, default=1200, help="Timeout (s) for RouterBench run")
    ap.add_argument("--timeout-compitum", type=int, default=900, help="Timeout (s) for Compitum run")
    ap.add_argument("--all", action="store_true", help="Run tests + both benchmarks + report")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    test_summary = {"stdout": "(skipped)", "returncode": "N/A"}
    run_meta = {"tests": None, "routerbench": None, "compitum": None}
    rb_files: List[Path] = []
    compitum_file: Optional[Path] = None
    metrics: Optional[MetricsSummary] = None

    # Helper: detect presence of RouterBench dataset (either location)
    def _has_rb_data() -> bool:
        candidates = [
            project_root / "src" / "routerbench" / "routerbench_5shot.pkl",
            project_root / "data" / "routerbench_5shot.pkl",
        ]
        override = env.get("ROUTERBENCH_DATA")
        if override:
            candidates.append(Path(override))
        return any(p.exists() for p in candidates)

    if args.tests or args.all:
        # Run unit tests quietly
        res = run_cmd([str(project_root / ".venv-routerbench" / "Scripts" / "python"), "-m", "pytest", "-q"], cwd=project_root, timeout=args.timeout_tests)
        test_summary = {k: res.get(k) for k in ("stdout","returncode","stderr","timed_out","duration_sec")}
        run_meta["tests"] = res

    rb_available = _has_rb_data()
    if (args.bench_routerbench or args.all) and rb_available:
        # Run upstream RouterBench via clean wrapper
        env_rb = env.copy()
        if args.max_evals and args.max_evals > 0:
            env_rb["RB_MAX_EVALS"] = str(args.max_evals)
        py = str(project_root / ".venv-routerbench" / "Scripts" / "python")
        cmd = [py, str(project_root / "tools" / "run_routerbench_clean.py"), f"--config={args.config}", "--local", f"--tokenizer-backend={args.tokenizer_backend}"]
        res = run_cmd(cmd, cwd=project_root, env=env_rb, timeout=args.timeout_rb)
        # Collect artifacts
        latest_collection = find_latest("data/rb_clean/eval_results/eval_results__*__rb_clean.csv")
        if latest_collection:
            rb_files.append(latest_collection)
        run_meta["routerbench"] = res

    elif (args.bench_routerbench or args.all) and not rb_available:
        # Note: dataset missing; record a skipped RB run
        run_meta["routerbench"] = {"skipped": True, "reason": "routerbench_5shot.pkl not found"}

    if (args.bench_compitum or args.all) and rb_available:
        # Run Compitum evaluation (uses pretrained predictors if available)
        py = str(project_root / ".venv-routerbench" / "Scripts" / "python")
        cmd = [py, str(project_root / "tools" / "evaluate_compitum.py"), f"--config={args.config}"]
        if args.max_evals and args.max_evals > 0:
            cmd.append(f"--max-evals={args.max_evals}")
        if args.wtp_list:
            cmd.append(f"--wtp-list={args.wtp_list}")
        res = run_cmd(cmd, cwd=project_root, env=env, timeout=args.timeout_compitum)
        # Find latest per-eval CSV produced by compitum eval
        latest_compitum = find_latest("data/rb_clean/eval_results/eval_results-eval-*-val_split.csv")
        if latest_compitum:
            compitum_file = latest_compitum
            # Build metrics summary
            wlist = [float(x.strip()) for x in args.wtp_list.split(',')] if args.wtp_list else None
            metrics = build_metrics_summary(compitum_file, wtp=args.wtp, wtp_list=wlist)
        run_meta["compitum"] = res

    elif (args.bench_compitum or args.all) and not rb_available:
        run_meta["compitum"] = {"skipped": True, "reason": "routerbench_5shot.pkl not found"}

    # Report
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    report_out = Path(args.report_out) if args.report_out else (project_root / "reports" / f"report_{ts}.html")
    # Persist machine-readable run log
    import json as _json
    log_path = report_out.with_suffix(".json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_json.dumps({"run_meta": run_meta}, indent=2), encoding="utf-8")

    out = build_html_report(report_out, test_summary, rb_files, compitum_file, metrics, run_meta)
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
