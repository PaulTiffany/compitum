from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so `tools.*` imports work when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.reporting.report_builder import (  # noqa: E402
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
    ap.add_argument("--bench-matbench", action="store_true", help="Run Matbench regret pipeline (smoke)")
    ap.add_argument("--config", type=str, default="data/rb_clean/evaluate_routers.yaml", help="Eval config path")
    ap.add_argument("--tokenizer-backend", type=str, default="tiktoken", choices=["tiktoken","tokencost","hf"], help="Tokenizer backend for RB runs")
    ap.add_argument("--wtp", type=float, default=1.0, help="Willingness to pay for Compitum summary")
    ap.add_argument("--wtp-list", type=str, default=None, help="Comma-separated WTP grid for Compitum (e.g. '0.0001,0.001,0.01,0.1,1.0')")
    ap.add_argument("--report-out", type=str, default=None, help="HTML report output path")
    ap.add_argument("--max-evals", type=int, default=0, help="Optional cap on number of eval rows (head)")
    ap.add_argument("--timeout-tests", type=int, default=900, help="Timeout (s) for unit tests")
    ap.add_argument("--timeout-rb", type=int, default=1200, help="Timeout (s) for RouterBench run")
    ap.add_argument("--timeout-compitum", type=int, default=900, help="Timeout (s) for Compitum run")
    ap.add_argument("--timeout-matbench", type=int, default=900, help="Timeout (s) for Matbench run")
    ap.add_argument("--matbench-csv", type=str, default="data/matbench_demo.csv", help="Matbench CSV path")
    ap.add_argument("--matbench-objective-col", type=str, default="y_true", help="Matbench objective column")
    ap.add_argument("--matbench-mode", type=str, default="max", choices=["max", "min"], help="Matbench objective direction")
    ap.add_argument("--matbench-topk-grid", type=str, default="1,5", help="Comma-separated k grid for Matbench")
    ap.add_argument("--matbench-lambda-grid", type=str, default="0.0,0.5,1.0", help="Comma-separated lambda grid for Matbench SRMF")
    ap.add_argument("--matbench-bootstrap", type=int, default=50, help="Bootstrap iterations for Matbench smoke")
    ap.add_argument("--matbench-out-prefix", type=str, default="reports/matbench", help="Output prefix for Matbench artifacts")
    ap.add_argument("--all", action="store_true", help="Run tests + both benchmarks + report")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    test_summary = {"stdout": "(skipped)", "returncode": "N/A"}
    run_meta = {"tests": None, "routerbench": None, "compitum": None, "matbench": None}
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

    # Resolve venv python path cross-platform
    venv_dir = project_root / ".venv-routerbench"
    py_exe = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    if args.tests or args.all:
        # Run unit tests quietly
        res = run_cmd([str(py_exe), "-m", "pytest", "-q"], cwd=project_root, timeout=args.timeout_tests)
        test_summary = {k: res.get(k) for k in ("stdout","returncode","stderr","timed_out","duration_sec")}
        run_meta["tests"] = res

    rb_available = _has_rb_data()
    if (args.bench_routerbench or args.all) and rb_available:
        # Run upstream RouterBench via clean wrapper
        env_rb = env.copy()
        if args.max_evals and args.max_evals > 0:
            env_rb["RB_MAX_EVALS"] = str(args.max_evals)
        py = str(py_exe)
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
        py = str(py_exe)
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

    if args.bench_matbench or args.all:
        py_mb = sys.executable  # use current interpreter
        mb_env = env.copy()
        mb_env.setdefault("OMP_NUM_THREADS", "1")
        mb_env.setdefault("MKL_NUM_THREADS", "1")
        mb_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        mb_env.setdefault("NUMEXPR_NUM_THREADS", "1")

        csv_path = args.matbench_csv
        objective = args.matbench_objective_col
        mode = args.matbench_mode
        topk = args.matbench_topk_grid
        lambdas = args.matbench_lambda_grid
        bootstrap = str(args.matbench_bootstrap)

        reports = project_root / "reports"
        reports.mkdir(exist_ok=True)
        prefix = Path(args.matbench_out_prefix)
        calib_json = reports / f"{prefix.stem}_calibration_smoke.json"
        scores_csv = reports / f"{prefix.stem}_scores_smoke.csv"
        regret_json = reports / f"{prefix.stem}_regret_smoke.json"
        regret_csv = reports / f"{prefix.stem}_regret_smoke.csv"
        baseline_json = reports / f"{prefix.stem}_baseline_regret_smoke.json"
        baseline_csv = reports / f"{prefix.stem}_baseline_regret_smoke.csv"

        # Calibrate lambda
        cmd_calib = [
            py_mb,
            str(project_root / "tools" / "calibrate_matbench_srmf.py"),
            "--path",
            csv_path,
            "--objective-col",
            objective,
            "--mode",
            mode,
            "--topk-grid",
            topk,
            "--lambda-grid",
            lambdas,
            "--bootstrap",
            bootstrap,
            "--seed",
            "0",
            "--out-json",
            str(calib_json),
            "--scores-out",
            str(scores_csv),
        ]
        calib_res = run_cmd(cmd_calib, cwd=project_root, env=mb_env, timeout=args.timeout_matbench)

        # Extract best lambda
        best_lambda = "0.0"
        try:
            import json as _json

            best_lambda = str(_json.loads(calib_json.read_text())["best_lambda"])
        except Exception:
            best_lambda = "0.0"

        # Evaluate regret with tuned lambda
        cmd_regret = [
            py_mb,
            str(project_root / "tools" / "eval_matbench_regret.py"),
            "--path",
            csv_path,
            "--objective-col",
            objective,
            "--mode",
            mode,
            "--use-srmf",
            "--lambda-weight",
            best_lambda,
            "--topk-grid",
            topk,
            "--bootstrap",
            bootstrap,
            "--seed",
            "0",
            "--out-csv",
            str(regret_csv),
            "--out-json",
            str(regret_json),
        ]
        regret_res = run_cmd(cmd_regret, cwd=project_root, env=mb_env, timeout=args.timeout_matbench)

        # Baseline ridge regret (small folds)
        cmd_base = [
            py_mb,
            str(project_root / "tools" / "eval_baseline_regret.py"),
            "--path",
            csv_path,
            "--objective-col",
            objective,
            "--model",
            "ridge",
            "--folds",
            "3",
            "--topk-grid",
            topk,
            "--bootstrap",
            bootstrap,
            "--seed",
            "0",
            "--out-csv",
            str(baseline_csv),
            "--out-json",
            str(baseline_json),
        ]
        baseline_res = run_cmd(cmd_base, cwd=project_root, env=mb_env, timeout=args.timeout_matbench)

        run_meta["matbench"] = {
            "calibration": calib_res,
            "regret": regret_res,
            "baseline": baseline_res,
            "best_lambda": best_lambda,
        }

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
