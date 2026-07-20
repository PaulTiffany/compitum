from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Dict:
    start = datetime.utcnow()
    import time as _t

    t0 = _t.perf_counter()
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "timed_out": False,
            "started_at": start.isoformat() + "Z",
            "duration_sec": round(_t.perf_counter() - t0, 3),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\n[timed out after {timeout}s]",
            "timed_out": True,
            "started_at": start.isoformat() + "Z",
            "duration_sec": round(_t.perf_counter() - t0, 3),
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run quality gates: ruff, mypy, bandit, pytest w/ coverage, cosmic-ray"
    )
    ap.add_argument("--mypy", action="store_true")
    ap.add_argument("--ruff", action="store_true")
    ap.add_argument("--bandit", action="store_true")
    ap.add_argument("--pytest", action="store_true")
    ap.add_argument("--cosmic", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = str(root / ".venv-routerbench" / "Scripts" / "python")
    env = os.environ.copy()
    env.setdefault("HYPOTHESIS_OPTIONAL", "1")
    # Ensure compitum imports resolve during subprocess runs
    pp = env.get("PYTHONPATH", "")
    src_path = str(root / "src")
    env["PYTHONPATH"] = (pp + os.pathsep + src_path) if pp else src_path
    os.environ.update(env)

    results: Dict[str, Dict] = {}

    def add(name: str, cmd: List[str]):
        results[name] = run(cmd, cwd=root, timeout=args.timeout)

    if args.all or args.ruff:
        add("ruff", [py, "-m", "ruff", "check", "--quiet", "src", "tests"])
    if args.all or args.mypy:
        add(
            "mypy",
            [py, "-m", "mypy", "--strict", "--disable-error-code", "no-any-return", "src/compitum"],
        )
    if args.all or args.bandit:
        # Limit Bandit to our core library and exclude vendored benchmark code
        add("bandit", [py, "-m", "bandit", "-q", "-r", "src/compitum", "-x", "src/routerbench"])
    if args.all or args.pytest:
        add(
            "pytest",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not routerbench",
            ],
        )
    if args.all or args.cosmic:
        # Cosmic Ray v8+ CLI changed entrypoints; use cosmic_ray.cli and dump JSON report
        session = str(root / "cr_session.sqlite")
        add("cosmic-ray-init", [py, "-m", "cosmic_ray.cli", "init", "cosmic-ray.toml", session])
        add("cosmic-ray-exec", [py, "-m", "cosmic_ray.cli", "exec", "cosmic-ray.toml", session])
        # Dump JSON to reports for reviewer consumption
        add("cosmic-ray-dump", [py, "-m", "cosmic_ray.cli", "dump", session])

    # Persist reports and post-process cosmic-ray dump to file + summary
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Write cosmic-ray dump stdout to a file for convenience
    if "cosmic-ray-dump" in results:
        dump_stdout = results["cosmic-ray-dump"].get("stdout", "")
        if dump_stdout:
            (reports_dir / "cr_report.json").write_text(dump_stdout, encoding="utf-8")
            # Also write a compact summary
            try:
                from tools.mutation_summary import summarize_dump_text  # type: ignore

                summary = summarize_dump_text(dump_stdout)
                import json as _json

                (reports_dir / "mutation_summary.json").write_text(
                    _json.dumps(summary, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

    out = reports_dir / f"quality_{ts}.json"
    import json as _json

    out.write_text(_json.dumps(results, indent=2), encoding="utf-8")
    print(f"Quality report written to: {out}")


if __name__ == "__main__":
    main()
