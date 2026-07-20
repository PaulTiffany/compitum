from __future__ import annotations

import argparse
import sys
from subprocess import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-line compitum CLI routing demo.")
    parser.add_argument(
        "--prompt",
        default="Prove that the harmonic series diverges.",
        help="Prompt to route.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show full certificate (pass --trace to CLI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed forwarded to CLI for deterministic synthetic fit.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "compitum.cli",
        "route",
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
    ]
    if args.trace:
        cmd.append("--trace")
    run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
