"""Optional-dependency capability detection and receipt verification.

Detection never imports ``fabricpc`` or ``jax``; it consults installed
package metadata only, so ordinary ``import compitum`` stays JAX-free and a
missing optional dependency yields a governed ``unavailable`` outcome rather
than an ImportError.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Optional


@dataclass
class FabricPCCapability:
    available: bool
    reason: Optional[str] = None
    fabricpc_version: Optional[str] = None
    jax_version: Optional[str] = None


def fabricpc_capability() -> FabricPCCapability:
    """Report whether the optional FabricPC dependency set is installed."""
    versions = {}
    for package in ("fabricpc", "jax"):
        try:
            versions[package] = _pkg_version(package)
        except PackageNotFoundError:
            return FabricPCCapability(
                available=False,
                reason=f"optional dependency {package!r} is not installed",
            )
    return FabricPCCapability(
        available=True,
        fabricpc_version=versions["fabricpc"],
        jax_version=versions["jax"],
    )


def verify_receipt(receipt_path: Path, checkout: Path) -> Optional[str]:
    """Check an external FabricPC checkout against its pinned receipt.

    Returns ``None`` when the checkout HEAD matches the receipt's commit, or
    a human-readable drift/failure description otherwise. Never raises for
    anticipated conditions: a missing receipt, missing checkout, or unreadable
    git state is a governed refusal reason, not a crash.
    """
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable receipt {receipt_path}: {exc}"
    expected = receipt.get("source", {}).get("commit")
    if not expected:
        return f"receipt {receipt_path} does not pin a source commit"
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"cannot resolve checkout HEAD at {checkout}: {exc}"
    if actual != expected:
        return f"FabricPC checkout drift: receipt={expected}, actual={actual}"
    return None
