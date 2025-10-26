from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def is_offline() -> bool:
    """Return True if offline mode is enabled via environment variable.

    Offline mode is enabled when COMPITUM_OFFLINE == "1".
    """
    return os.environ.get("COMPITUM_OFFLINE", "0") == "1"


def redaction_enabled() -> bool:
    """Return True if explicit redaction is requested via environment.

    Redaction mode is enabled when COMPITUM_REDACT == "1".
    """
    return os.environ.get("COMPITUM_REDACT", "0") == "1"


def redact_text(s: str) -> Dict[str, Any]:
    """Return a redacted representation of text without revealing content.

    Provides SHA-256 digest and length. Does not return the original text.
    """
    h = hashlib.sha256(s.encode()).hexdigest()
    return {"sha256": h, "len": len(s)}


def file_sha256(path: Path) -> Optional[str]:
    """Compute SHA-256 of a file, returning None if the path does not exist."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(data).hexdigest()


@dataclass
class AuditRecord:
    version: str
    offline: bool
    seed: int
    prompt: Dict[str, Any]
    config: Dict[str, Any]
    certificate: Dict[str, Any]
    commit: Optional[str] = None


def write_audit_record(record: AuditRecord, out_dir: Path) -> Path:
    """Write an audit record JSON to the given directory.

    The filename is monotonic using UNIX epoch milliseconds to simplify ordering.
    Returns the full path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Monotonic filename: run_<ms>.json
    import time

    ts_ms = int(time.time() * 1000)
    out_path = out_dir / f"run_{ts_ms}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2))
    return out_path


def git_commit_short(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the short git commit hash if available, else None.

    By default, this inspects the repository containing this file by reading
    from the local ``.git`` directory (no subprocess calls). For tests or
    tooling, an explicit ``repo_root`` may be provided to point to a directory
    that contains a ``.git`` folder.
    """
    try:
        if repo_root is None:
            # Resolve repository root (assumes this file lives under src/compitum)
            here = Path(__file__).resolve()
            # repo_root = .../compitum (one up from src)
            repo_root = here.parents[2]
        git_dir = repo_root / ".git"
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text(encoding="utf-8", errors="ignore").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                commit = ref_path.read_text(encoding="utf-8", errors="ignore").strip()
                return commit[:7] if commit else None
            return None
        # Detached HEAD: HEAD contains the commit hash
        return head[:7] if head else None
    except Exception:
        return None
