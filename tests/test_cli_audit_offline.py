from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from compitum.cli import route_command


def test_cli_offline_and_audit_creates_record(tmp_path: Path) -> None:
    # Ensure offline is toggled and an audit record is written without errors
    # Locate repo root that contains configs/
    cur = Path(__file__).resolve().parent
    repo_root = None
    for p in [cur] + list(cur.parents):
        if (p / "configs" / "router_defaults.yaml").exists():
            repo_root = p
            break
    assert repo_root is not None, "Could not locate repo root with configs/"

    args = Namespace(
        prompt="Test prompt for audit.",
        constraints=repo_root / "configs" / "constraints_us_default.yaml",
        defaults=repo_root / "configs" / "router_defaults.yaml",
        verbose=False,
        trace=False,
        seed=123,
        offline=True,
        audit=True,
        audit_dir=tmp_path,
        no_controller=False,
        no_metric_update=False,
    )

    # Call command; it will print to stdout but we only care about side effect
    route_command(args)

    # Verify audit file exists and is redacted
    files = list(tmp_path.glob("run_*.json"))
    assert files, "Expected a redacted audit record to be created"
    data = json.loads(files[0].read_text())
    assert data["offline"] is True
    assert "hash" in data["prompt"] and "redaction" in data["prompt"]
    assert data["prompt"]["redaction"]["len"] == len(args.prompt)
    # Prompt content must not appear
    assert args.prompt not in files[0].read_text()
