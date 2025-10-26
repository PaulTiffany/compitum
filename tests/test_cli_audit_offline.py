from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from compitum.cli import route_command


def test_cli_offline_and_audit_creates_record(tmp_path: Path) -> None:
    # Ensure offline is toggled and an audit record is written without errors
    args = Namespace(
        prompt="Test prompt for audit.",
        constraints=Path("configs/constraints_us_default.yaml"),
        defaults=Path("configs/router_defaults.yaml"),
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
