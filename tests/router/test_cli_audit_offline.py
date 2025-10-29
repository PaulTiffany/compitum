from argparse import Namespace
from pathlib import Path

from compitum.cli import route_command


def test_cli_offline_audit_writes_record(tmp_path: Path, monkeypatch):
    args = Namespace(
        prompt="Sketch AM-GM.",
        constraints=Path("configs/constraints_us_default.yaml"),
        defaults=Path("configs/router_defaults.yaml"),
        verbose=False,
        trace=False,
        no_controller=True,
        no_metric_update=True,
        offline=True,
        audit=True,
        audit_dir=tmp_path,
        seed=123,
    )
    route_command(args)
    files = list(tmp_path.glob("run_*.json"))
    assert files, "expected an audit record in tmp audit_dir"

