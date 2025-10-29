from pathlib import Path

from compitum.security import (
    AuditRecord,
    file_sha256,
    git_commit_short,
    is_offline,
    redact_text,
    write_audit_record,
)


def test_redact_text_structure():
    r = redact_text("hello world")
    assert set(r.keys()) == {"sha256", "len"}
    assert r["len"] == len("hello world")
    # sha256 should be 64 hex chars
    assert isinstance(r["sha256"], str) and len(r["sha256"]) == 64


def test_file_sha256_missing(tmp_path: Path):
    p = tmp_path / "nope.txt"
    assert file_sha256(p) is None


def test_write_audit_record_roundtrip(tmp_path: Path):
    out = write_audit_record(
        AuditRecord(
            version="0.0.0",
            offline=is_offline(),
            seed=123,
            prompt={"hash": "h", "redaction": {"sha256": "0" * 64, "len": 3}},
            config={"constraints_path": "c.yaml", "defaults_path": "d.yaml"},
            certificate={"model": "fast"},
            commit=git_commit_short(),
        ),
        tmp_path,
    )
    assert out.exists()
    assert out.name.startswith("run_") and out.suffix == ".json"

