import hashlib
from pathlib import Path

from compitum.security import (
    AuditRecord,
    file_sha256,
    git_commit_short,
    is_offline,
    redact_text,
    redaction_enabled,
    write_audit_record,
)


def test_redact_text_structure():
    r = redact_text("hello world")
    assert set(r.keys()) == {"sha256", "len"}
    assert r["len"] == len("hello world")
    # sha256 should be 64 hex chars
    assert isinstance(r["sha256"], str) and len(r["sha256"]) == 64


def test_redact_text_exact_hash_value():
    # The structure test above only checks length, not the actual digest --
    # a mutated encoding or wrong-algorithm bug would still produce a 64-char
    # hex string. Assert the exact expected SHA-256 value.
    r = redact_text("hello world")
    assert r["sha256"] == hashlib.sha256(b"hello world").hexdigest()


def test_file_sha256_exact_value(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    assert file_sha256(p) == hashlib.sha256(b"abc").hexdigest()


def test_is_offline_and_redaction_default_to_false(monkeypatch):
    # No existing test checks the *unset* env var case explicitly.
    monkeypatch.delenv("COMPITUM_OFFLINE", raising=False)
    monkeypatch.delenv("COMPITUM_REDACT", raising=False)
    assert is_offline() is False
    assert redaction_enabled() is False


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

