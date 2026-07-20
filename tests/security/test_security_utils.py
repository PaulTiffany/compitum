import hashlib
import re
import time
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


def test_audit_record_commit_defaults_to_none():
    # No existing test constructs AuditRecord without an explicit `commit=`
    # -- the dataclass default (None) was never exercised, so a mutation to
    # e.g. `= ""` would survive.
    record = AuditRecord(
        version="0.0.0",
        offline=False,
        seed=1,
        prompt={},
        config={},
        certificate={},
    )
    assert record.commit is None


def test_write_audit_record_creates_nested_directories(tmp_path: Path):
    # The roundtrip test above passes tmp_path directly, which pytest already
    # creates -- `mkdir(exist_ok=True)` succeeds there regardless of
    # `parents`. Use a not-yet-existing multi-level path so `parents=True`
    # is actually required.
    nested = tmp_path / "a" / "b" / "c"
    out = write_audit_record(
        AuditRecord(
            version="0.0.0", offline=False, seed=1, prompt={}, config={}, certificate={}
        ),
        nested,
    )
    assert out.exists()


def test_write_audit_record_filename_is_plausible_epoch_ms_and_exact_indent(tmp_path: Path):
    # The roundtrip test above only checks the filename's prefix/suffix, not
    # that the embedded number is actually a millisecond epoch timestamp (a
    # `* 1000` -> `/ 1000` or `* 1001` mutation, or `= None`, would all still
    # produce a "run_....json"-shaped name). Bound it against the real clock.
    before_ms = time.time() * 1000
    out = write_audit_record(
        AuditRecord(
            version="0.0.0", offline=False, seed=1, prompt={}, config={}, certificate={}
        ),
        tmp_path,
    )
    after_ms = time.time() * 1000
    ts_ms = int(out.stem.removeprefix("run_"))
    assert before_ms - 1000 <= ts_ms <= after_ms + 1000

    # The written JSON is never checked for its actual formatting -- an
    # `indent=2` -> `indent=3` mutation doesn't change parsed content, only
    # the raw text, so assert the literal indentation directly.
    lines = out.read_text().splitlines()
    assert lines[1].startswith('  "') and not lines[1].startswith('   "')


def test_git_commit_short_resolves_real_repo_head():
    # The roundtrip test above never asserts git_commit_short() returns a
    # non-None, well-formed value -- so a mutation that breaks repo-root
    # resolution (`here = None`, wrong `.parents[N]` index, `repo_root =
    # None`) or that breaks the `errors="ignore"` codec name on either the
    # HEAD or ref-file read (making the internal read raise, silently
    # swallowed by the function's broad `except Exception: return None`)
    # would all survive undetected. This test module lives inside the real
    # compitum git repository, so the default (no `repo_root` arg) code path
    # is exercised end-to-end against real `.git` state.
    result = git_commit_short()
    assert result is not None
    assert re.fullmatch(r"[0-9a-f]{7}", result)
