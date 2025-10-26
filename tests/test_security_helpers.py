from __future__ import annotations

from pathlib import Path

from compitum.security import (
    file_sha256,
    git_commit_short,
    is_offline,
    redact_text,
    redaction_enabled,
)


def test_offline_and_redaction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COMPITUM_OFFLINE", "1")
    assert is_offline() is True
    monkeypatch.setenv("COMPITUM_REDACT", "1")
    assert redaction_enabled() is True

    r = redact_text("secret")
    assert r["len"] == 6 and len(r["sha256"]) == 64

    p = tmp_path / "f.txt"
    p.write_text("abc")
    h = file_sha256(p)
    assert h is not None and len(h) == 64
    assert file_sha256(Path("__does_not_exist__.txt")) is None


def test_git_commit_short_resilient(monkeypatch) -> None:
    # We accept either a short hash or None depending on environment
    c = git_commit_short()
    assert c is None or (isinstance(c, str) and 4 <= len(c) <= 16)
    # Force exception path
    import subprocess as _sp

    def _boom(*args, **kwargs):
        raise RuntimeError("no git here")

    monkeypatch.setattr(_sp, "run", _boom)
    c2 = git_commit_short()
    # Implementation may not use subprocess; tolerate short hash or None
    assert c2 is None or (isinstance(c2, str) and 4 <= len(c2) <= 16)


def test_git_commit_short_git_dir_cases(tmp_path: Path) -> None:
    # Layout: tmp/.git
    git_dir = tmp_path / ".git"
    git_dir.mkdir(parents=True)

    # Case 1: missing HEAD -> None
    assert git_commit_short(repo_root=tmp_path) is None

    # Case 2: HEAD points to ref that does not exist -> None
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    assert git_commit_short(repo_root=tmp_path) is None

    # Case 3: HEAD points to existing ref -> short hash
    ref_path = git_dir / "refs/heads/main"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text("1234567890abcdef", encoding="utf-8")
    assert git_commit_short(repo_root=tmp_path) == "1234567"

    # Case 4: detached HEAD -> short hash from HEAD
    (git_dir / "HEAD").write_text("deadbeefcafebabe", encoding="utf-8")
    assert git_commit_short(repo_root=tmp_path) == "deadbee"

    # Case 5: exception path (HEAD is a directory)
    (git_dir / "HEAD").unlink()
    (git_dir / "HEAD").mkdir()
    assert git_commit_short(repo_root=tmp_path) is None
