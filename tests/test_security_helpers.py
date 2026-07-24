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


def test_git_commit_short_worktree_cases(tmp_path: Path) -> None:
    # Layout: a main repo with refs, plus a separate worktree directory whose
    # .git is a pointer FILE (not a directory) -- the real structure `git
    # worktree add` creates, and the case the original implementation missed.
    main_repo = tmp_path / "main"
    worktree = tmp_path / "worktree"
    main_git = main_repo / ".git"
    worktree_private_git = main_git / "worktrees" / "wt"
    worktree.mkdir(parents=True)
    worktree_private_git.mkdir(parents=True)

    ref_path = main_git / "refs" / "heads" / "feature"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text("abcdef1234567890", encoding="utf-8")

    (worktree / ".git").write_text(f"gitdir: {worktree_private_git}\n", encoding="utf-8")
    (worktree_private_git / "HEAD").write_text("ref: refs/heads/feature", encoding="utf-8")
    (worktree_private_git / "commondir").write_text("../..\n", encoding="utf-8")

    assert git_commit_short(repo_root=worktree) == "abcdef1"

    # Detached HEAD inside a worktree resolves directly, no commondir lookup.
    (worktree_private_git / "HEAD").write_text("1122334455667788", encoding="utf-8")
    assert git_commit_short(repo_root=worktree) == "1122334"

    # A relative gitdir pointer (resolved against repo_root) works too.
    relative_worktree = main_repo / "linked"
    relative_worktree.mkdir(parents=True)
    (relative_worktree / ".git").write_text("gitdir: ../.git/worktrees/wt\n", encoding="utf-8")
    assert git_commit_short(repo_root=relative_worktree) == "1122334"

    # A .git file that isn't a recognized worktree pointer -> None.
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / ".git").write_text("not a gitdir line\n", encoding="utf-8")
    assert git_commit_short(repo_root=bogus) is None

    # No .git at all (neither directory nor file) -> None.
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    assert git_commit_short(repo_root=no_git) is None

    # An absolute commondir pointer is used as-is, without re-resolving
    # against git_dir.
    (worktree_private_git / "commondir").write_text(str(main_git), encoding="utf-8")
    (worktree_private_git / "HEAD").write_text("ref: refs/heads/feature", encoding="utf-8")
    assert git_commit_short(repo_root=worktree) == "abcdef1"
