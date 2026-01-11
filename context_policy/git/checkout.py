"""Git worktree-based repo checkout utilities."""
from __future__ import annotations

import subprocess
from pathlib import Path

from context_policy.utils.paths import REPOS_CACHE_DIR, WORKTREES_DIR, repo_to_dirname


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a git command and capture output.

    Args:
        args: Git command arguments (without 'git' prefix).
        cwd: Working directory.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def _get_head_commit(repo_path: Path) -> str | None:
    """Get the current HEAD commit of a repo/worktree.

    Args:
        repo_path: Path to repo or worktree.

    Returns:
        Commit SHA or None if not a git repo.
    """
    code, stdout, _ = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    if code == 0:
        return stdout.strip()
    return None


def _ensure_bare_mirror(repo: str) -> Path:
    """Ensure a bare mirror of the repo exists.

    Args:
        repo: Repository in format "owner/name".

    Returns:
        Path to the bare mirror directory.
    """
    mirror_path = REPOS_CACHE_DIR / f"{repo_to_dirname(repo)}.git"

    if mirror_path.exists():
        # Update the mirror
        _run_git(["fetch", "--all"], cwd=mirror_path)
        return mirror_path

    # Clone as bare mirror
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    code, _, stderr = _run_git(
        ["clone", "--mirror", url, str(mirror_path)]
    )
    if code != 0:
        raise RuntimeError(f"Failed to clone mirror for {repo}: {stderr}")

    return mirror_path


def _ensure_worktree(mirror_path: Path, repo: str, commit: str) -> Path:
    """Ensure a worktree for the given commit exists.

    Args:
        mirror_path: Path to the bare mirror.
        repo: Repository in format "owner/name".
        commit: Commit SHA to checkout.

    Returns:
        Path to the worktree directory.
    """
    worktree_path = WORKTREES_DIR / repo_to_dirname(repo) / commit

    if worktree_path.exists():
        # Verify HEAD matches expected commit
        current_head = _get_head_commit(worktree_path)
        if current_head and current_head.startswith(commit[:7]):
            return worktree_path
        # HEAD mismatch - remove and recreate
        _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=mirror_path)
        # Also try shutil if git worktree remove fails
        if worktree_path.exists():
            import shutil
            shutil.rmtree(worktree_path)

    # Create worktree
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    code, _, stderr = _run_git(
        ["worktree", "add", "--detach", str(worktree_path), commit],
        cwd=mirror_path,
    )
    if code != 0:
        raise RuntimeError(f"Failed to create worktree for {repo}@{commit}: {stderr}")

    return worktree_path


def checkout_repo(repo: str, commit: str) -> Path:
    """Checkout a repository at a specific commit using cached mirror + worktree.

    Uses:
    - artifacts/repos_cache/<repo>.git  (bare mirror)
    - artifacts/worktrees/<repo>/<commit>/  (working tree)

    Args:
        repo: Repository in format "owner/name" (e.g., "django/django").
        commit: Commit SHA to checkout.

    Returns:
        Path to the worktree directory with the repo checked out at the commit.
    """
    mirror_path = _ensure_bare_mirror(repo)
    worktree_path = _ensure_worktree(mirror_path, repo, commit)
    return worktree_path
