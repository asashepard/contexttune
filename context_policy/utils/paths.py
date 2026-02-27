"""Path utilities and artifact location constants."""
from __future__ import annotations

from pathlib import Path

# Project root (resolved at import time)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREDS_DIR = ARTIFACTS_DIR / "preds"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CONTEXTS_DIR = ARTIFACTS_DIR / "contexts"
REPOS_CACHE_DIR = ARTIFACTS_DIR / "repos_cache"
WORKTREES_DIR = ARTIFACTS_DIR / "worktrees"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"


def repo_to_dirname(repo: str) -> str:
    """Convert repo slug to filesystem-safe directory name.

    Args:
        repo: Repository in "owner/name" format (e.g., "astropy/astropy").

    Returns:
        Safe directory name (e.g., "astropy__astropy").
    """
    return repo.replace("/", "__")


def get_context_path(repo: str, commit: str, instance_id: str | None = None) -> Path:
    """Get path to context.md for a repo/commit/instance.

    Args:
        repo: Repository in "owner/name" format.
        commit: Commit SHA.
        instance_id: Instance ID (required to avoid collisions).

    Returns:
        Path to context.md.
    """
    base = CONTEXTS_DIR / repo_to_dirname(repo) / commit
    if instance_id:
        return base / instance_id / "context.md"
    return base / "context.md"


def get_worktree_path(repo: str, commit: str) -> Path:
    """Get path to worktree for a repo/commit.

    Args:
        repo: Repository in "owner/name" format.
        commit: Commit SHA.

    Returns:
        Path to worktree directory.
    """
    return WORKTREES_DIR / repo_to_dirname(repo) / commit
