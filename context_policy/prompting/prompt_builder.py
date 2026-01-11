"""Prompt construction for patch generation."""
from __future__ import annotations

import os
from pathlib import Path

# Directories to ignore when building directory tree
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "_build",
    ".build",
    "htmlcov",
    ".coverage",
    ".cache",
}

# File patterns to ignore
IGNORE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.egg",
}


def _should_ignore(name: str) -> bool:
    """Check if a file/directory name should be ignored."""
    if name in IGNORE_DIRS or name in IGNORE_FILES:
        return True
    # Check wildcard patterns
    for pattern in IGNORE_DIRS | IGNORE_FILES:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False


def _build_tree(repo_dir: Path, max_depth: int = 2) -> str:
    """Build a directory tree string with limited depth.

    Args:
        repo_dir: Root directory to scan.
        max_depth: Maximum depth to traverse (default 2).

    Returns:
        Tree representation as string.
    """
    lines = []

    def _walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        # Filter ignored entries
        entries = [e for e in entries if not _should_ignore(e.name)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")

            if entry.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(f"{repo_dir.name}/")
    _walk(repo_dir, "", 1)

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a coding assistant. Output ONLY a unified diff patch that fixes the issue.
Do not include any commentary, explanation, or markdown formatting outside the diff.
The patch must apply cleanly to the repository at the specified commit."""

USER_TEMPLATE = """\
## Repository
{repo} @ {commit}

## Issue
{problem_statement}

## Repository Structure (depth=2)
```
{tree}
```

## Output Requirements
- Output a unified diff patch only
- The patch must apply cleanly with `git apply`
- Make minimal, focused changes
- Do not include unrelated modifications"""

CONTEXT_BLOCK_TEMPLATE = """

======BEGIN_REPO_CONTEXT=====
{context_md}
======END_REPO_CONTEXT====="""


def build_messages(
    problem_statement: str,
    repo: str,
    commit: str,
    repo_dir: Path,
    context_md: str | None = None,
) -> list[dict]:
    """Build OpenAI-format messages for patch generation.

    Args:
        problem_statement: The issue/bug description.
        repo: Repository in format "owner/name".
        commit: Commit SHA.
        repo_dir: Path to the checked-out repository.
        context_md: Optional additional context to append.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    tree = _build_tree(repo_dir, max_depth=2)

    user_content = USER_TEMPLATE.format(
        repo=repo,
        commit=commit,
        problem_statement=problem_statement,
        tree=tree,
    )

    # Append context block if provided
    if context_md:
        user_content += CONTEXT_BLOCK_TEMPLATE.format(context_md=context_md)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
