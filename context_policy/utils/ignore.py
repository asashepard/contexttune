"""Shared ignore patterns for directory traversal."""
from __future__ import annotations

# Directories to ignore when scanning/traversing repos
IGNORE_DIRS: frozenset[str] = frozenset({
    # Version control
    ".git",
    ".hg",
    ".svn",
    # Python
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".eggs",
    ".venv",
    "venv",
    "env",
    ".env",
    # Build outputs
    "dist",
    "build",
    "_build",
    ".build",
    "htmlcov",
    # Node
    "node_modules",
    # Coverage/cache
    ".coverage",
    ".cache",
})

# File patterns to ignore (exact match or suffix with *)
IGNORE_FILES: frozenset[str] = frozenset({
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.egg",
})

# Directory name patterns (suffix match with *)
IGNORE_DIR_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
})


def should_ignore_dir(name: str) -> bool:
    """Check if a directory name should be ignored.

    Args:
        name: Directory name (not full path).

    Returns:
        True if should be ignored.
    """
    if name in IGNORE_DIRS:
        return True
    # Check suffix patterns
    for pattern in IGNORE_DIR_PATTERNS:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False


def should_ignore_file(name: str) -> bool:
    """Check if a file name should be ignored.

    Args:
        name: File name (not full path).

    Returns:
        True if should be ignored.
    """
    if name in IGNORE_FILES:
        return True
    # Check suffix patterns
    for pattern in IGNORE_FILES:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False
