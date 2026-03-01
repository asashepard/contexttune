"""py-tree-sitter wrapper for parsing Python source files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from context_policy.utils.ignore import should_ignore_dir, should_ignore_file

# Lazy-init globals
_PYTHON_LANGUAGE: Any = None
_PARSER: Any = None


def _ensure_parser() -> tuple[Any, Any]:
    """Lazily initialize the tree-sitter parser and Python language."""
    global _PYTHON_LANGUAGE, _PARSER

    if _PARSER is not None:
        return _PARSER, _PYTHON_LANGUAGE

    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    _PYTHON_LANGUAGE = Language(tspython.language())
    _PARSER = Parser(_PYTHON_LANGUAGE)
    return _PARSER, _PYTHON_LANGUAGE


def parse_file(path: Path) -> Any:
    """Parse a single Python file and return its tree-sitter Tree.

    Args:
        path: Absolute or relative path to a ``.py`` file.

    Returns:
        ``tree_sitter.Tree`` for the file, or *None* on read/parse errors.
    """
    parser, _ = _ensure_parser()
    try:
        source = path.read_bytes()
    except (OSError, PermissionError):
        return None
    return parser.parse(source)


def parse_repo(repo_dir: Path) -> dict[str, Any]:
    """Parse all ``.py`` files in *repo_dir*, respecting ignore patterns.

    Args:
        repo_dir: Root of the checked-out repository.

    Returns:
        Dict mapping *relative* path strings to ``tree_sitter.Tree`` objects.
    """
    trees: dict[str, Any] = {}
    for root, dirs, files in os.walk(repo_dir):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        for fname in files:
            if not fname.endswith(".py") or should_ignore_file(fname):
                continue
            full = Path(root) / fname
            rel = str(full.relative_to(repo_dir)).replace("\\", "/")
            tree = parse_file(full)
            if tree is not None:
                trees[rel] = tree
    return trees
