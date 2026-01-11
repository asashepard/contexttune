"""Lightweight static import graph extraction."""
from __future__ import annotations

import ast
import os
from pathlib import Path

from context_policy.utils.ignore import should_ignore_dir


def _path_to_module(repo_dir: Path, py_file: Path) -> str:
    """Convert a .py file path to module name."""
    rel = py_file.relative_to(repo_dir)
    parts = rel.with_suffix("").parts
    return ".".join(parts)


def _get_python_files(repo_dir: Path) -> list[Path]:
    """Get all Python files in repo, respecting ignore rules."""
    py_files = []
    for root, dirs, files in os.walk(repo_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(root_path / fname)
    return py_files


def _extract_imports(source: str) -> list[str]:
    """Extract imported module names from Python source.

    Skips relative imports (from . import, from .. import).

    Args:
        source: Python source code.

    Returns:
        List of imported module names (may have duplicates).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (level > 0)
            if node.level > 0:
                continue
            if node.module:
                imports.append(node.module)

    return imports


def _find_local_module(imported: str, local_modules_set: set[str]) -> str | None:
    """Find the best matching local module for an import.

    Args:
        imported: The imported module name (e.g., "pkg.sub").
        local_modules_set: Set of known local module names.

    Returns:
        The matching local module name, or None if not local.
    """
    # Direct match
    if imported in local_modules_set:
        return imported

    # Check if it's a prefix of any local module (e.g., "pkg" when "pkg.sub" exists)
    for local_mod in local_modules_set:
        if local_mod.startswith(imported + "."):
            return imported

    # Check if any local module is a prefix (e.g., importing "pkg.sub.func" when "pkg.sub" exists)
    parts = imported.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in local_modules_set:
            return prefix

    return None


def build_import_graph(repo_dir: Path, local_modules: list[str]) -> dict:
    """Build a static import graph of local modules.

    Args:
        repo_dir: Repository root directory.
        local_modules: List of local module names from py_index.

    Returns:
        Dict with:
        - nodes: sorted list of local module names
        - edges: sorted list of [src_module, dst_module] edges
    """
    local_modules_set = set(local_modules)
    edges: set[tuple[str, str]] = set()

    py_files = _get_python_files(repo_dir)

    for py_file in py_files:
        src_module = _path_to_module(repo_dir, py_file)

        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        imported_names = _extract_imports(source)

        for imported in imported_names:
            dst_module = _find_local_module(imported, local_modules_set)
            if dst_module and dst_module != src_module:
                edges.add((src_module, dst_module))

    # Sort for determinism
    nodes = sorted(local_modules_set)
    edges_list = sorted([list(e) for e in edges])

    return {
        "nodes": nodes,
        "edges": edges_list,
    }
