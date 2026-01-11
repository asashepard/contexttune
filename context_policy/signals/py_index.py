"""Python index extraction using AST."""
from __future__ import annotations

import ast
import os
from collections import defaultdict
from pathlib import Path

from context_policy.utils.ignore import should_ignore_dir


def _get_python_files(repo_dir: Path) -> list[Path]:
    """Get all Python files in repo, respecting ignore rules.

    Args:
        repo_dir: Root directory to scan.

    Returns:
        Sorted list of .py file paths (absolute).
    """
    py_files = []
    for root, dirs, files in os.walk(repo_dir):
        root_path = Path(root)
        # Filter ignored directories in-place
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        dirs.sort()

        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(root_path / fname)

    return py_files


def _path_to_module(repo_dir: Path, py_file: Path) -> str:
    """Convert a .py file path to module name.

    Args:
        repo_dir: Repository root.
        py_file: Path to .py file.

    Returns:
        Module name like "pkg.subpkg.module".
    """
    rel = py_file.relative_to(repo_dir)
    # Convert path to module: foo/bar/baz.py -> foo.bar.baz
    parts = rel.with_suffix("").parts
    return ".".join(parts)


def _relative_path(repo_dir: Path, file_path: Path) -> str:
    """Get repo-relative path with forward slashes.

    Args:
        repo_dir: Repository root.
        file_path: Absolute file path.

    Returns:
        Relative path string with forward slashes.
    """
    return file_path.relative_to(repo_dir).as_posix()


def build_py_index(repo_dir: Path) -> dict:
    """Build Python index of modules, functions, and classes.

    Args:
        repo_dir: Repository root directory.

    Returns:
        Dict with:
        - modules: sorted list of module names
        - functions: {name: [sorted paths...]}
        - classes: {name: [sorted paths...]}
        - parse_errors: count of files that failed to parse
    """
    modules: list[str] = []
    functions: dict[str, list[str]] = defaultdict(list)
    classes: dict[str, list[str]] = defaultdict(list)
    parse_errors = 0

    py_files = _get_python_files(repo_dir)

    for py_file in py_files:
        # Add module
        module_name = _path_to_module(repo_dir, py_file)
        modules.append(module_name)

        # Parse AST
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, ValueError):
            parse_errors += 1
            continue

        rel_path = _relative_path(repo_dir, py_file)

        # Extract top-level definitions only
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name].append(rel_path)
            elif isinstance(node, ast.ClassDef):
                classes[node.name].append(rel_path)

    # Sort everything for determinism
    modules.sort()

    functions_sorted = {
        name: sorted(paths) for name, paths in sorted(functions.items())
    }
    classes_sorted = {
        name: sorted(paths) for name, paths in sorted(classes.items())
    }

    return {
        "modules": modules,
        "functions": functions_sorted,
        "classes": classes_sorted,
        "parse_errors": parse_errors,
    }
