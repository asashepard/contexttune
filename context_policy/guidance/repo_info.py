"""Lightweight repository introspection for guidance initialization.

Performs a quick tree walk + heuristic detection of test commands,
top-level modules, etc.  No external tooling required — just ``os``
and ``pathlib``.
"""
from __future__ import annotations

import os
from pathlib import Path

from context_policy.utils.ignore import should_ignore_dir, should_ignore_file


# ── tree walk ──────────────────────────────────────────────────


def get_repo_tree(repo_dir: Path, *, max_depth: int = 2) -> str:
    """Return an indented directory-tree string (depth-limited).

    Identical to prompt_builder._build_tree but exposed as a public API
    so that guidance init can include it in the LLM prompt.
    """
    lines: list[str] = [f"{repo_dir.name}/"]

    def _walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if not (should_ignore_dir(e.name) if e.is_dir() else should_ignore_file(e.name))]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension, depth + 1)

    _walk(repo_dir, "", 1)
    return "\n".join(lines)


# ── top-level directories ──────────────────────────────────────


def get_top_level_dirs(repo_dir: Path) -> list[str]:
    """Return sorted list of non-ignored top-level directory names."""
    dirs: list[str] = []
    try:
        for entry in repo_dir.iterdir():
            if entry.is_dir() and not should_ignore_dir(entry.name):
                dirs.append(entry.name)
    except PermissionError:
        pass
    return sorted(dirs)


# ── test detection ─────────────────────────────────────────────

_TEST_DIR_NAMES = {"tests", "test", "testing", "spec", "specs"}


def get_test_dirs(repo_dir: Path) -> list[str]:
    """Return relative paths to directories that look like test roots."""
    found: list[str] = []
    for root, dirs, _files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        rel = Path(root).relative_to(repo_dir)
        depth = len(rel.parts)
        if depth > 3:
            dirs.clear()
            continue
        for d in dirs:
            if d.lower() in _TEST_DIR_NAMES:
                found.append(str(rel / d))
    return sorted(found)


def detect_test_command(repo_dir: Path) -> str:
    """Best-effort guess of the test command for a Python repo."""
    # Check for common config files
    if (repo_dir / "pytest.ini").exists() or (repo_dir / "pyproject.toml").exists():
        return "pytest"
    if (repo_dir / "setup.cfg").exists():
        cfg = (repo_dir / "setup.cfg").read_text(encoding="utf-8", errors="ignore")
        if "[tool:pytest]" in cfg:
            return "pytest"
    if (repo_dir / "tox.ini").exists():
        return "tox"
    # Default
    return "pytest"


# ── Python module detection ────────────────────────────────────


def get_python_modules(repo_dir: Path) -> list[str]:
    """Return top-level Python package names (directories with __init__.py)."""
    modules: list[str] = []
    try:
        for entry in repo_dir.iterdir():
            if entry.is_dir() and not should_ignore_dir(entry.name):
                if (entry / "__init__.py").exists():
                    modules.append(entry.name)
    except PermissionError:
        pass
    return sorted(modules)


# ── aggregate repo info ────────────────────────────────────────


def build_repo_info_block(repo_dir: Path) -> str:
    """Build a concise repo-info text block for the LLM guidance init prompt.

    This replaces the old ``signals.json`` pipeline with a fast inline
    approach — no external tools, no artifact files.
    """
    parts: list[str] = []

    tree = get_repo_tree(repo_dir, max_depth=2)
    parts.append(f"## Directory tree (depth=2)\n```\n{tree}\n```")

    top_dirs = get_top_level_dirs(repo_dir)
    parts.append(f"## Top-level directories\n{', '.join(top_dirs)}")

    test_dirs = get_test_dirs(repo_dir)
    if test_dirs:
        parts.append(f"## Test directories\n{', '.join(test_dirs)}")

    test_cmd = detect_test_command(repo_dir)
    parts.append(f"## Likely test command\n`{test_cmd}`")

    modules = get_python_modules(repo_dir)
    if modules:
        parts.append(f"## Python packages\n{', '.join(modules)}")

    return "\n\n".join(parts) + "\n"
