"""Test manifest extraction with best-effort heuristics."""
from __future__ import annotations

import tomllib
from pathlib import Path


def build_test_manifest(repo_dir: Path) -> dict:
    """Build test manifest with suggested commands and test directories.

    Args:
        repo_dir: Repository root directory.

    Returns:
        Dict with:
        - suggested_commands: sorted list of test commands
        - test_dirs: sorted list of existing test directories
        - notes: sorted list of config file notes
    """
    suggested_commands: set[str] = set()
    test_dirs: list[str] = []
    notes: list[str] = []

    # Check for tox.ini
    tox_ini = repo_dir / "tox.ini"
    if tox_ini.exists():
        suggested_commands.add("tox -q")
        notes.append("tox.ini present")

    # Check for pyproject.toml
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        notes.append("pyproject.toml present")
        try:
            content = pyproject.read_text(encoding="utf-8")
            data = tomllib.loads(content)
            # Check for pytest configuration
            if "tool" in data and "pytest" in data["tool"]:
                suggested_commands.add("pytest -q")
        except (tomllib.TOMLDecodeError, OSError, KeyError):
            pass

    # Check for setup.cfg
    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.exists():
        notes.append("setup.cfg present")

    # Check for test directories
    for test_dir_name in ["tests", "test"]:
        test_dir = repo_dir / test_dir_name
        if test_dir.is_dir():
            test_dirs.append(test_dir_name)
            # If we have a test dir but no pytest command yet, suggest pytest
            if not any("pytest" in cmd for cmd in suggested_commands):
                suggested_commands.add("pytest -q")

    # Sort everything for determinism
    return {
        "suggested_commands": sorted(suggested_commands),
        "test_dirs": sorted(test_dirs),
        "notes": sorted(notes),
    }
