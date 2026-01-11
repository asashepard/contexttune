"""Subprocess utilities with log streaming."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
) -> int:
    """Run a command, streaming stdout/stderr to files.

    Args:
        cmd: Command and arguments as a list.
        cwd: Working directory for the command.
        env: Environment variables (merged with current env if provided).
        stdout_path: Path to write stdout. Created if needed.
        stderr_path: Path to write stderr. Created if needed.

    Returns:
        The process return code.
    """
    # Prepare environment
    run_env: dict[str, str] | None = None
    if env is not None:
        run_env = {**os.environ, **env}

    # Ensure output directories exist
    if stdout_path:
        stdout_path = Path(stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path:
        stderr_path = Path(stderr_path)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

    # Open files for streaming
    stdout_file: Any = None
    stderr_file: Any = None
    try:
        if stdout_path:
            stdout_file = open(stdout_path, "w", encoding="utf-8")
        if stderr_path:
            stderr_file = open(stderr_path, "w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=run_env,
            stdout=stdout_file if stdout_file else subprocess.PIPE,
            stderr=stderr_file if stderr_file else subprocess.PIPE,
            text=True,
        )
        proc.wait()
        return proc.returncode
    finally:
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
