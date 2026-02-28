"""Mini-SWE-Agent runner wrapper for patch generation."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from context_policy.git.checkout import checkout_repo
from context_policy.runner.patch_utils import (
    MAX_PATCH_SIZE,
    extract_diff,
    extract_patch_from_trajectory,
)

# Context block delimiters
CONTEXT_BLOCK_START = "# REPO GUIDANCE (AUTO-TUNED)"
CONTEXT_BLOCK_END = "# END REPO GUIDANCE"

# Legacy aliases
_LEGACY_START = "======BEGIN_REPO_CONTEXT====="
_LEGACY_END = "======END_REPO_CONTEXT====="


class DockerNotAvailableError(Exception):
    """Raised when Docker is required but not available."""

    pass


def check_docker_available() -> None:
    """Check if Docker daemon is running and accessible.

    Raises:
        DockerNotAvailableError: If Docker is not available.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise DockerNotAvailableError(
                "mini_swe_agent runner requires Docker daemon running and accessible. "
                f"'docker info' failed with: {result.stderr.decode()[:200]}"
            )
    except FileNotFoundError:
        raise DockerNotAvailableError(
            "mini_swe_agent runner requires Docker to be installed. "
            "'docker' command not found."
        )
    except subprocess.TimeoutExpired:
        raise DockerNotAvailableError(
            "mini_swe_agent runner requires Docker daemon running. "
            "'docker info' timed out."
        )


def build_task_with_context(problem_statement: str, context_md: str | None) -> str:
    """Build task string with optional guidance/context block prepended.

    Args:
        problem_statement: Original issue/problem text.
        context_md: Optional guidance text to prepend.

    Returns:
        Task string with guidance block if provided.
    """
    if not context_md:
        return problem_statement

    return f"""{CONTEXT_BLOCK_START}
{context_md}
{CONTEXT_BLOCK_END}

{problem_statement}"""


def generate_patch_with_mini(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    timeout_s: int = 600,
    cost_limit: float = 0.0,
) -> str:
    """Generate a patch using mini-swe-agent.

    Args:
        instance: Instance dict with instance_id, repo, base_commit, problem_statement.
        model: Model name for mini-swe-agent (e.g., "openai/gpt-4").
        context_md: Optional additional context to prepend to problem statement.
        timeout_s: Timeout for agent run in seconds (default 600 = 10 min).
        cost_limit: Cost limit for the run (0 = no limit).

    Returns:
        Extracted unified diff patch string, or empty string on failure.
    """
    # Check Docker is available (mini-swe-agent may need it)
    # Note: mini-swe-agent --task runs locally, but we check anyway for safety
    # check_docker_available()  # Commented out: base CLI runs locally

    repo = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]

    # Checkout repo at commit
    repo_dir = checkout_repo(repo, commit)

    # Build task with context
    task = build_task_with_context(problem_statement, context_md)

    # Create temp file for trajectory output
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".traj.json",
        delete=False,
    ) as traj_file:
        traj_path = traj_file.name

    try:
        # Build command
        cmd = [
            "mini-swe-agent",
            "--model", model,
            "--task", task,
            "--output", traj_path,
            "--exit-immediately",  # Don't prompt, just run
            "--yolo",  # No confirmation
        ]
        if cost_limit > 0:
            cmd.extend(["--cost-limit", str(cost_limit)])

        # Run mini-swe-agent in the repo directory
        env = os.environ.copy()
        # Ensure OpenAI env vars are passed through
        # (mini-swe-agent uses litellm which reads these)

        result = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )

        if result.returncode != 0:
            # Log but don't crash - return empty patch
            print(f"  mini-swe-agent failed (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return ""

        # Extract patch from trajectory
        patch = extract_patch_from_trajectory(traj_path)

        # Fallback: try to extract diff from stdout
        if not patch and result.stdout:
            patch = extract_diff(result.stdout)

        # Safety: reject oversized patches
        if len(patch) > MAX_PATCH_SIZE:
            return ""

        return patch

    except subprocess.TimeoutExpired:
        print(f"  mini-swe-agent timed out after {timeout_s}s")
        return ""
    except Exception as e:
        print(f"  mini-swe-agent error: {e}")
        return ""
    finally:
        # Clean up trajectory file
        try:
            os.unlink(traj_path)
        except OSError:
            pass
