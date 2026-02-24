"""Docker-native SWE-agent runner using mini-swe-agent with SWE-bench Docker environment."""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from context_policy.git.checkout import checkout_repo
from context_policy.runner.mini_swe_agent import (
    CONTEXT_BLOCK_END,
    CONTEXT_BLOCK_START,
    build_task_with_context,
    check_docker_available,
)
from context_policy.runner.patch_utils import (
    MAX_PATCH_SIZE,
    extract_diff,
    extract_patch_from_trajectory,
)


def _get_instance_docker_image(instance: dict) -> str:
    """Resolve Docker image name for SWE-bench instance.

    Priority:
    1. swebench harness utilities (multiple API variants across versions)
    2. instance.get("image_name") field
    3. Query local Docker for images matching the instance_id substring
    4. Fallback: hardcoded SWE-bench naming convention

    Args:
        instance: Instance dict with instance_id and optional image_name.

    Returns:
        Docker image name string.
    """
    instance_id = instance["instance_id"]

    # ---------- Try swebench helpers (API varies by version) ----------
    # Variant 1: get_instance_docker_image (newer swebench)
    try:
        from swebench.harness.docker_utils import get_instance_docker_image

        image = get_instance_docker_image(instance)
        print(f"  Docker image (swebench helper): {image}")
        return image
    except ImportError:
        pass
    except Exception as exc:
        print(f"  WARNING: get_instance_docker_image failed: {exc}")

    # Variant 2: make_test_spec → spec.instance_image_key (swebench >=2.x)
    try:
        from swebench.harness.test_spec import make_test_spec

        spec = make_test_spec(instance)
        image = spec.instance_image_key
        if image and not image.endswith(":latest"):
            image = f"{image}:latest"
        print(f"  Docker image (test_spec): {image}")
        return image
    except ImportError:
        pass
    except Exception as exc:
        print(f"  WARNING: make_test_spec failed: {exc}")

    # ---------- Instance field ----------
    if image_name := instance.get("image_name"):
        print(f"  Docker image (instance field): {image_name}")
        return image_name

    # ---------- Query Docker daemon for matching image ----------
    # Image names contain the repo and issue number, e.g.
    # swebench/sweb.eval.x86_64.django_1776_django-10097:latest
    # Extract the short id (e.g. "django-10097") to search
    short_id = instance_id.split("__")[-1]  # "django-10097"
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if short_id in line and "sweb.eval" in line:
                    print(f"  Docker image (docker query): {line}")
                    return line
    except Exception:
        pass

    # ---------- Last-resort fallback ----------
    fallback = f"swebench/sweb.eval.x86_64.{instance_id}:latest"
    print(f"  Docker image (fallback): {fallback}")
    return fallback


def _run_agent_in_docker(
    task: str,
    model: str,
    image_name: str,
    repo_dir: Path,
    traj_path: Path,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run mini-swe-agent with DockerEnvironment in a subprocess.

    This function is designed to run in a separate process for timeout enforcement.
    Results are placed in result_queue as (patch_str, error_msg) tuple.

    TODO: If minisweagent API changes, update imports and calls here.
    """
    try:
        # Import inside subprocess to isolate import errors
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        # Create Docker environment pointing to the SWE-bench image
        # DockerEnvironment expects image via config_class kwargs
        env = DockerEnvironment(image=image_name)

        # Create model instance
        # Model name format for litellm is "openai/gpt-4" or similar
        model_instance = LitellmModel(model_name=model)

        # Create agent with model and environment
        agent = DefaultAgent(
            model=model_instance,
            env=env,
        )

        # Run agent with task - returns (patch, trajectory_json_str)
        patch, traj_json = agent.run(task)

        # Save trajectory
        if traj_json:
            try:
                with open(traj_path, "w", encoding="utf-8") as f:
                    f.write(traj_json)
            except Exception:
                pass

        result_queue.put((patch or "", None))

    except ImportError as e:
        result_queue.put(
            (
                "",
                f"mini-swe-agent Docker API not available: {e}. "
                "Ensure mini-swe-agent is installed with Docker support.",
            )
        )
    except Exception as e:
        result_queue.put(("", f"Agent error: {e}"))


def generate_patch_with_mini_swebench(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    timeout_s: int = 600,
    traj_dir: Path | None = None,
) -> str:
    """Generate a patch using mini-swe-agent with SWE-bench Docker environment.

    This runner executes the agent inside the same Docker container environment
    that SWE-bench uses for evaluation, ensuring environment parity.

    Args:
        instance: Instance dict with:
            - instance_id: SWE-bench instance ID
            - repo: Repository (org/repo format)
            - base_commit: Commit SHA to checkout
            - problem_statement: Issue description
            - image_name (optional): Docker image name override
        model: Model name for mini-swe-agent (e.g., "openai/gpt-4").
        context_md: Optional context to prepend to problem statement.
        timeout_s: Timeout for agent run in seconds (default 600 = 10 min).
        traj_dir: Optional directory to save trajectory files.

    Returns:
        Extracted unified diff patch string, or empty string on failure.

    Raises:
        RuntimeError: If Docker is not available.
    """
    # Validate Docker is available
    check_docker_available()

    instance_id = instance["instance_id"]
    repo = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]

    # Checkout repo locally (for reference/logging, even though agent uses Docker env)
    repo_dir = checkout_repo(repo, commit)

    # Resolve Docker image
    image_name = _get_instance_docker_image(instance)

    # Build task with context
    task = build_task_with_context(problem_statement, context_md)

    # Create trajectory file path
    if traj_dir:
        traj_dir = Path(traj_dir)
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"{instance_id}.traj.json"
    else:
        traj_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".traj.json",
            delete=False,
        )
        traj_path = Path(traj_file.name)
        traj_file.close()

    try:
        # Run agent in separate process for timeout enforcement
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_run_agent_in_docker,
            args=(task, model, image_name, repo_dir, traj_path, result_queue),
        )
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            # Timeout: terminate the process
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            print(f"  mini-swe-agent-swebench timed out after {timeout_s}s")
            return ""

        # Get result from queue
        if result_queue.empty():
            print("  mini-swe-agent-swebench: no result returned")
            return ""

        patch, error = result_queue.get(timeout=1)

        if error:
            print(f"  mini-swe-agent-swebench error: {error}")
            return ""

        # Try to extract patch from trajectory if agent didn't return it directly
        if not patch and traj_path.exists():
            patch = extract_patch_from_trajectory(str(traj_path))

        # Safety: reject oversized patches
        if len(patch) > MAX_PATCH_SIZE:
            print(f"  Patch too large ({len(patch)} chars), rejecting")
            return ""

        return patch

    except Exception as e:
        print(f"  mini-swe-agent-swebench error: {e}")
        return ""
    finally:
        # Clean up temp trajectory file if we created one
        if not traj_dir:
            try:
                traj_path.unlink(missing_ok=True)
            except OSError:
                pass


# Introspection helper for debugging mini-swe-agent API
if __name__ == "__main__":
    import sys

    print("Checking mini-swe-agent API availability...")

    try:
        from minisweagent.agents.default import DefaultAgent

        print(f"  DefaultAgent: {DefaultAgent}")
        import inspect

        sig = inspect.signature(DefaultAgent.__init__)
        print(f"  DefaultAgent.__init__ signature: {sig}")
    except ImportError as e:
        print(f"  DefaultAgent not available: {e}")

    try:
        from minisweagent.environments.docker import DockerEnvironment

        print(f"  DockerEnvironment: {DockerEnvironment}")
        import inspect

        sig = inspect.signature(DockerEnvironment.__init__)
        print(f"  DockerEnvironment.__init__ signature: {sig}")
    except ImportError as e:
        print(f"  DockerEnvironment not available: {e}")

    try:
        from swebench.harness.docker_utils import get_instance_docker_image

        print(f"  get_instance_docker_image: {get_instance_docker_image}")
    except ImportError as e:
        print(f"  get_instance_docker_image not available: {e}")

    print("\nDone.")
    sys.exit(0)
