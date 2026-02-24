"""Docker-native SWE-agent runner using mini-swe-agent with SWE-bench Docker environment."""
from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import queue
import subprocess
import tempfile
import traceback
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

# Default step limit to prevent the agent from looping indefinitely.
# Experiment spec v1.1 uses 30 steps.
DEFAULT_MAX_STEPS = 30


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


def _get_running_container_id() -> str | None:
    """Get the container ID of any running minisweagent container."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=minisweagent-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        containers = result.stdout.strip().split()
        return containers[0] if containers else None
    except Exception:
        return None


def _extract_diff_from_container(container_id: str) -> str:
    """Extract git diff from a specific Docker container.

    SWE-bench containers have the repo at /testbed.  After the agent runs
    commands inside the container we can extract whatever changes were made.
    This is the most reliable way to get the patch — it doesn't depend on
    the mini-swe-agent Python API return type at all.
    """
    if not container_id:
        print("  No container ID for diff extraction")
        return ""

    # Check if container is still running
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running = result.stdout.strip().lower() == "true"
        print(f"  Container {container_id[:12]} running: {running}")
        if not running:
            return ""
    except Exception as e:
        print(f"  WARNING: container inspect failed: {e}")
        return ""

    # Try common repo locations in SWE-bench containers
    for workdir in ["/testbed", "/workspace", "/repo"]:
        # Try both unstaged and staged (HEAD) diffs
        for diff_args in [["git", "diff"], ["git", "diff", "HEAD"]]:
            try:
                result = subprocess.run(
                    ["docker", "exec", "-w", workdir, container_id] + diff_args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    diff = result.stdout.strip()
                    cmd_label = " ".join(diff_args)
                    print(f"  Extracted '{cmd_label}' from container at {workdir} ({len(diff)} chars)")
                    return diff
            except Exception:
                continue

    # No diff found — check if the workdir even exists
    for workdir in ["/testbed", "/workspace", "/repo"]:
        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "ls", "-la", workdir],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"  {workdir} exists but no git diff found")
                # Try git status for debugging
                result = subprocess.run(
                    ["docker", "exec", "-w", workdir, container_id, "git", "status", "--short"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                print(f"  git status at {workdir}: {result.stdout.strip()[:200]}")
        except Exception:
            continue

    print("  No diff found in any container workdir")
    return ""


def _run_agent_in_docker(
    task: str,
    model: str,
    image_name: str,
    repo_dir: Path,
    traj_path: Path,
    result_queue: multiprocessing.Queue,
    step_limit: int = DEFAULT_MAX_STEPS,
) -> None:
    """Run mini-swe-agent with DockerEnvironment in a subprocess.

    This function is designed to run in a separate process for timeout enforcement.
    Results are placed in result_queue as (patch_str, error_msg) tuple.

    mini-swe-agent 1.17 API (confirmed via introspection):
      - DefaultAgent(model, env, **kwargs)  → kwargs forwarded to AgentConfig
      - AgentConfig(step_limit=0, cost_limit=3.0, ...)
      - agent.run(task) -> tuple[str, str]  → (result_label, result_detail)
      - DockerEnvironment starts container in __init__, cleans up in __del__

    CRITICAL: The DockerEnvironment destroys the container when agent.run()
    finishes (via __del__ or explicit cleanup). We must extract the diff
    while the env object is still alive — i.e., before the function returns.
    """
    container_id: str | None = None
    env = None

    try:
        # Import inside subprocess to isolate import errors
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        # ---- Build components ----
        # DockerEnvironment starts the container in __init__
        env = DockerEnvironment(image=image_name)

        # Capture container ID immediately after env creation
        container_id = _get_running_container_id()
        print(f"  Container ID after env init: {container_id}")

        model_instance = LitellmModel(model_name=model)

        # Pass step_limit AND cost_limit to AgentConfig via **kwargs.
        # cost_limit default is $3 which may trigger LimitsExceeded prematurely.
        print(f"  Creating agent: step_limit={step_limit}, cost_limit=0 (unlimited)")
        agent = DefaultAgent(
            model=model_instance,
            env=env,
            step_limit=step_limit,
            cost_limit=0.0,  # disable cost limit; we control via step_limit + timeout
        )

        # ---- Run the agent ----
        print(f"  Running agent on task ({len(task)} chars)...")
        result_label, result_detail = agent.run(task)
        print(f"  Agent finished: label={result_label!r}, detail_len={len(result_detail)}")

        # ---- Extract diff IMMEDIATELY, before env/container is destroyed ----
        # Re-check container ID (should be same, but just in case)
        current_cid = _get_running_container_id() or container_id
        print(f"  Container ID post-run: {current_cid}")

        patch = ""
        if current_cid:
            patch = _extract_diff_from_container(current_cid)

        # Fallback: check if agent returned a diff in result_detail
        if not patch and result_detail:
            patch = extract_diff(result_detail)
            if patch:
                print(f"  Extracted patch from agent output ({len(patch)} chars)")

        # ---- Save trajectory / agent output for debugging ----
        try:
            # Try to capture agent's message history
            messages = getattr(agent, "messages", None)
            if messages is None:
                messages = getattr(agent, "history", None)

            traj_data: dict[str, Any] = {
                "result_label": result_label,
                "result_detail": result_detail,
                "patch_len": len(patch) if patch else 0,
                "container_id": container_id,
            }
            if messages:
                traj_data["messages"] = messages

            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(traj_data, f, indent=2, default=str)
            print(f"  Saved agent output to {traj_path}")
        except Exception as e:
            print(f"  WARNING: failed to save trajectory: {e}")

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
        tb_str = traceback.format_exc()
        print(f"  Agent error: {e}\n{tb_str}")
        # Even on error, try to recover diff from the container
        fallback = ""
        cid = _get_running_container_id() or container_id
        if cid:
            fallback = _extract_diff_from_container(cid)
        result_queue.put(
            (fallback, f"Agent error: {e}" if not fallback else None)
        )


def _stop_orphan_containers() -> None:
    """Stop any leftover minisweagent-* Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=minisweagent-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split()
        for cid in container_ids:
            if cid:
                subprocess.run(
                    ["docker", "stop", cid],
                    capture_output=True,
                    timeout=15,
                )
    except Exception:
        pass


def _salvage_patch(traj_path: Path, result_queue: multiprocessing.Queue) -> str:
    """Try to recover a patch from the trajectory file or queue after timeout/crash.

    Returns the patch string, or "" if nothing found.
    """
    # Try the queue first — the process may have finished in the instant
    # between timeout firing and us checking.
    try:
        patch, error = result_queue.get_nowait()
        if patch and not error and len(patch) <= MAX_PATCH_SIZE:
            print(f"  Salvaged patch from queue ({len(patch)} chars)")
            return patch
    except Exception:
        pass

    # Try trajectory file
    if traj_path.exists():
        patch = extract_patch_from_trajectory(str(traj_path))
        if patch and len(patch) <= MAX_PATCH_SIZE:
            print(f"  Salvaged patch from trajectory ({len(patch)} chars)")
            return patch

    return ""


def generate_patch_with_mini_swebench(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    timeout_s: int = 600,
    step_limit: int = DEFAULT_MAX_STEPS,
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
        step_limit: Maximum number of agent steps (default 30; 0=unlimited).
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
            args=(task, model, image_name, repo_dir, traj_path, result_queue, step_limit),
        )
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            # Timeout: terminate the Python process but container stays alive
            # (started with 'sleep 2h', process kill doesn't docker-stop it)
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            print(f"  mini-swe-agent-swebench timed out after {timeout_s}s")

            # CRITICAL: extract diff from Docker container BEFORE stopping it.
            cid = _get_running_container_id()
            container_patch = ""
            if cid:
                container_patch = _extract_diff_from_container(cid)

            # Now stop orphaned Docker containers
            _stop_orphan_containers()

            if container_patch and len(container_patch) <= MAX_PATCH_SIZE:
                print(f"  Recovered patch from container on timeout ({len(container_patch)} chars)")
                return container_patch

            # Fall back to trajectory / queue
            return _salvage_patch(traj_path, result_queue)

        # Get result from queue (use .get with timeout, not .empty() which is unreliable)
        try:
            patch, error = result_queue.get(timeout=5)
        except queue.Empty:
            print("  mini-swe-agent-swebench: no result in queue")
            # Process exited without putting result — try trajectory
            return _salvage_patch(traj_path, result_queue)

        if error:
            print(f"  mini-swe-agent-swebench error: {error}")
            # Even on error, trajectory might contain a partial patch
            return _salvage_patch(traj_path, result_queue)

        # Try to extract patch from trajectory if agent didn't return it directly
        if not patch and traj_path.exists():
            patch = extract_patch_from_trajectory(str(traj_path))

        # Safety: reject oversized patches
        if patch and len(patch) > MAX_PATCH_SIZE:
            print(f"  Patch too large ({len(patch)} chars), rejecting")
            return ""

        return patch or ""

    except Exception as e:
        print(f"  mini-swe-agent-swebench error: {e}")
        return ""
    finally:
        # Clean up temp trajectory file if we created one (keep when traj_dir is set)
        if not traj_dir:
            try:
                traj_path.unlink(missing_ok=True)
            except OSError:
                pass


# Introspection helper for debugging mini-swe-agent API
if __name__ == "__main__":
    import sys

    print("Checking mini-swe-agent API availability...")
    print()

    try:
        from minisweagent.agents.default import DefaultAgent

        print(f"  DefaultAgent: {DefaultAgent}")
        sig = inspect.signature(DefaultAgent.__init__)
        print(f"  DefaultAgent.__init__ signature: {sig}")
        print(f"  __init__ params: {list(sig.parameters.keys())}")

        # Check run() method
        run_sig = inspect.signature(DefaultAgent.run)
        print(f"  DefaultAgent.run signature: {run_sig}")
        print(f"  run() params: {list(run_sig.parameters.keys())}")

        # Print run() source to see how step loop works
        try:
            run_source = inspect.getsource(DefaultAgent.run)
            print(f"\n  DefaultAgent.run source:")
            for line in run_source.split("\n"):
                print(f"    {line}")
        except Exception as e:
            print(f"  Could not get run() source: {e}")

        # Print step() source
        try:
            step_source = inspect.getsource(DefaultAgent.step)
            print(f"\n  DefaultAgent.step source:")
            for line in step_source.split("\n"):
                print(f"    {line}")
        except Exception as e:
            print(f"  Could not get step() source: {e}")

        # List all public methods/attrs
        public = [m for m in dir(DefaultAgent) if not m.startswith("_")]
        print(f"\n  Public members: {public}")
    except ImportError as e:
        print(f"  DefaultAgent not available: {e}")

    print()

    # ---- AgentConfig ----
    try:
        from minisweagent.agents.default import AgentConfig

        print(f"  AgentConfig: {AgentConfig}")
        sig = inspect.signature(AgentConfig.__init__)
        print(f"  AgentConfig.__init__ signature: {sig}")
        print(f"  AgentConfig params: {list(sig.parameters.keys())}")

        # Print source to see all config fields
        try:
            config_source = inspect.getsource(AgentConfig)
            print(f"\n  AgentConfig source:")
            for line in config_source.split("\n")[:40]:
                print(f"    {line}")
        except Exception as e:
            print(f"  Could not get AgentConfig source: {e}")
    except ImportError as e:
        print(f"  AgentConfig not available: {e}")

    print()

    try:
        from minisweagent.environments.docker import DockerEnvironment

        print(f"  DockerEnvironment: {DockerEnvironment}")
        sig = inspect.signature(DockerEnvironment.__init__)
        print(f"  DockerEnvironment.__init__ signature: {sig}")
    except ImportError as e:
        print(f"  DockerEnvironment not available: {e}")

    print()

    try:
        from minisweagent.models.litellm_model import LitellmModel

        print(f"  LitellmModel: {LitellmModel}")
        sig = inspect.signature(LitellmModel.__init__)
        print(f"  LitellmModel.__init__ signature: {sig}")
    except ImportError as e:
        print(f"  LitellmModel not available: {e}")

    print()

    try:
        from swebench.harness.docker_utils import get_instance_docker_image

        print(f"  get_instance_docker_image: {get_instance_docker_image}")
    except ImportError as e:
        print(f"  get_instance_docker_image not available: {e}")

    print("\nDone.")
    sys.exit(0)
