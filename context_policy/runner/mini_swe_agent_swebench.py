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


def _extract_diff_from_containers() -> str:
    """Extract git diff from any running minisweagent Docker container.

    SWE-bench containers have the repo at /testbed.  After the agent runs
    commands inside the container we can extract whatever changes were made.
    This is the most reliable way to get the patch — it doesn't depend on
    the mini-swe-agent Python API return type at all.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=minisweagent-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        containers = result.stdout.strip().split()
        if not containers:
            return ""

        cid = containers[0]
        # Try common repo locations in SWE-bench containers
        for workdir in ["/testbed", "/workspace", "/repo"]:
            # Try both unstaged and staged (HEAD) diffs
            for diff_cmd in [
                ["docker", "exec", "-w", workdir, cid, "git", "diff"],
                ["docker", "exec", "-w", workdir, cid, "git", "diff", "HEAD"],
            ]:
                try:
                    result = subprocess.run(
                        diff_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        diff = result.stdout.strip()
                        cmd_label = " ".join(diff_cmd[-2:])
                        print(f"  Extracted {cmd_label} from container {cid[:12]} at {workdir} ({len(diff)} chars)")
                        return diff
                except Exception:
                    continue
    except Exception as e:
        print(f"  WARNING: container diff extraction failed: {e}")
    return ""


def _run_agent_in_docker(
    task: str,
    model: str,
    image_name: str,
    repo_dir: Path,
    traj_path: Path,
    result_queue: multiprocessing.Queue,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    """Run mini-swe-agent with DockerEnvironment in a subprocess.

    This function is designed to run in a separate process for timeout enforcement.
    Results are placed in result_queue as (patch_str, error_msg) tuple.

    The function is defensively coded: it discovers the actual mini-swe-agent
    API at runtime rather than assuming return types or parameter names.
    """
    try:
        # Import inside subprocess to isolate import errors
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        # ---- Discover API signatures ----
        init_sig = inspect.signature(DefaultAgent.__init__)
        init_params = list(init_sig.parameters.keys())
        print(f"  DefaultAgent.__init__ params: {init_params}")

        # ---- Discover AgentConfig params (step limit lives here) ----
        config_params: list[str] = []
        try:
            from minisweagent.agents.default import AgentConfig
            config_sig = inspect.signature(AgentConfig.__init__)
            config_params = list(config_sig.parameters.keys())
            print(f"  AgentConfig params: {config_params}")
        except ImportError:
            # Try to get it from the default value in DefaultAgent.__init__
            config_cls = init_sig.parameters.get("config_class")
            if config_cls and config_cls.default is not inspect.Parameter.empty:
                try:
                    config_sig = inspect.signature(config_cls.default.__init__)
                    config_params = list(config_sig.parameters.keys())
                    print(f"  config_class params: {config_params}")
                except Exception:
                    pass

        # Print run() source for debugging (helps us understand the step loop)
        try:
            run_source = inspect.getsource(DefaultAgent.run)
            # Print first 30 lines
            lines = run_source.split("\n")[:30]
            print(f"  DefaultAgent.run source (first 30 lines):")
            for line in lines:
                print(f"    {line}")
        except Exception:
            pass

        # ---- Build environment + model ----
        env = DockerEnvironment(image=image_name)
        model_instance = LitellmModel(model_name=model)

        agent_kwargs: dict[str, Any] = {
            "model": model_instance,
            "env": env,
        }

        # Try various parameter names for step limit.
        # DefaultAgent takes **kwargs and forwards them to AgentConfig,
        # so we check BOTH sets of params.
        _STEP_PARAM_NAMES = [
            "max_steps", "max_iterations", "step_limit",
            "n_steps", "max_turns", "steps",
        ]
        all_known_params = set(init_params) | set(config_params)
        step_applied = False
        for step_param in _STEP_PARAM_NAMES:
            if step_param in all_known_params:
                agent_kwargs[step_param] = max_steps
                print(f"  Step limit: {step_param}={max_steps}")
                step_applied = True
                break

        # Even if not in known params, DefaultAgent takes **kwargs.
        # Try passing max_steps anyway — it'll forward to AgentConfig.
        if not step_applied and "kwargs" in init_params:
            agent_kwargs["max_steps"] = max_steps
            print(f"  Step limit: passing max_steps={max_steps} via **kwargs (speculative)")
            step_applied = True

        if not step_applied:
            print(
                f"  WARNING: could not inject step-limit. "
                f"Agent may loop indefinitely."
            )

        agent = DefaultAgent(**agent_kwargs)

        # ---- Run the agent ----
        # run() -> tuple[str, str] confirmed by introspection
        run_sig = inspect.signature(agent.run)
        print(f"  Calling agent.run() (params: {list(run_sig.parameters.keys())})")

        result = agent.run(task)
        print(f"  agent.run() returned type={type(result).__name__}")

        # ---- Extract patch from result (handle ANY return type) ----
        patch = ""
        traj_data = None

        if isinstance(result, tuple):
            if len(result) >= 2:
                patch = result[0] if isinstance(result[0], str) else ""
                traj_data = result[1]
            elif len(result) == 1:
                patch = result[0] if isinstance(result[0], str) else ""
            print(f"  Tuple result: len={len(result)}, patch_len={len(patch)}")

        elif isinstance(result, dict):
            for k in ["patch", "model_patch", "diff", "output"]:
                if k in result and isinstance(result[k], str) and result[k].strip():
                    patch = result[k].strip()
                    break
            traj_data = result.get(
                "trajectory", result.get("traj", result.get("history", None))
            )
            print(f"  Dict result: keys={list(result.keys())}, patch_len={len(patch)}")

        elif isinstance(result, str):
            patch = result
            print(f"  String result: len={len(patch)}")

        elif result is not None:
            # Object with attributes
            for attr in ["patch", "model_patch", "diff", "output"]:
                val = getattr(result, attr, None)
                if isinstance(val, str) and val.strip():
                    patch = val.strip()
                    break
            traj_data = getattr(
                result, "trajectory", getattr(result, "traj", None)
            )
            public_attrs = [a for a in dir(result) if not a.startswith("_")]
            print(
                f"  Object result: type={type(result).__name__}, "
                f"attrs={public_attrs}, patch_len={len(patch)}"
            )
        else:
            print("  None result from agent.run()")

        # ---- Save trajectory data if we got any ----
        if traj_data:
            try:
                if isinstance(traj_data, str):
                    content = traj_data
                elif isinstance(traj_data, (dict, list)):
                    content = json.dumps(traj_data, indent=2)
                else:
                    content = str(traj_data)
                with open(traj_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  Saved trajectory to {traj_path}")
            except Exception as e:
                print(f"  WARNING: failed to save trajectory: {e}")

        # ---- Fallback: extract git diff directly from Docker container ----
        if not patch:
            print("  No patch from agent return value, trying container diff...")
            patch = _extract_diff_from_containers()

        # ---- Fallback: extract from trajectory file ----
        if not patch and traj_path.exists():
            patch = extract_patch_from_trajectory(str(traj_path))
            if patch:
                print(f"  Extracted patch from trajectory file ({len(patch)} chars)")

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
        fallback = _extract_diff_from_containers()
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
    max_steps: int = DEFAULT_MAX_STEPS,
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
        max_steps: Maximum number of agent steps (default 30).
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
            args=(task, model, image_name, repo_dir, traj_path, result_queue, max_steps),
        )
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            # Timeout: terminate the Python process
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            print(f"  mini-swe-agent-swebench timed out after {timeout_s}s")

            # CRITICAL: extract diff from the Docker container BEFORE stopping it.
            # The container is still running (started with 'sleep 2h') and holds
            # whatever file changes the agent made during its run.
            container_patch = _extract_diff_from_containers()

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
