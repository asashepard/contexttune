"""Docker-native SWE-agent runner using mini-swe-agent with SWE-bench Docker environment."""
from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import queue
import subprocess
import tempfile
import time
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
DEFAULT_AGENT_MAX_TOKENS = 1024
DEFAULT_CONTEXT_WINDOW_TOKENS = 32768
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 512
DEFAULT_CONTEXT_BOUNDARY_BUFFER_TOKENS = 32


def _first_nonempty_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _litellm_model_kwargs_from_env() -> dict[str, Any]:
    """Build explicit LiteLLM kwargs from env to avoid provider fallback."""
    model_kwargs: dict[str, Any] = {}

    api_base = _first_nonempty_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "LITELLM_API_BASE")
    if api_base:
        model_kwargs["api_base"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base

    api_key = _first_nonempty_env("OPENAI_API_KEY", "LITELLM_API_KEY")
    if api_key:
        model_kwargs["api_key"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    custom_provider = _first_nonempty_env("LITELLM_CUSTOM_LLM_PROVIDER")
    if custom_provider:
        model_kwargs["custom_llm_provider"] = custom_provider

    api_version = _first_nonempty_env("OPENAI_API_VERSION", "LITELLM_API_VERSION")
    if api_version:
        model_kwargs["api_version"] = api_version

    env_max_tokens = _first_nonempty_env("LITELLM_MAX_TOKENS", "OPENAI_MAX_TOKENS")
    if env_max_tokens:
        try:
            parsed = int(env_max_tokens)
            if parsed > 0:
                model_kwargs["max_tokens"] = parsed
        except ValueError:
            pass

    return model_kwargs


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (heuristic): ~4 chars per token."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _resolve_context_window_tokens() -> int:
    raw = _first_nonempty_env(
        "AGENT_CONTEXT_WINDOW_TOKENS",
        "OPENAI_CONTEXT_WINDOW_TOKENS",
        "LITELLM_CONTEXT_WINDOW_TOKENS",
        "VLLM_MAX_MODEL_LEN",
    )
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def _trim_context_for_token_budget(
    problem_statement: str,
    context_md: str | None,
    *,
    max_output_tokens: int,
    context_window_tokens: int,
    safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
) -> tuple[str, str | None, int, bool, int, int]:
    """Trim context block so prompt leaves room for output tokens.

    Returns:
      (task, trimmed_context, estimated_prompt_tokens, was_trimmed,
       max_input_budget_tokens, trimmed_chars)
    """
    if not context_md:
        task = build_task_with_context(problem_statement, None)
        max_input_budget_tokens = max(
            1,
            context_window_tokens
            - max_output_tokens
            - safety_margin_tokens
            - DEFAULT_CONTEXT_BOUNDARY_BUFFER_TOKENS,
        )
        return task, None, _estimate_tokens(task), False, max_input_budget_tokens, 0

    task_with_context = build_task_with_context(problem_statement, context_md)
    estimated_prompt_tokens = _estimate_tokens(task_with_context)
    max_input_budget_tokens = max(
        1,
        context_window_tokens
        - max_output_tokens
        - safety_margin_tokens
        - DEFAULT_CONTEXT_BOUNDARY_BUFFER_TOKENS,
    )
    if estimated_prompt_tokens <= max_input_budget_tokens:
        return task_with_context, context_md, estimated_prompt_tokens, False, max_input_budget_tokens, 0

    base_task = build_task_with_context(problem_statement, None)
    base_tokens = _estimate_tokens(base_task)
    if base_tokens >= max_input_budget_tokens:
        # Even without context we're near/over budget; drop context entirely.
        trimmed_chars = len(context_md)
        return base_task, None, base_tokens, True, max_input_budget_tokens, trimmed_chars

    remaining_for_context_tokens = max(0, max_input_budget_tokens - base_tokens)
    remaining_for_context_chars = max(0, remaining_for_context_tokens * 4)
    trimmed_context = context_md[:remaining_for_context_chars]
    trimmed_task = build_task_with_context(problem_statement, trimmed_context)
    trimmed_chars = max(0, len(context_md) - len(trimmed_context))
    return (
        trimmed_task,
        trimmed_context,
        _estimate_tokens(trimmed_task),
        True,
        max_input_budget_tokens,
        trimmed_chars,
    )


def _sum_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _extract_token_usage_from_any(node: Any) -> dict[str, int]:
    """Best-effort recursive token extraction from arbitrary trajectory JSON."""
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "usage" in x and isinstance(x["usage"], dict):
                u = x["usage"]
                usage["prompt_tokens"] += _sum_int(u.get("prompt_tokens", 0))
                usage["completion_tokens"] += _sum_int(u.get("completion_tokens", 0))
                usage["total_tokens"] += _sum_int(u.get("total_tokens", 0))

            usage["prompt_tokens"] += _sum_int(x.get("input_tokens", 0))
            usage["completion_tokens"] += _sum_int(x.get("output_tokens", 0))
            usage["total_tokens"] += _sum_int(x.get("tokens", 0))

            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(node)
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _read_traj_token_usage(traj_path: Path) -> dict[str, int]:
    if not traj_path.exists():
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return _extract_token_usage_from_any(data)


def _docker_image_exists(image_name: str) -> bool:
    """Return True if Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


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

    # Include untracked/new files by staging everything first, then reading cached diff.
    # Try common repo locations in SWE-bench containers.
    for workdir in ["/testbed", "/workspace", "/repo"]:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-lc",
                    f"cd {workdir} && git add -A && git diff --cached",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                diff = result.stdout.strip()
                print(f"  Extracted cached diff from container at {workdir} ({len(diff)} chars)")
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
        import yaml
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_path
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        # ---- Load SWE-bench config shipped with mini-swe-agent ----
        # This config contains the prompts that instruct the agent to edit
        # files in /testbed and submit with `git add -A && git diff --cached`.
        # Without it, the agent uses bare defaults that don't produce patches.
        swebench_config: dict = {}
        try:
            cfg_path = get_config_path("swebench")
            swebench_config = yaml.safe_load(cfg_path.read_text()) or {}
            print(f"  Loaded SWE-bench config from {cfg_path}")
        except Exception as exc:
            print(f"  WARNING: could not load swebench.yaml: {exc}")

        agent_kwargs = dict(swebench_config.get("agent", {}))
        env_config = dict(swebench_config.get("environment", {}))

        # ---- Build components ----
        # DockerEnvironment starts the container in __init__.
        # Pass cwd and env from swebench config for proper /testbed workdir.
        env_cwd = env_config.get("cwd", "/testbed")
        env_vars = env_config.get("env", {})
        # Remove keys that DockerEnvironmentConfig doesn't accept
        # (e.g. environment_class, forward_env if not in config)
        env_config.pop("environment_class", None)

        env = DockerEnvironment(
            image=image_name,
            cwd=env_cwd,
            env=env_vars,
        )

        # Capture container ID immediately after env creation
        container_id = _get_running_container_id()
        print(f"  Container ID after env init: {container_id}")

        model_kwargs = _litellm_model_kwargs_from_env()
        model_kwargs.setdefault("max_tokens", DEFAULT_AGENT_MAX_TOKENS)
        chosen_max_tokens = model_kwargs.get("max_tokens")
        estimated_prompt_tokens = _estimate_tokens(task)
        context_window_tokens = _resolve_context_window_tokens()
        debug_base = model_kwargs.get("api_base") or "<default>"
        debug_provider = model_kwargs.get("custom_llm_provider") or "<auto>"
        print(
            "  Model routing: "
            f"model={model!r}, api_base={debug_base!r}, provider={debug_provider!r}, "
            f"max_tokens={chosen_max_tokens}, est_prompt_tokens={estimated_prompt_tokens}, "
            f"context_window_tokens={context_window_tokens}"
        )
        model_instance = LitellmModel(model_name=model, model_kwargs=model_kwargs)

        # Override step_limit and cost_limit with our experiment values.
        # cost_limit default is $3 which may trigger LimitsExceeded prematurely.
        agent_kwargs["step_limit"] = step_limit
        agent_kwargs["cost_limit"] = 1e9  # effectively unlimited for experiment runs

        print(f"  Creating agent: step_limit={step_limit}, cost_limit=1e9 (effectively unlimited)")
        print(f"  Agent config keys: {sorted(agent_kwargs.keys())}")
        agent = DefaultAgent(
            model=model_instance,
            env=env,
            **agent_kwargs,
        )

        # ---- Run the agent ----
        print(f"  Running agent on task ({len(task)} chars)...")
        result_label, result_detail = agent.run(task)
        print(f"  Agent finished: label={result_label!r}, detail_len={len(result_detail)}")

        try:
            raw_detail_path = Path(traj_path).with_suffix(".raw_result_detail.txt")
            raw_detail_path.parent.mkdir(parents=True, exist_ok=True)
            raw_detail_path.write_text(result_detail or "", encoding="utf-8")
        except Exception as exc:
            print(f"  WARNING: failed to write raw result detail: {exc}")

        # ---- Extract diff IMMEDIATELY, before env/container is destroyed ----
        # Re-check container ID (should be same, but just in case)
        current_cid = _get_running_container_id() or container_id
        print(f"  Container ID post-run: {current_cid}")

        try:
            container_git_path = Path(traj_path).with_suffix(".container_git.txt")
            container_git_path.parent.mkdir(parents=True, exist_ok=True)
            if current_cid:
                debug_cmd = [
                    "docker",
                    "exec",
                    current_cid,
                    "bash",
                    "-lc",
                    "cd /testbed && git status --porcelain && echo '===CACHED_DIFF===' && git add -A && git diff --cached",
                ]
                debug_result = subprocess.run(
                    debug_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                debug_output = (debug_result.stdout or "") + ("\n" + debug_result.stderr if debug_result.stderr else "")
            else:
                debug_output = "No container id available for docker exec debug capture.\n"
            container_git_path.write_text(debug_output, encoding="utf-8")
        except Exception as exc:
            print(f"  WARNING: failed to write container git debug dump: {exc}")

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
    result = generate_patch_with_mini_swebench_result(
        instance=instance,
        model=model,
        context_md=context_md,
        timeout_s=timeout_s,
        step_limit=step_limit,
        traj_dir=traj_dir,
    )
    return result.get("patch", "")


def generate_patch_with_mini_swebench_result(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    timeout_s: int = 600,
    step_limit: int = DEFAULT_MAX_STEPS,
    traj_dir: Path | None = None,
) -> dict:
    """Generate patch plus structured run metadata.

    Returns dict with keys:
      - patch: unified diff or ""
      - elapsed_s: wall-clock seconds spent for this instance
      - token_usage: prompt/completion/total token counts when available
      - status: one of ok, timeout, error, missing_image
      - error: optional error message
      - trajectory_path: optional saved trajectory path
    """
    started = time.perf_counter()
    traj_path: Path | None = None

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

    # Preflight: fail fast on missing images with actionable guidance.
    if not _docker_image_exists(image_name):
        instance_id = instance.get("instance_id", "<unknown>")
        print(
            "  ERROR: SWE-bench Docker image not found locally: "
            f"{image_name} (instance_id={instance_id})"
        )
        print(
            "  Build required images first, e.g.:\n"
            "    python scripts/build_docker_images.py --instance_ids_file <ids.txt>"
        )
        return {
            "patch": "",
            "elapsed_s": time.perf_counter() - started,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "status": "missing_image",
            "error": f"missing_docker_image:{image_name}",
            "trajectory_path": None,
        }

    # Build task with context, trimming context block if needed to preserve
    # output-token room near context limits.
    desired_max_tokens_raw = _first_nonempty_env("LITELLM_MAX_TOKENS", "OPENAI_MAX_TOKENS")
    desired_max_tokens = DEFAULT_AGENT_MAX_TOKENS
    if desired_max_tokens_raw:
        try:
            parsed = int(desired_max_tokens_raw)
            if parsed > 0:
                desired_max_tokens = parsed
        except ValueError:
            pass

    context_window_tokens = _resolve_context_window_tokens()
    task, trimmed_context, est_prompt_tokens, was_trimmed, max_input_budget_tokens, trimmed_chars = _trim_context_for_token_budget(
        problem_statement,
        context_md,
        max_output_tokens=desired_max_tokens,
        context_window_tokens=context_window_tokens,
    )
    if was_trimmed:
        before_chars = len(context_md or "")
        after_chars = len(trimmed_context or "")
        print(
            "  Context budget trim applied: "
            f"before={before_chars} chars, after={after_chars} chars, "
            f"est_prompt_tokens={est_prompt_tokens}, max_tokens={desired_max_tokens}, "
            f"context_window_tokens={context_window_tokens}, "
            f"safety_margin_tokens={DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS}, "
            f"boundary_buffer_tokens={DEFAULT_CONTEXT_BOUNDARY_BUFFER_TOKENS}, "
            f"max_input_budget={max_input_budget_tokens}, "
            f"trimmed_chars={trimmed_chars}"
        )
    else:
        print(
            "  Context budget check: "
            f"est_prompt_tokens={est_prompt_tokens}, max_tokens={desired_max_tokens}, "
            f"context_window_tokens={context_window_tokens}, "
            f"safety_margin_tokens={DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS}, "
            f"boundary_buffer_tokens={DEFAULT_CONTEXT_BOUNDARY_BUFFER_TOKENS}, "
            f"max_input_budget={max_input_budget_tokens}, "
            f"trimmed_chars={trimmed_chars}"
        )

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

    trajectory_path_value = str(traj_path) if traj_path is not None else None

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
                return {
                    "patch": container_patch,
                    "elapsed_s": time.perf_counter() - started,
                    "token_usage": _read_traj_token_usage(traj_path),
                    "status": "timeout",
                    "error": f"timeout:{timeout_s}",
                    "trajectory_path": trajectory_path_value,
                }

            # Fall back to trajectory / queue
            salvage = _salvage_patch(traj_path, result_queue)
            return {
                "patch": salvage,
                "elapsed_s": time.perf_counter() - started,
                "token_usage": _read_traj_token_usage(traj_path),
                "status": "timeout",
                "error": f"timeout:{timeout_s}",
                "trajectory_path": trajectory_path_value,
            }

        # Get result from queue (use .get with timeout, not .empty() which is unreliable)
        try:
            patch, error = result_queue.get(timeout=5)
        except queue.Empty:
            print("  mini-swe-agent-swebench: no result in queue")
            # Process exited without putting result — try trajectory
            salvage = _salvage_patch(traj_path, result_queue)
            return {
                "patch": salvage,
                "elapsed_s": time.perf_counter() - started,
                "token_usage": _read_traj_token_usage(traj_path),
                "status": "error",
                "error": "no_result_in_queue",
                "trajectory_path": trajectory_path_value,
            }

        if error:
            print(f"  mini-swe-agent-swebench error: {error}")
            # Even on error, trajectory might contain a partial patch
            salvage = _salvage_patch(traj_path, result_queue)
            return {
                "patch": salvage,
                "elapsed_s": time.perf_counter() - started,
                "token_usage": _read_traj_token_usage(traj_path),
                "status": "error",
                "error": error,
                "trajectory_path": trajectory_path_value,
            }

        # Try to extract patch from trajectory if agent didn't return it directly
        if not patch and traj_path.exists():
            patch = extract_patch_from_trajectory(str(traj_path))

        # Safety: reject oversized patches
        if patch and len(patch) > MAX_PATCH_SIZE:
            print(f"  Patch too large ({len(patch)} chars), rejecting")
            return {
                "patch": "",
                "elapsed_s": time.perf_counter() - started,
                "token_usage": _read_traj_token_usage(traj_path),
                "status": "error",
                "error": "patch_too_large",
                "trajectory_path": trajectory_path_value,
            }

        return {
            "patch": patch or "",
            "elapsed_s": time.perf_counter() - started,
            "token_usage": _read_traj_token_usage(traj_path),
            "status": "ok",
            "error": None,
            "trajectory_path": trajectory_path_value,
        }

    except Exception as e:
        print(f"  mini-swe-agent-swebench error: {e}")
        return {
            "patch": "",
            "elapsed_s": time.perf_counter() - started,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "status": "error",
            "error": str(e),
            "trajectory_path": trajectory_path_value,
        }
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
