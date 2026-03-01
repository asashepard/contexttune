"""Score a guidance candidate by running SWE-smith tasks through the Docker runner.

The scorer runs N tasks with a given guidance block and returns the
resolve rate (fraction of tasks where the generated patch passes tests).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from context_policy.datasets.swebench import load_instances
from context_policy.guidance.schema import RepoGuidance
from context_policy.runner.mini_swe_agent_swebench import (
    generate_patch_with_mini_swebench_result,
)
from context_policy.utils.paths import PREDS_DIR


@dataclass
class ScoreResult:
    """Detailed result for scoring one guidance candidate."""

    rate: float
    resolved: int
    total: int
    non_empty_patches: int
    total_elapsed_s: float
    token_usage: dict[str, int]
    instance_metrics_path: str

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "resolved": self.resolved,
            "total": self.total,
            "non_empty_patches": self.non_empty_patches,
            "non_empty_patch_rate": (self.non_empty_patches / self.total) if self.total else 0.0,
            "total_elapsed_s": self.total_elapsed_s,
            "mean_elapsed_s": (self.total_elapsed_s / self.total) if self.total else 0.0,
            "token_usage": dict(self.token_usage),
            "instance_metrics_path": self.instance_metrics_path,
        }


def score_candidate(
    guidance: RepoGuidance,
    tasks_file: Path,
    model: str,
    *,
    n_tasks: int = 20,
    timeout_s: int = 600,
    step_limit: int = 30,
    preds_dir: Path | None = None,
    eval_fn: callable | None = None,
) -> float:
    """Compatibility wrapper returning only resolve rate."""
    detailed = score_candidate_detailed(
        guidance=guidance,
        tasks_file=tasks_file,
        model=model,
        n_tasks=n_tasks,
        timeout_s=timeout_s,
        step_limit=step_limit,
        preds_dir=preds_dir,
        eval_fn=eval_fn,
    )
    return detailed.rate


def score_candidate_detailed(
    guidance: RepoGuidance,
    tasks_file: Path,
    model: str,
    *,
    n_tasks: int = 20,
    timeout_s: int = 600,
    step_limit: int = 30,
    preds_dir: Path | None = None,
    eval_fn: callable | None = None,
) -> ScoreResult:
    """Score a guidance candidate against SWE-smith tasks.

    Runs the Docker agent on *n_tasks* tasks from ``tasks_file``, each with
    ``guidance.render()`` prepended to the problem statement.  Returns the
    fraction of tasks where evaluation passes.

    Args:
        guidance: The guidance candidate to evaluate.
        tasks_file: Path to the JSONL task file for this repo.
        model: LLM model name.
        n_tasks: Number of tasks to evaluate.
        timeout_s: Per-task agent timeout.
        step_limit: Agent step limit.
        preds_dir: Directory to write prediction JSONL (for debugging).
        eval_fn: Optional custom evaluator ``(instance, patch) -> bool``.
            Defaults to ``_default_eval`` which calls SWE-bench harness.

    Returns:
        Detailed scoring metrics including resolve rate.
    """
    instances = load_instances(
        dataset_name="",
        split="",
        tasks_file=str(tasks_file),
        limit=n_tasks,
    )

    if not instances:
        print(f"  [score] No instances loaded from {tasks_file}")
        return 0.0

    guidance_text = guidance.render()
    tag = f"{guidance.repo.replace('/', '__')}_v{guidance.version}"

    if preds_dir is None:
        preds_dir = PREDS_DIR / "tuning" / tag
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / "preds.jsonl"
    metrics_path = preds_dir / "instance_metrics.jsonl"

    # Resume support
    completed: dict[str, str] = {}
    completed_metrics: dict[str, dict] = {}
    if preds_path.exists():
        for line in preds_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed[rec["instance_id"]] = rec.get("model_patch", "")
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed_metrics[rec["instance_id"]] = rec

    resolved = 0
    total = 0
    non_empty_patches = 0
    total_elapsed_s = 0.0
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        total += 1

        if iid in completed:
            patch = completed[iid]
            prev = completed_metrics.get(iid, {})
            elapsed_s = float(prev.get("elapsed_s", 0.0))
            usage = prev.get("token_usage", {}) if isinstance(prev, dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)
        else:
            started = time.perf_counter()
            try:
                result = generate_patch_with_mini_swebench_result(
                    instance=instance,
                    model=model,
                    context_md=guidance_text or None,
                    timeout_s=timeout_s,
                    step_limit=step_limit,
                    traj_dir=preds_dir / "trajectories",
                )
                patch = result.get("patch", "")
                elapsed_s = float(result.get("elapsed_s", 0.0) or 0.0)
                usage = result.get("token_usage", {}) if isinstance(result, dict) else {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)
            except Exception as exc:
                print(f"  [score] Error on {iid}: {exc}", file=sys.stderr)
                patch = ""
                elapsed_s = time.perf_counter() - started
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            # Append to preds file
            record = {
                "instance_id": iid,
                "model_name_or_path": model,
                "model_patch": patch,
            }
            with preds_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

            metrics_record = {
                "instance_id": iid,
                "elapsed_s": elapsed_s,
                "patch_non_empty": bool(patch and patch.strip()),
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_record, sort_keys=True, ensure_ascii=False) + "\n")

        # Evaluate
        if eval_fn is not None:
            passed = eval_fn(instance, patch)
        else:
            passed = _default_eval(instance, patch)

        if patch and patch.strip():
            non_empty_patches += 1
        total_elapsed_s += elapsed_s
        token_usage["prompt_tokens"] += prompt_tokens
        token_usage["completion_tokens"] += completion_tokens
        token_usage["total_tokens"] += total_tokens

        if passed:
            resolved += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [score] [{i+1}/{len(instances)}] {iid} -> {status}")

    rate = resolved / total if total else 0.0
    print(f"  [score] {tag}: {resolved}/{total} = {rate:.1%}")
    return ScoreResult(
        rate=rate,
        resolved=resolved,
        total=total,
        non_empty_patches=non_empty_patches,
        total_elapsed_s=total_elapsed_s,
        token_usage=token_usage,
        instance_metrics_path=str(metrics_path),
    )


def _default_eval(instance: dict, patch: str) -> bool:
    """Default evaluator: run SWE-bench harness for a single instance.

    Does not use heuristics: harness failures are counted as failures.
    """
    if not patch or not patch.strip():
        return False

    # Try to run the SWE-bench harness for this single instance
    try:
        return _run_swebench_eval_single(instance, patch)
    except Exception as exc:
        print(f"  [eval] Harness failed for {instance['instance_id']}: {exc}")
        return False


def _run_swebench_eval_single(instance: dict, patch: str) -> bool:
    """Run SWE-bench evaluation for a single instance+patch.

    Uses the swebench Python API if available (swebench >= 2.x).
    """
    import subprocess
    import tempfile

    iid = instance["instance_id"]

    # Write a temporary preds file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="score_"
    ) as f:
        pred = {
            "instance_id": iid,
            "model_name_or_path": "tuning",
            "model_patch": patch,
        }
        f.write(json.dumps(pred) + "\n")
        preds_file = f.name

    try:
        result = subprocess.run(
            [
                "python", "-m", "swebench.harness.run_evaluation",
                "--predictions_path", preds_file,
                "--swe_bench_tasks", instance.get("dataset_name", "princeton-nlp/SWE-bench_Verified"),
                "--log_level", "ERROR",
                "--timeout", "120",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        # Parse output for PASS/FAIL
        if "resolved" in result.stdout.lower() or iid in result.stdout:
            return True
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise
    finally:
        Path(preds_file).unlink(missing_ok=True)
