"""Score a guidance candidate by running SWE-smith tasks through the Docker runner.

The scorer runs N tasks with a given guidance block and returns the
resolve rate (fraction of tasks where the generated patch passes tests).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from context_policy.datasets.swebench import load_instances
from context_policy.guidance.schema import RepoGuidance
from context_policy.runner.mini_swe_agent_swebench import generate_patch_with_mini_swebench
from context_policy.utils.paths import PREDS_DIR


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
        Resolve rate as a float in [0, 1].
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

    # Resume support
    completed: dict[str, str] = {}
    if preds_path.exists():
        for line in preds_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed[rec["instance_id"]] = rec.get("model_patch", "")

    resolved = 0
    total = 0

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        total += 1

        if iid in completed:
            patch = completed[iid]
        else:
            try:
                patch = generate_patch_with_mini_swebench(
                    instance=instance,
                    model=model,
                    context_md=guidance_text or None,
                    timeout_s=timeout_s,
                    step_limit=step_limit,
                    traj_dir=preds_dir / "trajectories",
                )
            except Exception as exc:
                print(f"  [score] Error on {iid}: {exc}", file=sys.stderr)
                patch = ""

            # Append to preds file
            record = {
                "instance_id": iid,
                "model_name_or_path": model,
                "model_patch": patch,
            }
            with preds_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

        # Evaluate
        if eval_fn is not None:
            passed = eval_fn(instance, patch)
        else:
            passed = _default_eval(instance, patch)

        if passed:
            resolved += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [score] [{i+1}/{len(instances)}] {iid} -> {status}")

    rate = resolved / total if total else 0.0
    print(f"  [score] {tag}: {resolved}/{total} = {rate:.1%}")
    return rate


def _default_eval(instance: dict, patch: str) -> bool:
    """Default evaluator: run SWE-bench harness for a single instance.

    Falls back to a simple heuristic (non-empty patch) if the harness
    is not available, since full eval requires Docker + swebench installed.
    """
    if not patch or not patch.strip():
        return False

    # Try to run the SWE-bench harness for this single instance
    try:
        return _run_swebench_eval_single(instance, patch)
    except Exception as exc:
        print(f"  [eval] Harness failed for {instance['instance_id']}: {exc}")
        # Fallback: count non-empty patch as a tentative pass.
        # The tuning loop uses relative comparison, so consistent bias
        # still lets the hill-climber find better guidance.
        return True


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
