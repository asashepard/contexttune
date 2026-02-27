"""Adaptive context tuning loop orchestrator."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from context_policy.context_gen.baseline import (
    generate_context,
    generate_context_dry_run,
    write_context,
)
from context_policy.datasets.swebench import load_instances
from context_policy.datasets.swesmith_adapter import generate_swesmith_tasks, import_swesmith_tasks
from context_policy.loop.contracts import LoopState, RoundState, read_loop_state, write_loop_state
from context_policy.policy.state import default_policy, load_policy, write_policy
from context_policy.policy.update import update_policy_with_llm
from context_policy.report.summarize import (
    compute_rate,
    load_instance_records,
    load_results,
    summarize_failure_taxonomy,
)
from context_policy.runner.mini_swe_agent import generate_patch_with_mini
from context_policy.runner.mini_swe_agent_swebench import generate_patch_with_mini_swebench
from context_policy.runner.single_shot import generate_patch
from context_policy.utils.jsonl import read_jsonl
from context_policy.utils.paths import PREDS_DIR, PROJECT_ROOT, RESULTS_DIR, repo_to_dirname
from context_policy.utils.subproc import run as subproc_run


def _run_step(name: str, cmd: list[str], logs_dir: Path, timeout_s: int = 1800) -> int:
    """Run an external command (eval scripts only)."""
    stdout_log = logs_dir / f"{name}.stdout.log"
    stderr_log = logs_dir / f"{name}.stderr.log"
    return subproc_run(cmd, cwd=PROJECT_ROOT, stdout_path=stdout_log, stderr_path=stderr_log, timeout_s=timeout_s)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# In-process context build
# ---------------------------------------------------------------------------

def _build_contexts(
    *,
    instances: list[dict],
    policy: dict,
    model: str,
    contexts_root: Path,
    round_id: str | None = None,
    source_task_batch: str | None = None,
    timeout_s: int = 120,
    dry_run: bool = False,
) -> int:
    """Build context artifacts for all instances in-process.

    Returns:
        Number of contexts successfully built.
    """
    success = 0
    for instance in instances:
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        commit = instance["base_commit"]
        repo_dirname = repo_to_dirname(repo)

        json_path = contexts_root / repo_dirname / commit / instance_id / "context.json"
        md_path = contexts_root / repo_dirname / commit / instance_id / "context.md"

        try:
            if dry_run:
                ctx = generate_context_dry_run(
                    instance, policy=policy,
                    round_id=round_id, source_task_batch=source_task_batch,
                )
            else:
                ctx = generate_context(
                    instance, policy=policy, model=model,
                    round_id=round_id, source_task_batch=source_task_batch,
                    timeout_s=timeout_s,
                )
            write_context(ctx, json_path, md_path)
            success += 1
        except Exception as exc:
            print(f"  Context build error for {instance_id}: {exc}", file=sys.stderr)
    return success


# ---------------------------------------------------------------------------
# In-process inference
# ---------------------------------------------------------------------------

def _load_context_md(contexts_root: Path, repo: str, commit: str, instance_id: str) -> str:
    """Load context.md for an instance, return empty string if missing."""
    md_path = contexts_root / repo_to_dirname(repo) / commit / instance_id / "context.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return ""


def _run_inference(
    *,
    instances: list[dict],
    model: str,
    runner: str,
    contexts_root: Path,
    preds_path: Path,
    logs_dir: Path,
    timeout_s: int,
    step_limit: int,
    dry_run: bool = False,
) -> None:
    """Run inference for all instances in-process, appending to preds_path."""
    # Resume support
    completed_ids: set[str] = set()
    if preds_path.exists():
        completed_ids = {r["instance_id"] for r in read_jsonl(preds_path)}

    preds_path.parent.mkdir(parents=True, exist_ok=True)

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        if instance_id in completed_ids:
            continue

        context_md = _load_context_md(
            contexts_root, instance["repo"], instance["base_commit"], instance_id,
        )

        try:
            if dry_run:
                patch = ""
            elif runner == "single_shot":
                patch = generate_patch(
                    instance=instance, model=model,
                    context_md=context_md or None,
                    temperature=0.0, top_p=1.0, max_tokens=1024,
                    timeout_s=timeout_s,
                )
            elif runner == "mini_swe_agent":
                patch = generate_patch_with_mini(
                    instance=instance, model=model,
                    context_md=context_md or None,
                    timeout_s=timeout_s, cost_limit=0.0,
                )
            elif runner == "mini_swe_agent_swebench":
                patch = generate_patch_with_mini_swebench(
                    instance=instance, model=model,
                    context_md=context_md or None,
                    timeout_s=timeout_s,
                    traj_dir=logs_dir / "trajectories",
                    step_limit=step_limit,
                )
            else:
                raise ValueError(f"Unknown runner: {runner}")
        except Exception as exc:
            print(f"  Inference error for {instance_id}: {exc}", file=sys.stderr)
            patch = ""

        record = {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": patch,
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        with preds_path.open("a", encoding="utf-8") as f:
            f.write(line)

        status = "OK" if patch else "EMPTY"
        print(f"  [{i+1}/{len(instances)}] {instance_id} -> {status}")


def run_adaptive_loop(
    *,
    loop_id: str,
    model: str,
    runner: str,
    rounds: int,
    conditions: list[str],
    tasks_file: str | None,
    swesmith_source: str | None,
    swesmith_generate_cmd: str | None,
    timeout_s: int,
    step_limit: int,
    max_workers_eval: int,
    dataset_name: str,
    split: str,
    policy_file: str | None,
    force: bool,
    dry_run: bool = False,
) -> Path:
    loop_root = RESULTS_DIR / loop_id
    logs_root = loop_root / "logs"
    policies_root = loop_root / "policies"
    tasks_root = loop_root / "tasks"
    state_path = loop_root / "loop_state.json"

    loop_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    policies_root.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)

    if state_path.exists() and not force:
        state = read_loop_state(state_path)
    else:
        state = LoopState(
            loop_id=loop_id,
            model=model,
            runner=runner,
            created_at=datetime.now(timezone.utc).isoformat(),
            rounds=[],
        )

    current_policy = load_policy(policy_file) if policy_file else default_policy()
    if not current_policy.get("policy_version"):
        current_policy["policy_version"] = "v0"
    current_policy_path = policies_root / f"{current_policy['policy_version']}.json"
    if not current_policy_path.exists():
        write_policy(current_policy_path, current_policy)

    completed_rounds = {round_state.round_index for round_state in state.rounds}

    for round_index in range(1, rounds + 1):
        round_id = f"{loop_id}__r{round_index:02d}"
        round_logs = logs_root / round_id
        round_logs.mkdir(parents=True, exist_ok=True)

        if round_index in completed_rounds and not force:
            continue

        round_tasks_file = tasks_root / f"{round_id}.jsonl"
        if swesmith_generate_cmd:
            count = generate_swesmith_tasks(
                swesmith_generate_cmd,
                round_tasks_file,
                round_index=round_index,
            )
            print(f"Round {round_index}: generated {count} SWE-Smith tasks")
        elif swesmith_source:
            count = import_swesmith_tasks(swesmith_source, round_tasks_file)
            print(f"Round {round_index}: imported {count} SWE-Smith tasks")
        elif tasks_file:
            imported = import_swesmith_tasks(tasks_file, round_tasks_file)
            print(f"Round {round_index}: imported {imported} tasks")
        else:
            raise ValueError("Provide one of: --tasks_file, --swesmith_source, --swesmith_generate_cmd")

        policy_path = policies_root / f"{current_policy['policy_version']}.json"
        if not policy_path.exists():
            write_policy(policy_path, current_policy)

        contexts_base = Path("artifacts") / "contexts" / loop_id / f"r{round_index:02d}"
        baseline_contexts_root = contexts_base / "baseline"
        tuned_contexts_root = contexts_base / "tuned"

        # Load instances for this round
        instances = load_instances(
            dataset_name=dataset_name,
            split=split,
            tasks_file=str(round_tasks_file),
        )
        print(f"Round {round_index}: loaded {len(instances)} instances")

        # Build contexts in-process
        baseline_policy = default_policy()
        baseline_policy["policy_version"] = "baseline"

        print(f"Round {round_index}: building baseline contexts...")
        baseline_ok = _build_contexts(
            instances=instances,
            policy=baseline_policy,
            model=model,
            contexts_root=baseline_contexts_root,
            round_id=round_id,
            source_task_batch=str(round_tasks_file),
            timeout_s=timeout_s,
            dry_run=dry_run,
        )
        print(f"  baseline contexts built: {baseline_ok}/{len(instances)}")

        print(f"Round {round_index}: building tuned contexts...")
        tuned_ok = _build_contexts(
            instances=instances,
            policy=current_policy,
            model=model,
            contexts_root=tuned_contexts_root,
            round_id=round_id,
            source_task_batch=str(round_tasks_file),
            timeout_s=timeout_s,
            dry_run=dry_run,
        )
        print(f"  tuned contexts built: {tuned_ok}/{len(instances)}")

        round_results: dict[str, dict] = {}

        for condition in conditions:
            cond_run_id = f"{round_id}__{condition}"
            preds_path = PREDS_DIR / loop_id / f"r{round_index:02d}" / condition / "preds.jsonl"
            cond_logs = logs_root / round_id / condition

            if condition == "baseline":
                cond_contexts_root = baseline_contexts_root
            elif condition == "tuned":
                cond_contexts_root = tuned_contexts_root
            else:
                raise ValueError(f"Unsupported condition: {condition}. Expected baseline,tuned")

            # Run inference in-process
            print(f"Round {round_index}: running inference ({condition})...")
            _run_inference(
                instances=instances,
                model=model,
                runner=runner,
                contexts_root=cond_contexts_root,
                preds_path=preds_path,
                logs_dir=cond_logs,
                timeout_s=timeout_s,
                step_limit=step_limit,
                dry_run=dry_run,
            )

            # Evaluation (external — needs Docker/SWE-bench harness)
            eval_cmd = [
                "bash", "scripts/run_swebench_eval.sh",
                dataset_name,
                str(preds_path),
                cond_run_id,
                str(max_workers_eval),
            ]
            round_logs = logs_root / round_id
            round_logs.mkdir(parents=True, exist_ok=True)
            if _run_step(f"r{round_index:02d}_eval_{condition}", eval_cmd, round_logs, timeout_s=3600) != 0:
                print(f"Warning: evaluation failed for condition {condition}")

            cond_results_dir = RESULTS_DIR / cond_run_id
            resolved, total = load_results(cond_results_dir)
            instance_records = load_instance_records(cond_results_dir)
            failure_taxonomy = summarize_failure_taxonomy(instance_records)
            round_results[condition] = {
                "run_id": cond_run_id,
                "resolved": resolved,
                "total": total,
                "rate": compute_rate(resolved, total),
                "failure_taxonomy": failure_taxonomy,
                "instance_record_count": len(instance_records),
                "preds_path": str(preds_path),
                "results_dir": str(cond_results_dir),
            }

        baseline = round_results.get("baseline")
        tuned = round_results.get("tuned")
        delta = None
        if baseline and tuned and baseline["total"]:
            delta_resolved = tuned["resolved"] - baseline["resolved"]
            delta = {
                "resolved": delta_resolved,
                "rate": delta_resolved / baseline["total"],
            }

        round_summary = {
            "loop_id": loop_id,
            "round_index": round_index,
            "round_id": round_id,
            "policy_version": current_policy["policy_version"],
            "policy_file": str(policy_path),
            "task_batch_file": str(round_tasks_file),
            "conditions": round_results,
            "delta": delta,
        }
        _write_json(loop_root / f"round_{round_index:02d}_summary.json", round_summary)

        next_policy_version = f"v{round_index}"
        next_policy = update_policy_with_llm(
            model=model,
            current_policy=current_policy,
            round_summary=round_summary,
            next_version=next_policy_version,
            timeout_s=timeout_s,
        )
        next_policy_path = policies_root / f"{next_policy_version}.json"
        write_policy(next_policy_path, next_policy)
        current_policy = next_policy

        state.rounds.append(
            RoundState(
                round_index=round_index,
                round_id=round_id,
                policy_version=round_summary["policy_version"],
                task_batch_file=str(round_tasks_file),
                conditions=conditions,
                results=round_results,
            )
        )
        write_loop_state(state_path, state)

    return state_path
