"""Adaptive context tuning loop orchestrator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

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
from context_policy.utils.paths import PREDS_DIR, PROJECT_ROOT, RESULTS_DIR
from context_policy.utils.subproc import run as subproc_run


def _run_step(name: str, cmd: list[str], logs_dir: Path) -> int:
    stdout_log = logs_dir / f"{name}.stdout.log"
    stderr_log = logs_dir / f"{name}.stderr.log"
    return subproc_run(cmd, cwd=PROJECT_ROOT, stdout_path=stdout_log, stderr_path=stderr_log)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
            created_at=datetime.utcnow().isoformat(),
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

        build_baseline_context_cmd = [
            "python", "scripts/build_context.py",
            "--dataset_name", dataset_name,
            "--split", split,
            "--tasks_file", str(round_tasks_file),
            "--mode", "baseline",
            "--contexts_root", str(baseline_contexts_root),
            "--round_id", round_id,
            "--source_task_batch", str(round_tasks_file),
        ]
        if _run_step(f"r{round_index:02d}_build_context_baseline", build_baseline_context_cmd, round_logs) != 0:
            raise RuntimeError(f"Round {round_index} baseline context build failed")

        build_tuned_context_cmd = [
            "python", "scripts/build_context.py",
            "--dataset_name", dataset_name,
            "--split", split,
            "--tasks_file", str(round_tasks_file),
            "--mode", "tuned",
            "--policy_file", str(policy_path),
            "--contexts_root", str(tuned_contexts_root),
            "--round_id", round_id,
            "--source_task_batch", str(round_tasks_file),
        ]
        if _run_step(f"r{round_index:02d}_build_context_tuned", build_tuned_context_cmd, round_logs) != 0:
            raise RuntimeError(f"Round {round_index} tuned context build failed")

        round_results: dict[str, dict] = {}

        for condition in conditions:
            cond_run_id = f"{round_id}__{condition}"
            preds_path = PREDS_DIR / loop_id / f"r{round_index:02d}" / condition / "preds.jsonl"

            infer_cmd = [
                "python", "scripts/run_inference.py",
                "--dataset_name", dataset_name,
                "--split", split,
                "--tasks_file", str(round_tasks_file),
                "--model", model,
                "--ablation", condition,
                "--runner", runner,
                "--timeout_s", str(timeout_s),
                "--step_limit", str(step_limit),
                "--run_id", cond_run_id,
                "--out", str(preds_path),
            ]
            if condition == "baseline":
                infer_cmd.extend(["--contexts_root", str(baseline_contexts_root)])
            elif condition == "tuned":
                infer_cmd.extend(["--contexts_root", str(tuned_contexts_root)])
            else:
                raise ValueError(f"Unsupported condition: {condition}. Expected baseline,tuned")

            if _run_step(f"r{round_index:02d}_infer_{condition}", infer_cmd, round_logs) != 0:
                raise RuntimeError(f"Round {round_index} inference failed for {condition}")

            eval_cmd = [
                "bash", "scripts/run_swebench_eval.sh",
                dataset_name,
                str(preds_path),
                cond_run_id,
                str(max_workers_eval),
            ]
            if _run_step(f"r{round_index:02d}_eval_{condition}", eval_cmd, round_logs) != 0:
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
