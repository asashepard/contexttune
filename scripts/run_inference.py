#!/usr/bin/env python3
"""Run single-shot inference to generate SWE-bench prediction JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.runner.mini_swe_agent import generate_patch_with_mini
from context_policy.runner.mini_swe_agent_swebench import generate_patch_with_mini_swebench
from context_policy.runner.single_shot import generate_patch
from context_policy.utils.jsonl import read_jsonl
from context_policy.utils.paths import CONTEXTS_DIR, LOGS_DIR, PREDS_DIR, get_context_path
from context_policy.utils.run_id import make_run_id


def get_completed_ids(preds_path: Path) -> set[str]:
    """Get set of already-completed instance IDs from predictions file."""
    if not preds_path.exists():
        return set()
    records = read_jsonl(preds_path)
    return {r["instance_id"] for r in records}


def append_prediction(preds_path: Path, record: dict) -> None:
    """Append a single prediction record to JSONL file."""
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with preds_path.open("a", encoding="utf-8") as f:
        f.write(line)


def write_instance_log(
    logs_dir: Path,
    instance_id: str,
    prompt_len: int,
    diff_extracted: bool,
    diff_preview: str,
) -> None:
    """Write per-instance debug log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{instance_id}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"instance_id: {instance_id}\n")
        f.write(f"prompt_length_chars: {prompt_len}\n")
        f.write(f"diff_extracted: {diff_extracted}\n")
        f.write(f"diff_preview (first 5 lines):\n")
        f.write(diff_preview + "\n")


def load_context(contexts_root: Path, repo: str, commit: str) -> str:
    """Load context file if it exists."""
    context_path = get_context_path(repo, commit)
    # Allow override via contexts_root if different from default
    if contexts_root != CONTEXTS_DIR:
        from context_policy.utils.paths import repo_to_dirname
        context_path = contexts_root / repo_to_dirname(repo) / commit / "context.md"
    if context_path.exists():
        return context_path.read_text(encoding="utf-8")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-shot inference for SWE-bench instances."
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--instance_ids_file",
        default=None,
        help="Path to file with instance IDs (one per line).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to process.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for inference (required).",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run ID. If not provided, auto-generated.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output predictions path. Default: artifacts/preds/<run_id>/preds.jsonl",
    )
    parser.add_argument(
        "--ablation",
        choices=["no_context", "baseline_context"],
        default="no_context",
        help="Ablation mode (default: no_context).",
    )
    parser.add_argument(
        "--contexts_root",
        default="artifacts/contexts",
        help="Root directory for context files (used with baseline_context).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens for model response (default: 1024).",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip model calls; emit empty patches. Useful for plumbing validation.",
    )
    parser.add_argument(
        "--runner",
        choices=["single_shot", "mini_swe_agent", "mini_swe_agent_swebench"],
        default="single_shot",
        help="Runner backend (default: single_shot).",
    )
    parser.add_argument(
        "--cost_limit",
        type=float,
        default=0.0,
        help="Cost limit for mini_swe_agent runner (0 = no limit).",
    )
    parser.add_argument(
        "--step_limit",
        type=int,
        default=30,
        help="Agent step limit for mini_swe_agent_swebench runner (default: 30; 0=unlimited).",
    )

    args = parser.parse_args()

    # Auto-bump timeout for agent runners if the user left the default
    _RUNNER_DEFAULT_TIMEOUT: dict[str, int] = {
        "single_shot": 120,
        "mini_swe_agent": 300,
        "mini_swe_agent_swebench": 600,
    }
    if args.timeout_s == 120 and args.runner in _RUNNER_DEFAULT_TIMEOUT:
        recommended = _RUNNER_DEFAULT_TIMEOUT[args.runner]
        if recommended != args.timeout_s:
            print(
                f"NOTE: bumping --timeout_s from {args.timeout_s} to "
                f"{recommended} (default for {args.runner} runner)."
            )
            args.timeout_s = recommended

    # Generate run_id if not provided
    run_id = args.run_id or make_run_id("infer")

    # Determine output path
    if args.out:
        preds_path = Path(args.out)
    else:
        preds_path = Path("artifacts") / "preds" / run_id / "preds.jsonl"

    # Logs directory
    logs_dir = Path("artifacts") / "logs" / run_id

    print(f"Run ID: {run_id}")
    print(f"Output: {preds_path}")
    print(f"Model: {args.model}")
    print(f"Ablation: {args.ablation}")
    print()

    # Load instance IDs if provided
    instance_ids = None
    if args.instance_ids_file:
        instance_ids = read_instance_ids(args.instance_ids_file)
        print(f"Loaded {len(instance_ids)} instance IDs from {args.instance_ids_file}")

    # Load instances
    print("Loading instances from dataset...")
    instances = load_instances(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=instance_ids,
        limit=args.limit,
    )
    print(f"Loaded {len(instances)} instances")

    # Get already-completed IDs for resume
    completed_ids = get_completed_ids(preds_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")

    # Context root for baseline_context ablation
    contexts_root = Path(args.contexts_root)

    # Process each instance
    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]

        # Skip if already completed
        if instance_id in completed_ids:
            print(f"[{i+1}/{len(instances)}] {instance_id} - SKIPPED (already done)")
            continue

        print(f"[{i+1}/{len(instances)}] {instance_id} - processing...")

        # Determine context
        context_md: str | None = None
        if args.ablation == "baseline_context":
            context_md = load_context(contexts_root, instance["repo"], instance["base_commit"])
            if context_md:
                print(f"  Loaded context: {len(context_md)} chars")
            else:
                context_md = None  # Empty string -> None

        try:
            # Generate patch (or placeholder in dry-run mode)
            if args.dry_run:
                patch = ""  # Empty patch for plumbing validation
            elif args.runner == "single_shot":
                patch = generate_patch(
                    instance=instance,
                    model=args.model,
                    context_md=context_md,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
            elif args.runner == "mini_swe_agent":
                patch = generate_patch_with_mini(
                    instance=instance,
                    model=args.model,
                    context_md=context_md,
                    timeout_s=args.timeout_s,
                    cost_limit=args.cost_limit,
                )
            elif args.runner == "mini_swe_agent_swebench":
                patch = generate_patch_with_mini_swebench(
                    instance=instance,
                    model=args.model,
                    context_md=context_md,
                    timeout_s=args.timeout_s,
                    traj_dir=logs_dir / "trajectories",
                    step_limit=args.step_limit,
                )
            else:
                raise ValueError(f"Unknown runner: {args.runner}")

            # Build prediction record
            record = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": patch,
            }

            # Append to JSONL
            append_prediction(preds_path, record)

            # Write debug log
            # Approximate prompt length (we don't have it here, so estimate)
            diff_lines = patch.split("\n")[:5] if patch else []
            write_instance_log(
                logs_dir=logs_dir,
                instance_id=instance_id,
                prompt_len=len(instance["problem_statement"]),  # rough estimate
                diff_extracted=bool(patch),
                diff_preview="\n".join(diff_lines),
            )

            status = "OK" if patch else "EMPTY"
            print(f"  -> {status} ({len(patch)} chars)")

        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            # Write empty patch on error to allow resume
            record = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": "",
            }
            append_prediction(preds_path, record)

    print()
    print(f"Done. Predictions written to: {preds_path}")


if __name__ == "__main__":
    main()
