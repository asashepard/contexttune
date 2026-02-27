#!/usr/bin/env python3
"""Build context artifacts for SWE-bench instances without repo signal walking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.context_gen.baseline import (
    generate_context,
    generate_context_dry_run,
    write_context,
)
from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.policy.state import default_policy, load_policy
from context_policy.utils.paths import (
    CONTEXTS_DIR,
    repo_to_dirname,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build baseline/tuned context artifacts for SWE-bench instances."
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
        "--tasks_file",
        default=None,
        help="Path to normalized task JSON/JSONL file (overrides dataset loading).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to process.",
    )
    parser.add_argument(
        "--contexts_root",
        default=str(CONTEXTS_DIR),
        help="Root directory for context outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing context files.",
    )
    parser.add_argument(
        "--policy_file",
        default=None,
        help="Optional policy JSON file for policy-driven context generation.",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "tuned"],
        help="Context mode: baseline uses default policy, tuned uses --policy_file if provided.",
    )
    parser.add_argument(
        "--round_id",
        default=None,
        help="Optional adaptive round ID metadata field.",
    )
    parser.add_argument(
        "--source_task_batch",
        default=None,
        help="Optional task batch source metadata field.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for LLM context generation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip LLM calls; emit placeholder context.",
    )

    args = parser.parse_args()
    contexts_root = Path(args.contexts_root)
    if args.mode == "baseline":
        policy = default_policy()
        policy["policy_version"] = "baseline"
    else:
        policy = load_policy(args.policy_file) if args.policy_file else default_policy()
        if not policy.get("policy_version"):
            policy["policy_version"] = "tuned"

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
        tasks_file=args.tasks_file,
    )
    print(f"Loaded {len(instances)} instances")
    print()

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        commit = instance["base_commit"]
        repo_dirname = repo_to_dirname(repo)

        # Output paths (include instance_id to prevent collisions)
        json_path = contexts_root / repo_dirname / commit / instance_id / "context.json"
        md_path = contexts_root / repo_dirname / commit / instance_id / "context.md"

        # Skip if exists (unless --force)
        if md_path.exists() and not args.force:
            print(f"[{i+1}/{len(instances)}] {instance_id} - SKIPPED (exists)")
            skip_count += 1
            continue

        print(f"[{i+1}/{len(instances)}] {instance_id} - processing...")

        try:
            # Build context via LLM (or dry-run placeholder)
            if args.dry_run or not args.model:
                context = generate_context_dry_run(
                    instance,
                    policy=policy,
                    round_id=args.round_id,
                    source_task_batch=args.source_task_batch,
                )
            else:
                context = generate_context(
                    instance,
                    policy=policy,
                    model=args.model,
                    round_id=args.round_id,
                    source_task_batch=args.source_task_batch,
                )

            # Write outputs
            write_context(context, json_path, md_path)

            # Summary
            total_chars = len(context.get("body", ""))
            print(f"  -> wrote context.md ({total_chars} chars)")
            success_count += 1

        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            error_count += 1

    print()
    print(f"Done. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
