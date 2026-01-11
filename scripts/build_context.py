#!/usr/bin/env python3
"""Build baseline context for SWE-bench instances from signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.context_gen.baseline import build_baseline_context, write_context
from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.utils.paths import (
    CONTEXTS_DIR,
    PROJECT_ROOT,
    SIGNALS_DIR,
    get_context_path,
    get_signals_path,
    repo_to_dirname,
)


def load_signals(signals_path: Path) -> dict:
    """Load signals from JSON file.

    Args:
        signals_path: Path to signals.json.

    Returns:
        Signals dict.

    Raises:
        FileNotFoundError: If signals.json doesn't exist.
    """
    if not signals_path.exists():
        raise FileNotFoundError(
            f"Signals file not found: {signals_path}\n"
            f"Run 'python scripts/build_signals.py' first to generate signals."
        )
    with signals_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build baseline context for SWE-bench instances."
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
        "--signals_root",
        default=str(SIGNALS_DIR),
        help="Root directory for signals files.",
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

    args = parser.parse_args()
    signals_root = Path(args.signals_root)
    contexts_root = Path(args.contexts_root)

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
    print()

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        commit = instance["base_commit"]
        repo_dirname = repo_to_dirname(repo)

        # Output paths
        json_path = contexts_root / repo_dirname / commit / "context.json"
        md_path = contexts_root / repo_dirname / commit / "context.md"

        # Skip if exists (unless --force)
        if md_path.exists() and not args.force:
            print(f"[{i+1}/{len(instances)}] {instance_id} - SKIPPED (exists)")
            skip_count += 1
            continue

        print(f"[{i+1}/{len(instances)}] {instance_id} - processing...")

        try:
            # Load signals
            signals_path = signals_root / repo_dirname / commit / "signals.json"
            signals = load_signals(signals_path)

            # Build context
            context = build_baseline_context(signals, repo, commit)

            # Write outputs
            write_context(context, json_path, md_path)

            # Summary
            total_chars = sum(len(c["body"]) for c in context["cards"])
            print(f"  -> wrote context.md ({total_chars} chars, {len(context['cards'])} cards)")
            success_count += 1

        except FileNotFoundError as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            error_count += 1
        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            error_count += 1

    print()
    print(f"Done. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
