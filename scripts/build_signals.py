#!/usr/bin/env python3
"""Build signals for SWE-bench instances."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.git.checkout import checkout_repo
from context_policy.signals.build import build_signals, write_signals

# Signals output directory (absolute)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = _PROJECT_ROOT / "artifacts" / "signals"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build signals for SWE-bench instances."
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
        "--force",
        action="store_true",
        help="Overwrite existing signals files.",
    )

    args = parser.parse_args()

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

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        commit = instance["base_commit"]

        # Output path
        safe_repo = repo.replace("/", "__")
        out_path = SIGNALS_DIR / safe_repo / commit / "signals.json"

        # Skip if exists (unless --force)
        if out_path.exists() and not args.force:
            print(f"[{i+1}/{len(instances)}] {instance_id} - SKIPPED (exists)")
            continue

        print(f"[{i+1}/{len(instances)}] {instance_id} - processing...")

        try:
            # Checkout repo
            repo_dir = checkout_repo(repo, commit)

            # Build signals
            signals = build_signals(repo_dir, repo, commit)

            # Write signals
            write_signals(signals, out_path)

            # Summary stats
            n_modules = len(signals["py_index"]["modules"])
            n_edges = len(signals["import_graph"]["edges"])
            print(f"  -> wrote {out_path.relative_to(_PROJECT_ROOT)}")
            print(f"     modules={n_modules}, edges={n_edges}, hot_paths={len(signals['hot_paths'])}")

        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
