#!/usr/bin/env python3
"""Smoke test for mini_swe_agent_swebench runner.

This script runs a single SWE-bench instance through the docker-native runner
to verify the integration works end-to-end.

Prerequisites:
- Docker daemon running
- Linux/WSL2 environment
- mini-swe-agent installed: pip install mini-swe-agent
- OPENAI_API_KEY or OPENAI_BASE_URL configured

Usage:
    python scripts/smoke_mini_agent_swebench.py

    # With custom model
    python scripts/smoke_mini_agent_swebench.py --model openai/gpt-4

    # With specific instance
    python scripts/smoke_mini_agent_swebench.py --instance_id django__django-16379
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for mini_swe_agent_swebench runner."
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4",
        help="Model name for agent (default: openai/gpt-4).",
    )
    parser.add_argument(
        "--instance_id",
        default=None,
        help="Specific instance ID to test. If not provided, uses first from Verified.",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--with_context",
        action="store_true",
        help="Include baseline context in task.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Smoke Test: mini_swe_agent_swebench runner")
    print("=" * 60)
    print()

    # Step 1: Check Docker availability
    print("[1/5] Checking Docker availability...")
    try:
        from context_policy.runner.mini_swe_agent import check_docker_available

        check_docker_available()
        print("  ✓ Docker is available")
    except Exception as e:
        print(f"  ✗ Docker check failed: {e}")
        print()
        print("This runner requires Docker daemon running on Linux/WSL2.")
        return 1

    # Step 2: Check mini-swe-agent API availability
    print("[2/5] Checking mini-swe-agent API...")
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment

        print(f"  ✓ DefaultAgent available: {DefaultAgent}")
        print(f"  ✓ DockerEnvironment available: {DockerEnvironment}")
    except ImportError as e:
        print(f"  ✗ mini-swe-agent import failed: {e}")
        print()
        print("Install mini-swe-agent: pip install mini-swe-agent")
        return 1

    # Step 3: Load test instance
    print("[3/5] Loading test instance...")
    from context_policy.datasets.swebench import load_instances

    if args.instance_id:
        instances = load_instances(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            instance_ids=[args.instance_id],
        )
        if not instances:
            print(f"  ✗ Instance {args.instance_id} not found in dataset")
            return 1
    else:
        instances = load_instances(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            limit=1,
        )

    instance = instances[0]
    print(f"  Instance ID: {instance['instance_id']}")
    print(f"  Repo: {instance['repo']}")
    print(f"  Commit: {instance['base_commit'][:12]}...")
    print(f"  Problem length: {len(instance['problem_statement'])} chars")

    # Step 4: Load context if requested
    context_md = None
    if args.with_context:
        print("[4/5] Loading baseline context...")
        from context_policy.utils.paths import get_context_path

        context_path = get_context_path(instance["repo"], instance["base_commit"])
        if context_path.exists():
            context_md = context_path.read_text(encoding="utf-8")
            print(f"  ✓ Loaded context: {len(context_md)} chars")
        else:
            print(f"  ⚠ Context not found at {context_path}")
            print("  Run: python scripts/build_context.py first")
    else:
        print("[4/5] Skipping context (use --with_context to include)")

    # Step 5: Run the agent
    print("[5/5] Running mini_swe_agent_swebench...")
    print(f"  Model: {args.model}")
    print(f"  Timeout: {args.timeout_s}s")
    print()

    from context_policy.runner.mini_swe_agent_swebench import (
        generate_patch_with_mini_swebench,
    )

    try:
        patch = generate_patch_with_mini_swebench(
            instance=instance,
            model=args.model,
            context_md=context_md,
            timeout_s=args.timeout_s,
        )
    except Exception as e:
        print(f"  ✗ Agent run failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Report results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    if patch:
        print(f"✓ Patch generated: {len(patch)} chars")
        print()
        print("First 20 lines of patch:")
        print("-" * 40)
        lines = patch.split("\n")[:20]
        for line in lines:
            print(line)
        if len(patch.split("\n")) > 20:
            print("...")
        print("-" * 40)
        return 0
    else:
        print("✗ No patch generated (empty result)")
        print()
        print("Possible causes:")
        print("- Model did not produce a valid diff")
        print("- Agent timed out")
        print("- Docker environment issue")
        return 1


if __name__ == "__main__":
    sys.exit(main())
