#!/usr/bin/env python3
"""Run adaptive context tuning loop with SWE-Smith task integration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.loop.orchestrator import run_adaptive_loop
from context_policy.utils.run_id import make_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive context tuning loop.")
    parser.add_argument("--model", required=True, help="Model for inference and policy updates.")
    parser.add_argument(
        "--runner",
        default="mini_swe_agent_swebench",
        choices=["single_shot", "mini_swe_agent", "mini_swe_agent_swebench"],
        help="Runner backend.",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Number of adaptive rounds.")
    parser.add_argument(
        "--conditions",
        default="baseline,tuned",
        help="Comma-separated condition names (supports baseline,tuned).",
    )
    parser.add_argument("--tasks_file", default=None, help="Normalized task JSON/JSONL file.")
    parser.add_argument("--swesmith_source", default=None, help="SWE-Smith source JSON/JSONL file to import each round.")
    parser.add_argument(
        "--swesmith_generate_cmd",
        default=None,
        help="Command template to generate SWE-Smith tasks. Supports {round} and {out} placeholders.",
    )
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--timeout_s", type=int, default=1200)
    parser.add_argument("--step_limit", type=int, default=60)
    parser.add_argument("--max_workers_eval", type=int, default=1)
    parser.add_argument("--policy_file", default=None, help="Initial policy JSON file.")
    parser.add_argument("--loop_id", default=None, help="Loop run ID.")
    parser.add_argument("--force", action="store_true", help="Force rerun rounds even if loop_state exists.")
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM/model calls; emit placeholder context and empty patches.")

    args = parser.parse_args()

    loop_id = args.loop_id or make_run_id("adaptive")
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    for condition in conditions:
        if condition not in ("baseline", "tuned"):
            parser.error(f"Unsupported condition: {condition}. Use baseline,tuned")

    state_path = run_adaptive_loop(
        loop_id=loop_id,
        model=args.model,
        runner=args.runner,
        rounds=args.rounds,
        conditions=conditions,
        tasks_file=args.tasks_file,
        swesmith_source=args.swesmith_source,
        swesmith_generate_cmd=args.swesmith_generate_cmd,
        timeout_s=args.timeout_s,
        step_limit=args.step_limit,
        max_workers_eval=args.max_workers_eval,
        dataset_name=args.dataset_name,
        split=args.split,
        policy_file=args.policy_file,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"Adaptive loop complete. State: {state_path}")


if __name__ == "__main__":
    main()
