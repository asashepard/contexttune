#!/usr/bin/env python3
"""Run the per-repo guidance tuning experiment.

Usage (dev smoke test):
    python scripts/run_experiment.py \\
        --model openai/my-model \\
        --repo-config repos.json \\
        --iterations 3 --candidates 4 --tasks-per-score 10 \\
        --dry-run

Usage (full run):
    python scripts/run_experiment.py \\
        --model openai/my-model \\
        --repo-config repos.json \\
        --iterations 10 --candidates 6 --tasks-per-score 20

The repo-config JSON file has the structure:
[
  {"repo": "django/django", "commit": "<sha>", "tasks_file": "artifacts/tasks/django__django/train.jsonl"},
  ...
]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.loop.orchestrator import ExperimentConfig, run_experiment
from context_policy.guidance.tuner import MAX_TUNING_ITERATIONS
from context_policy.utils.run_id import make_run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-repo guidance tuning experiment."
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. openai/my-model).")
    parser.add_argument(
        "--repo-config", required=True,
        help="JSON file listing repos with commit + tasks_file.",
    )
    parser.add_argument("--experiment-id", default=None, help="Experiment run ID.")

    # Tuning hyperparams
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help=f"Hill-climbing iterations T (0..{MAX_TUNING_ITERATIONS}).",
    )
    parser.add_argument("--candidates", type=int, default=6, help="Candidates per iteration K.")
    parser.add_argument("--tasks-per-score", type=int, default=20, help="Tasks per scoring run N.")
    parser.add_argument("--char-budget", type=int, default=3200, help="Guidance char budget.")

    # Runner settings
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-task agent timeout.")
    parser.add_argument("--step-limit", type=int, default=30, help="Agent step limit.")

    # Eval settings
    parser.add_argument("--eval-dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-instance-ids", default=None, help="File with instance IDs to evaluate.")
    parser.add_argument("--max-workers-eval", type=int, default=4)

    parser.add_argument("--dry-run", action="store_true", help="Skip LLM/agent calls.")

    args = parser.parse_args()

    if args.iterations < 0 or args.iterations > MAX_TUNING_ITERATIONS:
        parser.error(f"--iterations must be between 0 and {MAX_TUNING_ITERATIONS}.")

    # Load repo config
    repo_config_path = Path(args.repo_config)
    if not repo_config_path.exists():
        parser.error(f"Repo config file not found: {repo_config_path}")
    repos = json.loads(repo_config_path.read_text(encoding="utf-8"))

    experiment_id = args.experiment_id or make_run_id("guidance_tune")

    config = ExperimentConfig(
        experiment_id=experiment_id,
        model=args.model,
        repos=repos,
        iterations=args.iterations,
        candidates_per_iter=args.candidates,
        tasks_per_score=args.tasks_per_score,
        char_budget=args.char_budget,
        timeout_s=args.timeout_s,
        step_limit=args.step_limit,
        eval_dataset=args.eval_dataset,
        eval_split=args.eval_split,
        eval_instance_ids_file=args.eval_instance_ids,
        max_workers_eval=args.max_workers_eval,
    )

    print(f"Experiment: {experiment_id}")
    print(f"Model: {args.model}")
    print(f"Repos: {len(repos)}")
    print(f"Tuning: T={args.iterations}, K={args.candidates}, N={args.tasks_per_score}")
    print(f"Dry run: {args.dry_run}")
    print()

    result_dir = run_experiment(config, dry_run=args.dry_run)
    print(f"\nExperiment complete. Results: {result_dir}")


if __name__ == "__main__":
    main()
