#!/usr/bin/env python3
"""Tune guidance for a single repository.

Usage:
    python scripts/tune_single_repo.py \\
        --repo django/django \\
        --commit abc123 \\
        --tasks-file artifacts/tasks/django__django/train.jsonl \\
        --model openai/my-model \\
        --output-dir results/exp1/guidance/django__django

This is the building block for slurm array jobs — each array task
runs this script for one repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.guidance.tuner import MAX_TUNING_ITERATIONS, TuningConfig, run_tuning_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune guidance for a single repo.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help=f"Hill-climbing iterations (0..{MAX_TUNING_ITERATIONS}).",
    )
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--tasks-per-score", type=int, default=20)
    parser.add_argument("--char-budget", type=int, default=3200)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--step-limit", type=int, default=30)

    args = parser.parse_args()

    if args.iterations < 0 or args.iterations > MAX_TUNING_ITERATIONS:
        parser.error(f"--iterations must be between 0 and {MAX_TUNING_ITERATIONS}.")

    config = TuningConfig(
        repo=args.repo,
        commit=args.commit,
        tasks_file=args.tasks_file,
        model=args.model,
        iterations=args.iterations,
        candidates_per_iter=args.candidates,
        tasks_per_score=args.tasks_per_score,
        char_budget=args.char_budget,
        timeout_s=args.timeout_s,
        step_limit=args.step_limit,
        output_dir=args.output_dir,
    )

    best = run_tuning_loop(config)
    print(f"\nBest guidance: v{best.version} ({best.char_count()} chars, {len(best.lines)} lines)")


if __name__ == "__main__":
    main()
