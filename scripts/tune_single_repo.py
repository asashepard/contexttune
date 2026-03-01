#!/usr/bin/env python3
"""Tune guidance for a single repository using the oracle loop.

Usage:
    python scripts/tune_single_repo.py \\
        --repo django/django \\
        --commit abc123 \\
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

from context_policy.oracle.loop import run_oracle_loop
from context_policy.oracle.schema import OracleConfig


MAX_ORACLE_ITERATIONS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune guidance for a single repo.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help=f"Oracle iterations (0..{MAX_ORACLE_ITERATIONS}).  0 = static KB only.",
    )
    parser.add_argument("--timeout-s", type=int, default=120)

    args = parser.parse_args()

    if args.iterations < 0 or args.iterations > MAX_ORACLE_ITERATIONS:
        parser.error(f"--iterations must be between 0 and {MAX_ORACLE_ITERATIONS}.")

    config = OracleConfig(
        repo=args.repo,
        commit=args.commit,
        model=args.model,
        iterations=args.iterations,
        timeout_s=args.timeout_s,
        output_dir=args.output_dir,
    )

    kb, best = run_oracle_loop(config)
    print(f"\nBest guidance: v{best.version} ({best.char_count()} chars, {len(best.lines)} lines)")


if __name__ == "__main__":
    main()
