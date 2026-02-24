#!/usr/bin/env python3
"""Run Verified Mini baseline experiment: compare no_context vs baseline_context.

Usage:
    python scripts/run_verified_mini_baseline.py --model local/placeholder --dry_run
    python scripts/run_verified_mini_baseline.py --model openai/gpt-4 --max_workers_eval 4

This script orchestrates:
1. Build signals for all instances (once)
2. Build context from signals (once)
3. Run inference for each ablation condition
4. Run SWE-bench harness evaluation for each condition
5. Summarize and compare results

Requires Docker to be running for harness evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from context_policy.report.summarize import compute_rate, load_results
from context_policy.utils.paths import PREDS_DIR, RESULTS_DIR
from context_policy.utils.run_id import make_run_id
from context_policy.utils.subproc import run as subproc_run

DEFAULT_CONDITIONS = ["no_context", "baseline_context"]

# Runner-specific default timeouts (seconds)
_RUNNER_DEFAULT_TIMEOUT: dict[str, int] = {
    "single_shot": 120,
    "mini_swe_agent": 300,
    "mini_swe_agent_swebench": 600,
}


def load_instance_ids(path: Path) -> list[str]:
    """Load instance IDs from file, ignoring comments and blank lines."""
    if not path.exists():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def run_step(
    name: str,
    cmd: list[str],
    logs_dir: Path,
    *,
    check: bool = True,
) -> int:
    """Run a pipeline step with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    stdout_log = logs_dir / f"{name}.stdout.log"
    stderr_log = logs_dir / f"{name}.stderr.log"

    exit_code = subproc_run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout_path=stdout_log,
        stderr_path=stderr_log,
    )

    if exit_code == 0:
        print(f"  -> OK (exit 0)")
    else:
        print(f"  -> FAILED (exit {exit_code})")
        print(f"     See: {stderr_log}")
        if check:
            sys.exit(exit_code)

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Verified Mini baseline experiment comparing ablations."
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--instance_ids_file",
        default="scripts/verified_mini_ids.txt",
        help="Path to instance IDs file (one per line, # comments allowed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Fallback limit if instance_ids_file is empty or missing.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for inference (written to predictions JSONL).",
    )
    parser.add_argument(
        "--max_workers_eval",
        type=int,
        default=1,
        help="Max workers for harness evaluation.",
    )
    parser.add_argument(
        "--tag",
        default="mini",
        help="Tag for run ID (default: mini).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all artifacts.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Pass --dry_run to inference (skip model calls).",
    )
    parser.add_argument(
        "--runner",
        choices=["single_shot", "mini_swe_agent", "mini_swe_agent_swebench"],
        default="single_shot",
        help="Runner backend forwarded to run_inference.py (default: single_shot).",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated conditions to run (default: no_context,baseline_context). "
            "Example: --conditions baseline_context"
        ),
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=None,
        help=(
            "Per-instance timeout in seconds forwarded to run_inference.py. "
            "If omitted, a runner-specific default is used (120/300/600)."
        ),
    )
    parser.add_argument(
        "--step_limit",
        type=int,
        default=30,
        help="Agent step limit for mini_swe_agent_swebench (default: 30; 0=unlimited).",
    )

    args = parser.parse_args()

    # Resolve conditions
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        for c in conditions:
            if c not in ("no_context", "baseline_context"):
                parser.error(f"Unknown condition: {c}")
    else:
        conditions = list(DEFAULT_CONDITIONS)

    # Resolve timeout
    timeout_s = args.timeout_s or _RUNNER_DEFAULT_TIMEOUT.get(args.runner, 120)

    # Generate group ID
    group_id = make_run_id(f"verified_{args.tag}")
    print(f"Group ID: {group_id}")
    print(f"Model: {args.model}")
    print(f"Runner: {args.runner}")
    print(f"Timeout: {timeout_s}s")
    print(f"Conditions: {conditions}")

    # Setup logs directory
    logs_dir = RESULTS_DIR / group_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load instance IDs
    ids_file = Path(args.instance_ids_file)
    instance_ids = load_instance_ids(ids_file)
    use_ids_file = bool(instance_ids)

    if instance_ids:
        print(f"Instance IDs: {len(instance_ids)} from {ids_file}")
    else:
        print(f"Instance IDs: using --limit {args.limit} (no IDs file)")

    # Build common args for instance selection
    if use_ids_file:
        instance_args = ["--instance_ids_file", str(ids_file)]
    else:
        instance_args = ["--limit", str(args.limit)]

    # =========================================================================
    # Step 1: Build signals (prerequisite for baseline_context)
    # =========================================================================
    force_arg = ["--force"] if args.force else []
    run_step(
        "build_signals",
        [
            sys.executable, "scripts/build_signals.py",
            "--dataset_name", args.dataset_name,
            "--split", args.split,
            *instance_args,
            *force_arg,
        ],
        logs_dir,
    )

    # =========================================================================
    # Step 2: Build context from signals
    # =========================================================================
    run_step(
        "build_context",
        [
            sys.executable, "scripts/build_context.py",
            "--dataset_name", args.dataset_name,
            "--split", args.split,
            *instance_args,
            *force_arg,
        ],
        logs_dir,
    )

    # =========================================================================
    # Step 3 & 4: For each condition, run inference + evaluation
    # =========================================================================
    condition_results: dict[str, dict] = {}

    for condition in conditions:
        run_id = f"{group_id}__{condition}"
        preds_dir = PREDS_DIR / group_id / condition
        preds_path = preds_dir / "preds.jsonl"
        results_dir = RESULTS_DIR / run_id

        print(f"\n{'#'*60}")
        print(f"# CONDITION: {condition}")
        print(f"# Run ID: {run_id}")
        print(f"# Preds: {preds_path}")
        print(f"# Results: {results_dir}")
        print(f"{'#'*60}")

        # ----- Inference -----
        if preds_path.exists() and not args.force:
            print(f"  Predictions exist, skipping inference.")
        else:
            dry_run_arg = ["--dry_run"] if args.dry_run else []
            run_step(
                f"inference_{condition}",
                [
                    sys.executable, "scripts/run_inference.py",
                    "--dataset_name", args.dataset_name,
                    "--split", args.split,
                    *instance_args,
                    "--model", args.model,
                    "--ablation", condition,
                    "--runner", args.runner,
                    "--timeout_s", str(timeout_s),
                    "--step_limit", str(args.step_limit),
                    "--run_id", run_id,
                    "--out", str(preds_path),
                    *dry_run_arg,
                ],
                logs_dir,
            )

        # ----- Evaluation -----
        results_json = results_dir / "results.json"
        if results_json.exists() and not args.force:
            print(f"  Results exist, skipping evaluation.")
        else:
            # Run SWE-bench harness via bash script
            run_step(
                f"eval_{condition}",
                [
                    "bash", "scripts/run_swebench_eval.sh",
                    args.dataset_name,
                    str(preds_path),
                    run_id,
                    str(args.max_workers_eval),
                ],
                logs_dir,
                check=False,  # Don't abort on eval failure (Docker might not be running)
            )

        # ----- Load results -----
        resolved, total = load_results(results_dir)
        rate = compute_rate(resolved, total)
        condition_results[condition] = {
            "resolved": resolved,
            "total": total,
            "rate": rate,
            "run_id": run_id,
            "preds_path": str(preds_path),
            "results_dir": str(results_dir),
        }

    # =========================================================================
    # Step 5: Summarize
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Determine total from any condition that has data, or fall back to ID count
    total = 0
    for cr in condition_results.values():
        if cr["total"]:
            total = cr["total"]
            break
    if not total:
        total = len(instance_ids) or args.limit

    for cond, cr in condition_results.items():
        print(f"  {cond:20s} {cr['resolved']:3}/{total} ({cr['rate']*100:5.1f}%)")

    # Compute delta only when both conditions are present
    delta_info: dict | None = None
    if "no_context" in condition_results and "baseline_context" in condition_results:
        no_ctx = condition_results["no_context"]
        base_ctx = condition_results["baseline_context"]
        delta = base_ctx["resolved"] - no_ctx["resolved"]
        delta_rate = delta / total if total else 0.0
        delta_info = {"resolved": delta, "rate": delta_rate}
        print(f"  {'delta':20s} {delta:+3}/{total} ({delta_rate*100:+5.1f}%)")

    # Write summary JSON
    summary: dict = {
        "group_id": group_id,
        "dataset": args.dataset_name,
        "split": args.split,
        "model": args.model,
        "runner": args.runner,
        "timeout_s": timeout_s,
        "instance_count": total,
        "conditions": condition_results,
    }
    if delta_info is not None:
        summary["delta"] = delta_info

    summary_path = RESULTS_DIR / group_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
