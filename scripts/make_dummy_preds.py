#!/usr/bin/env python3
"""Generate a dummy predictions JSONL file for SWE-bench harness testing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.utils.jsonl import write_jsonl
from context_policy.utils.run_id import make_run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dummy predictions JSONL for SWE-bench harness."
    )
    parser.add_argument(
        "--instance_id",
        required=True,
        help="SWE-bench instance ID (e.g., django__django-16379).",
    )
    parser.add_argument(
        "--model_name",
        default="dummy",
        help="Model name to record in predictions (default: dummy).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Default: artifacts/preds/<auto_run_id>/preds.jsonl",
    )
    parser.add_argument(
        "--patch_path",
        default=None,
        help="Path to a patch file to use as model_patch. If not provided, empty string.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Explicit run ID. If not provided, auto-generated.",
    )
    args = parser.parse_args()

    # Determine run_id
    run_id = args.run_id or make_run_id(f"dummy_{args.instance_id}")

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("artifacts") / "preds" / run_id / "preds.jsonl"

    # Read patch if provided
    model_patch = ""
    if args.patch_path:
        patch_file = Path(args.patch_path)
        if not patch_file.exists():
            print(f"Error: patch file not found: {patch_file}", file=sys.stderr)
            sys.exit(1)
        model_patch = patch_file.read_text(encoding="utf-8")

    # Build prediction record
    record = {
        "instance_id": args.instance_id,
        "model_name_or_path": args.model_name,
        "model_patch": model_patch,
    }

    # Write JSONL
    write_jsonl(out_path, [record])
    print(f"Wrote predictions to: {out_path}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
