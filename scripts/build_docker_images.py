#!/usr/bin/env python3
"""Build SWE-bench Docker images for a set of instances.

This pre-builds the Docker images that mini-swe-agent needs, so actual
inference runs don't spend time on image construction.

Usage:
    python scripts/build_docker_images.py --instance_ids_file scripts/smoke_3_ids.txt
    python scripts/build_docker_images.py --instance_ids_file scripts/verified_mini_ids.txt
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.datasets.swebench import load_instances, read_instance_ids


def _images_exist(instance_ids: list[str]) -> dict[str, str | None]:
    """Check which SWE-bench Docker images already exist locally.

    Returns a dict mapping instance_id -> image name (or None if missing).
    """
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        all_images = result.stdout.strip().splitlines() if result.returncode == 0 else []
    except Exception:
        all_images = []

    mapping: dict[str, str | None] = {}
    for iid in instance_ids:
        short_id = iid.split("__")[-1]  # e.g. "django-10097"
        found = None
        for img in all_images:
            if short_id in img and "sweb.eval" in img:
                found = img
                break
        mapping[iid] = found
    return mapping


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build SWE-bench Docker images.")
    parser.add_argument(
        "--instance_ids_file",
        required=True,
        help="Path to file with instance IDs (one per line).",
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Parallel workers for image building (default: 1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild images even if they already exist.",
    )
    args = parser.parse_args()

    # Load instance IDs
    instance_ids = read_instance_ids(args.instance_ids_file)
    if not instance_ids:
        print("No instance IDs found.")
        return 1
    print(f"Loaded {len(instance_ids)} instance IDs from {args.instance_ids_file}")

    # Check which images already exist
    if not args.force:
        existing = _images_exist(instance_ids)
        missing = [iid for iid, img in existing.items() if img is None]
        present = [iid for iid, img in existing.items() if img is not None]
        if present:
            print(f"Already built: {len(present)} images")
            for iid in present:
                print(f"  ✓ {iid} -> {existing[iid]}")
        if not missing:
            print("All images exist. Use --force to rebuild.")
            return 0
        print(f"Need to build: {len(missing)} images")
        build_ids = missing
    else:
        build_ids = instance_ids

    # Load full instance data (needed by swebench harness)
    print("Loading instance data from dataset...")
    instances = load_instances(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=build_ids,
    )

    # Write dummy predictions JSONL (non-empty patch to force image build)
    dummy_patch = "--- a/dummy.txt\n+++ b/dummy.txt\n@@ -0,0 +1 @@\n+dummy\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for inst in instances:
            record = {
                "instance_id": inst["instance_id"],
                "model_name_or_path": "dummy",
                "model_patch": dummy_patch,
            }
            f.write(json.dumps(record) + "\n")
        preds_path = f.name

    print(f"Building {len(instances)} Docker images...")
    print(f"  Dummy predictions: {preds_path}")
    print(f"  Workers: {args.max_workers}")
    print()

    # Run swebench harness to build images
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", args.dataset_name,
        "--predictions_path", preds_path,
        "--run_id", "image_build",
        "--max_workers", str(args.max_workers),
        "--cache_level", "instance",
    ]
    print(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # Clean up temp file
    try:
        Path(preds_path).unlink()
    except OSError:
        pass

    # Verify images were built
    print()
    print("Verifying images...")
    final = _images_exist(build_ids)
    ok = 0
    fail = 0
    for iid, img in final.items():
        if img:
            print(f"  ✓ {iid} -> {img}")
            ok += 1
        else:
            print(f"  ✗ {iid} -> NOT FOUND")
            fail += 1

    print(f"\nResult: {ok} built, {fail} missing")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
