#!/usr/bin/env python3
"""Split instance IDs file into deterministic shards for Slurm job fan-out."""
from __future__ import annotations

import argparse
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def write_shard(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(ids) + ("\n" if ids else "")
    path.write_text(body, encoding="utf-8")


def split_round_robin(ids: list[str], shard_count: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, instance_id in enumerate(ids):
        shards[index % shard_count].append(instance_id)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split instance IDs into N round-robin shard files."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Source IDs file (one instance_id per line).",
    )
    parser.add_argument(
        "--shards",
        type=int,
        required=True,
        help="Number of shard files to generate.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for shard files.",
    )
    parser.add_argument(
        "--prefix",
        default="verified_shard",
        help="Filename prefix for shard files (default: verified_shard).",
    )

    args = parser.parse_args()

    if args.shards <= 0:
        raise ValueError("--shards must be > 0")

    input_path = Path(args.input_file)
    out_dir = Path(args.out_dir)

    ids = read_ids(input_path)
    if not ids:
        raise ValueError(f"No instance IDs found in {input_path}")

    shards = split_round_robin(ids, args.shards)

    print(f"Loaded {len(ids)} IDs from {input_path}")
    print(f"Writing {args.shards} shards to {out_dir}")

    for i, shard_ids in enumerate(shards):
        shard_path = out_dir / f"{args.prefix}_{i:03d}.txt"
        write_shard(shard_path, shard_ids)
        print(f"  {shard_path}: {len(shard_ids)} IDs")


if __name__ == "__main__":
    main()
