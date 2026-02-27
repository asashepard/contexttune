#!/usr/bin/env python3
"""Import SWE-Smith tasks into normalized local task JSONL."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_policy.datasets.swesmith_adapter import import_swesmith_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SWE-Smith tasks into normalized task JSONL.")
    parser.add_argument("--source", required=True, help="Path to SWE-Smith source JSON/JSONL")
    parser.add_argument("--out", required=True, help="Output path for normalized JSONL")
    args = parser.parse_args()

    count = import_swesmith_tasks(args.source, args.out)
    print(f"Imported {count} tasks -> {args.out}")


if __name__ == "__main__":
    main()
