"""JSONL read/write utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file.

    Each record is written as a single-line JSON object with sorted keys,
    UTF-8 encoding, and newline termination.

    Args:
        path: Output file path. Parent directories are created if needed.
        records: List of dictionaries to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, sort_keys=True, ensure_ascii=False)
            f.write(line + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSONL file.

    Args:
        path: Input file path.

    Returns:
        List of dictionaries parsed from the file.
    """
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
