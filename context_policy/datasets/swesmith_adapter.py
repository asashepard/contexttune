"""SWE-Smith task adapters and normalization helpers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def normalize_swesmith_record(row: dict) -> dict:
    """Normalize a SWE-Smith-like task record into local task format."""
    instance_id = row.get("instance_id") or row.get("task_id") or row.get("id")
    repo = row.get("repo") or row.get("repository")
    base_commit = row.get("base_commit") or row.get("base_sha") or row.get("commit")
    problem_statement = (
        row.get("problem_statement")
        or row.get("issue_text")
        or row.get("issue")
        or row.get("prompt")
        or ""
    )

    if not instance_id or not repo or not base_commit:
        raise ValueError("SWE-Smith record missing required fields")

    return {
        "instance_id": str(instance_id),
        "repo": str(repo),
        "base_commit": str(base_commit),
        "problem_statement": str(problem_statement),
        "source": "swe_smith",
        "metadata": {
            "original_keys": sorted(row.keys()),
        },
    }


def load_swesmith_records(path: str | Path) -> list[dict]:
    """Load SWE-Smith records from JSON or JSONL file."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"SWE-Smith source file not found: {source_path}")

    if source_path.suffix.lower() == ".jsonl":
        rows = []
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "tasks" in payload:
            return payload["tasks"]
        if "instances" in payload:
            return payload["instances"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported SWE-Smith input payload")


def write_normalized_tasks(path: str | Path, tasks: list[dict]) -> None:
    """Write normalized tasks to JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True, ensure_ascii=False) + "\n")


def import_swesmith_tasks(source_path: str | Path, out_path: str | Path) -> int:
    """Convert source SWE-Smith records into normalized local task JSONL."""
    rows = load_swesmith_records(source_path)
    tasks = [normalize_swesmith_record(row) for row in rows]
    write_normalized_tasks(out_path, tasks)
    return len(tasks)


def generate_swesmith_tasks(command_template: str, out_path: str | Path, *, round_index: int) -> int:
    """Run a user-provided SWE-Smith generation command and import outputs.

    The command can include placeholders:
    - {round}: round index
    - {out}: output path
    """
    out_file = Path(out_path)
    rendered = command_template.format(round=round_index, out=str(out_file))
    result = subprocess.run(rendered, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"SWE-Smith generation command failed: {rendered}")
    rows = load_swesmith_records(out_file)
    tasks = [normalize_swesmith_record(row) for row in rows]
    write_normalized_tasks(out_file, tasks)
    return len(tasks)
