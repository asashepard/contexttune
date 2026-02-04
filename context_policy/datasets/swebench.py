"""SWE-bench dataset loading utilities."""
from __future__ import annotations

from pathlib import Path


def read_instance_ids(path: str | Path) -> list[str]:
    """Read instance IDs from a file, one per line.

    Ignores blank lines and lines starting with '#'.

    Args:
        path: Path to the instance IDs file.

    Returns:
        List of instance ID strings.
    """
    path = Path(path)
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def load_instances(
    dataset_name: str,
    split: str,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load SWE-bench instances from HuggingFace datasets.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "princeton-nlp/SWE-bench_Verified").
        split: Dataset split (e.g., "test").
        instance_ids: Optional list of instance IDs to filter. If None, load all.
        limit: Optional maximum number of instances to return.

    Returns:
        List of instance dicts with keys:
        - instance_id
        - repo
        - base_commit
        - problem_statement
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    # Filter by instance_ids if provided
    if instance_ids is not None:
        id_set = set(instance_ids)
        ds = ds.filter(lambda x: x["instance_id"] in id_set)

    # Apply limit
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))

    # Extract needed fields
    # Additional fields for docker-native runner: version, environment_setup_commit
    instances = []
    for row in ds:
        inst = {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        }
        # Optional fields for docker-native runner
        if "version" in row:
            inst["version"] = row["version"]
        if "environment_setup_commit" in row:
            inst["environment_setup_commit"] = row["environment_setup_commit"]
        instances.append(inst)

    return instances
