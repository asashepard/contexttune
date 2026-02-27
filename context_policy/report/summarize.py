"""Load and summarize SWE-bench evaluation results."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir: Path) -> tuple[int, int]:
    """Load results from a SWE-bench evaluation directory.

    Tries results.json first, then falls back to instance_results.jsonl.

    Returns:
        (resolved, total) counts.
    """
    # Try results.json first (summary file from harness)
    results_json = results_dir / "results.json"
    if results_json.exists():
        try:
            data = json.loads(results_json.read_text(encoding="utf-8"))
            # Handle different result formats
            if isinstance(data, dict):
                # Format: {"resolved": [...], "applied": [...], ...}
                resolved_list = data.get("resolved", [])
                # Total = resolved + unresolved (or count from applied)
                applied_list = data.get("applied", [])
                if applied_list:
                    total = len(applied_list)
                else:
                    # Fallback: count all instances mentioned
                    all_ids = set()
                    for key in ["resolved", "applied", "failed", "error"]:
                        all_ids.update(data.get(key, []))
                    total = len(all_ids) if all_ids else len(resolved_list)
                return len(resolved_list), total
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to jsonl

    # Fallback: parse instance_results.jsonl
    instance_results = results_dir / "instance_results.jsonl"
    if instance_results.exists():
        resolved = 0
        total = 0
        for line in instance_results.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                total += 1
                # Check for resolved/passed field (various formats)
                if record.get("resolved") or record.get("passed"):
                    resolved += 1
            except json.JSONDecodeError:
                continue
        return resolved, total

    # No results found
    return 0, 0


def compute_rate(resolved: int, total: int) -> float:
    """Compute success rate as a float between 0.0 and 1.0."""
    return resolved / total if total > 0 else 0.0


def load_instance_records(results_dir: Path) -> list[dict]:
    """Load per-instance harness records when available."""
    instance_results = results_dir / "instance_results.jsonl"
    if not instance_results.exists():
        return []

    records: list[dict] = []
    for raw_line in instance_results.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def classify_failure(record: dict) -> str:
    """Classify unresolved instance outcome into coarse failure taxonomy."""
    if record.get("resolved") or record.get("passed"):
        return "resolved"

    text_parts = []
    for key in ["error", "error_message", "failure_reason", "report", "status"]:
        value = record.get(key)
        if value:
            text_parts.append(str(value).lower())
    text = " ".join(text_parts)

    if "timeout" in text:
        return "timeout"
    if "apply" in text or "patch" in text:
        return "patch_apply_failure"
    if "importerror" in text or "module" in text or "environment" in text:
        return "environment_failure"
    if "test" in text or "assert" in text or "fail" in text:
        return "test_failure"
    if "error" in text or "exception" in text:
        return "runtime_error"
    return "unresolved_unknown"


def summarize_failure_taxonomy(records: list[dict]) -> dict[str, int]:
    """Summarize failure taxonomy counts from instance records."""
    counts: dict[str, int] = {}
    for record in records:
        category = classify_failure(record)
        counts[category] = counts.get(category, 0) + 1
    return counts
