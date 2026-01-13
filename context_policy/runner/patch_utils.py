"""Utility functions for extracting patches from model/agent output."""
from __future__ import annotations

import json
import re


# Maximum allowed patch size (chars) - safety limit
MAX_PATCH_SIZE = 200_000


def extract_diff(text: str) -> str:
    """Extract unified diff from model output.

    Tries in order:
    1. Fenced code block with ```diff ... ```
    2. First line starting with "diff --git" and everything after
    3. First line starting with "--- " and everything after
    4. Empty string if no diff found

    Args:
        text: Raw model output.

    Returns:
        Extracted diff string or empty string.
    """
    # Try fenced diff block
    fence_pattern = r"```(?:diff)?\s*\n(.*?)```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    if matches:
        # Return the first fenced block that looks like a diff
        for match in matches:
            if "---" in match or "diff --git" in match:
                return match.strip()

    # Try to find diff --git line
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            return "\n".join(lines[i:]).strip()

    # Try to find --- line (start of unified diff)
    for i, line in enumerate(lines):
        if line.startswith("--- "):
            return "\n".join(lines[i:]).strip()

    return ""


def extract_patch_from_trajectory(traj_path: str) -> str:
    """Extract patch from a mini-swe-agent trajectory JSON file.

    Tries in order:
    1. Top-level 'patch' or 'model_patch' field
    2. Last action/step with a diff in its output
    3. Empty string if no patch found

    Args:
        traj_path: Path to trajectory JSON file.

    Returns:
        Extracted patch string or empty string.
    """
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    # Try top-level patch fields
    for key in ["patch", "model_patch", "diff"]:
        if key in data and isinstance(data[key], str) and data[key].strip():
            return data[key].strip()

    # Try to find patch in actions/steps/messages
    for key in ["actions", "steps", "messages", "history"]:
        if key in data and isinstance(data[key], list):
            # Scan from end (most recent) to find a diff
            for item in reversed(data[key]):
                if isinstance(item, dict):
                    for field in ["output", "content", "result", "patch"]:
                        if field in item and isinstance(item[field], str):
                            diff = extract_diff(item[field])
                            if diff:
                                return diff
                elif isinstance(item, str):
                    diff = extract_diff(item)
                    if diff:
                        return diff

    return ""
