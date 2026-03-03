"""Utility functions for extracting patches from model/agent output."""
from __future__ import annotations

import json
import re
from collections import Counter


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


def _strip_to_first_fenced_diff_block(text: str) -> str:
    """If fenced blocks exist, keep only the first one containing a diff marker."""
    if "```" not in text:
        return text

    fence_pattern = r"```[^\n]*\n(.*?)```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for block in matches:
        for line in block.splitlines():
            if line.startswith("diff --git ") or line.startswith("--- a/"):
                return block.strip()

    # If fenced blocks exist but none look like a diff, remove fence markers.
    return text.replace("```", "")


def _slice_from_first_diff_start(text: str) -> str:
    """Keep content starting at the first unified-diff start marker."""
    lines = text.splitlines()
    start_index = None
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- a/"):
            start_index = idx
            break
    if start_index is None:
        return ""
    return "\n".join(lines[start_index:]).strip()


def _is_noop_diff(patch: str) -> bool:
    """Return True when a diff has no effective changes.

    A patch is treated as no-op when:
    - it is empty, or
    - it has no +/- hunk lines, or
    - removed and added hunk lines are identical as multisets.
    """
    if not patch or not patch.strip():
        return True

    minus_lines: list[str] = []
    plus_lines: list[str] = []

    for line in patch.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("-"):
            minus_lines.append(line[1:])
        elif line.startswith("+"):
            plus_lines.append(line[1:])

    if not minus_lines and not plus_lines:
        return True

    return Counter(minus_lines) == Counter(plus_lines)


def sanitize_patch_for_preds(patch: str) -> tuple[str, bool]:
    """Sanitize a patch before writing model_patch to preds.jsonl.

    Steps:
    1) If fenced blocks are present, keep the first fenced block with diff markers.
    2) Keep only content from first `diff --git` or `--- a/` line onward.
    3) Treat empty/no-op diffs as empty patch.

    Returns:
        (sanitized_patch, is_noop)
    """
    candidate = _strip_to_first_fenced_diff_block(patch or "")
    sanitized = _slice_from_first_diff_start(candidate)
    is_noop = _is_noop_diff(sanitized)
    if is_noop:
        return "", True
    return sanitized, False
