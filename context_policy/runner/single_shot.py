"""Single-shot patch generation runner."""
from __future__ import annotations

import re
from pathlib import Path

from context_policy.git.checkout import checkout_repo
from context_policy.llm.openai_compat import chat_completion
from context_policy.prompting.prompt_builder import build_messages


# Maximum allowed patch size (chars) - safety limit
MAX_PATCH_SIZE = 200_000


def _extract_diff(text: str) -> str:
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


def generate_patch(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout_s: int = 120,
) -> str:
    """Generate a patch for a SWE-bench instance using a single model call.

    Args:
        instance: Instance dict with instance_id, repo, base_commit, problem_statement.
        model: Model name to use for inference.
        context_md: Optional additional context to include in prompt.
        temperature: Sampling temperature (default 0.0).
        top_p: Top-p parameter (default 1.0).
        max_tokens: Maximum tokens to generate (default 1024).
        timeout_s: Request timeout in seconds (default 120).

    Returns:
        Extracted unified diff patch string, or empty string if extraction failed.
    """
    repo = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]

    # Checkout repo at commit
    repo_dir = checkout_repo(repo, commit)

    # Build prompt messages
    messages = build_messages(
        problem_statement=problem_statement,
        repo=repo,
        commit=commit,
        repo_dir=repo_dir,
        context_md=context_md,
    )

    # Call model
    response = chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    # Extract diff
    diff = _extract_diff(response)

    # Safety: reject oversized patches
    if len(diff) > MAX_PATCH_SIZE:
        return ""

    return diff
