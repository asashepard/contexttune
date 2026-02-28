"""Initialize guidance G₀ for a repository using an LLM and repo info.

The initializer asks the model to produce a concise, actionable guidance
block grounded in the repository's structure and conventions.
"""
from __future__ import annotations

from pathlib import Path

from context_policy.guidance.gating import truncate_to_budget, validate_guidance
from context_policy.guidance.repo_info import build_repo_info_block
from context_policy.guidance.schema import DEFAULT_CHAR_BUDGET, RepoGuidance
from context_policy.llm.openai_compat import chat_completion

_INIT_SYSTEM = """\
You are an expert software‑engineering assistant.
Your job is to produce a concise GUIDANCE BLOCK that will be prepended to
every issue a coding agent sees when working on a specific open‑source
repository.  The guidance should help the agent produce correct patches
more often.

Rules for the guidance block:
- Maximum {char_budget} characters (hard limit).
- Focus on ACTIONABLE tips: where key modules live, naming conventions,
  test patterns, common pitfalls, import style.
- Do NOT repeat information already visible in the directory tree
  (the agent always sees the tree separately).
- Do NOT include generic Python advice.  Be repo‑specific.
- Write in terse bullet‑point style.  No headings, no markdown fences.
- Every line should start with "- ".
- Output ONLY the guidance lines.  No preamble, no closing remarks."""

_INIT_USER = """\
Repository: {repo}
Commit: {commit}

{repo_info}

Write the guidance block now (max {char_budget} chars)."""


def initialize_guidance(
    repo: str,
    commit: str,
    repo_dir: Path,
    model: str,
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    timeout_s: int = 120,
) -> RepoGuidance:
    """Create the initial guidance G₀ for a repository.

    Args:
        repo: Repository slug (e.g. "django/django").
        commit: Commit SHA the worktree is checked out to.
        repo_dir: Path to the checked-out worktree.
        model: LLM model name for the OpenAI-compat endpoint.
        char_budget: Maximum characters for the guidance text.
        timeout_s: LLM call timeout.

    Returns:
        A validated ``RepoGuidance`` at version 0.
    """
    repo_info = build_repo_info_block(repo_dir)

    messages = [
        {
            "role": "system",
            "content": _INIT_SYSTEM.format(char_budget=char_budget),
        },
        {
            "role": "user",
            "content": _INIT_USER.format(
                repo=repo,
                commit=commit,
                repo_info=repo_info,
                char_budget=char_budget,
            ),
        },
    ]

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=2048,
        timeout_s=timeout_s,
    )

    # Parse into lines, stripping blanks
    lines = [l.rstrip() for l in raw.strip().splitlines() if l.strip()]

    guidance = RepoGuidance(
        repo=repo,
        commit=commit,
        lines=lines,
        version=0,
        char_budget=char_budget,
    )

    # Enforce budget
    guidance = truncate_to_budget(guidance)

    warnings = validate_guidance(guidance)
    if warnings:
        print(f"  [init] Guidance warnings for {repo}: {warnings}")

    return guidance
