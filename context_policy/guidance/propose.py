"""Propose candidate guidance edits for hill-climbing.

Given the current best guidance G*, the proposer asks the LLM to produce
K variant guidance blocks.  Each candidate is validated and truncated to
fit within the character budget before being returned.
"""
from __future__ import annotations

import json
import re

from context_policy.guidance.gating import truncate_to_budget, validate_guidance
from context_policy.guidance.schema import RepoGuidance
from context_policy.llm.openai_compat import chat_completion

_PROPOSE_SYSTEM = """\
You are an expert at tuning guidance blocks for a coding agent that fixes
open‑source issues.  You will be given the CURRENT best guidance for a
repository together with its recent score.  Produce exactly {k} VARIANT
guidance blocks, each a plausible improvement.

Rules:
- Each variant must be ≤ {char_budget} characters.
- Each variant is a list of lines starting with "- ".
- Make diverse edits: add tips, remove unhelpful ones, rephrase, reorder.
- Keep changes incremental — do NOT rewrite from scratch.
- Output valid JSON: a list of {k} objects, each with a "lines" key
  containing a list of strings.
- Output ONLY the JSON array.  No commentary."""

_PROPOSE_USER = """\
Repository: {repo}

Current guidance (version {version}, score {score:.1%}):
---
{current_text}
---

Previous scores: {history}

Produce {k} variant guidance blocks as JSON."""


def propose_candidates(
    guidance: RepoGuidance,
    score: float,
    model: str,
    *,
    k: int = 6,
    history: list[tuple[int, float]] | None = None,
    timeout_s: int = 120,
) -> list[RepoGuidance]:
    """Ask the LLM to propose *k* candidate guidance variants.

    Args:
        guidance: The current best guidance.
        score: Current best resolve rate (0-1).
        model: LLM model name.
        k: Number of candidates to request.
        history: List of (version, score) tuples for context.
        timeout_s: LLM call timeout.

    Returns:
        List of up to *k* validated ``RepoGuidance`` candidates,
        each at ``guidance.version + 1``.
    """
    hist_str = "none"
    if history:
        hist_str = ", ".join(f"v{v}={s:.1%}" for v, s in history)

    messages = [
        {
            "role": "system",
            "content": _PROPOSE_SYSTEM.format(k=k, char_budget=guidance.char_budget),
        },
        {
            "role": "user",
            "content": _PROPOSE_USER.format(
                repo=guidance.repo,
                version=guidance.version,
                score=score,
                current_text=guidance.render(),
                history=hist_str,
                k=k,
            ),
        },
    ]

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        timeout_s=timeout_s,
    )

    candidates = _parse_candidates(raw, guidance, k)
    return candidates


def _parse_candidates(
    raw: str,
    base: RepoGuidance,
    k: int,
) -> list[RepoGuidance]:
    """Parse LLM JSON output into validated RepoGuidance objects."""
    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  [propose] Failed to parse JSON from LLM output ({len(text)} chars)")
                return []
        else:
            print(f"  [propose] No JSON array found in LLM output")
            return []

    if not isinstance(arr, list):
        print(f"  [propose] Expected JSON array, got {type(arr).__name__}")
        return []

    next_version = base.version + 1
    results: list[RepoGuidance] = []

    for i, item in enumerate(arr[:k]):
        if isinstance(item, dict) and "lines" in item:
            lines = [str(l).rstrip() for l in item["lines"] if str(l).strip()]
        elif isinstance(item, list):
            lines = [str(l).rstrip() for l in item if str(l).strip()]
        else:
            print(f"  [propose] Skipping candidate {i}: unexpected format")
            continue

        candidate = base.copy(version=next_version, lines=lines)
        candidate = truncate_to_budget(candidate)

        warnings = validate_guidance(candidate)
        if warnings:
            print(f"  [propose] Candidate {i} warnings: {warnings}")
        # Include even with warnings — the tuner scores everything
        results.append(candidate)

    return results
