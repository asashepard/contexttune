"""Failure diagnosis — proposes structured edits to AGENTS.md.

Collects all failing verdicts, sends them with current AGENTS.md to
the LLM, and gets back structured edit proposals (section, action, content).
"""
from __future__ import annotations

import json
import re

from context_policy.llm.openai_compat import chat_completion
from context_policy.oracle.schema import Edit, ProbeResult

_DIAGNOSE_SYSTEM = """\
You are an expert at diagnosing why a coding assistant's AGENTS.md
instructions failed to guide correct behavior.  You will be given:

1. The current AGENTS.md content.
2. A list of probe failures — each describing a task, an expected
   behavior, and the assistant's reasoning for why it was rated FAIL.

Your job is to propose TARGETED edits to AGENTS.md that would fix
the failures without breaking things that already work.

Output a JSON array of edit objects, each with:
- "section": which AGENTS.md section to edit (e.g. "Hub Safety", "Testing", "Conventions", or "new")
- "action": one of "add", "modify", "strengthen", "remove"
- "content": the specific text to add, modify, or strengthen

Rules:
- Be specific and actionable.  Don't add vague platitudes.
- Prefer "strengthen" over "add" when a rule exists but is too weak.
- Prefer "modify" over "remove" + "add" when refining existing text.
- Keep the total AGENTS.md under 3,200 characters.
- Output ONLY the JSON array."""

_DIAGNOSE_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

PROBE FAILURES:
{failures}

Propose edits to fix these failures."""


def diagnose_failures(
    agents_md: str,
    failures: list[ProbeResult],
    model: str,
    *,
    timeout_s: int = 120,
) -> list[Edit]:
    """Diagnose probe failures and propose AGENTS.md edits.

    Args:
        agents_md: Current AGENTS.md content.
        failures: List of ProbeResults that have at least one FAIL verdict.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        List of structured ``Edit`` proposals.
    """
    # Build failure summary
    failure_lines: list[str] = []
    for pr in failures:
        for v in pr.verdicts:
            if not v.passed:
                failure_lines.append(
                    f"- [{pr.category}] Probe {pr.probe_id}: "
                    f"Expected: {v.behavior} — "
                    f"Reasoning: {v.reasoning}"
                )

    if not failure_lines:
        return []

    failures_text = "\n".join(failure_lines)

    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {
            "role": "user",
            "content": _DIAGNOSE_USER.format(
                agents_md=agents_md,
                failures=failures_text,
            ),
        },
    ]

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        timeout_s=timeout_s,
    )

    return _parse_edits(raw)


def _parse_edits(raw: str) -> list[Edit]:
    """Parse LLM JSON output into Edit objects."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
            except json.JSONDecodeError:
                print("  [diagnose] Failed to parse edits from LLM output")
                return []
        else:
            print("  [diagnose] No JSON array found in LLM output")
            return []

    if not isinstance(arr, list):
        return []

    valid_actions = {"add", "modify", "strengthen", "remove"}
    edits: list[Edit] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "")).strip()
        action = str(item.get("action", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not section or not content:
            continue
        if action not in valid_actions:
            action = "add"
        edits.append(Edit(section=section, action=action, content=content))

    return edits
