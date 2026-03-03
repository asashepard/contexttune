"""Diagnostics aggregation — proposes structured edits to AGENTS.md."""
from __future__ import annotations

import json
import re

from context_policy.llm.openai_compat import chat_completion
from context_policy.oracle.schema import Edit, ProbeResult

_DIAGNOSE_SYSTEM = """\
You are an expert AGENTS.md editor. You will be given the current AGENTS.md
and diagnostic probe outcomes. Your goal is to improve future assistant
behavior by proposing targeted edits.

Output a JSON array of edit objects, each with:
- "section": which AGENTS.md section to edit (e.g. "Hub Safety", "Testing", "Conventions", or "new")
- "action": one of "add", "modify", "strengthen", "remove"
- "content": the specific text to add, modify, or strengthen

Rules:
- Be specific and actionable.  Don't add vague platitudes.
- Prefer "strengthen" or "modify" over adding duplicate rules.
- Keep the total AGENTS.md under 3,200 characters.
- Output ONLY the JSON array."""

_DIAGNOSE_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

PROBE DIAGNOSTICS:
{diagnostics}

Propose edits to improve AGENTS.md for future iterations."""


def diagnose_failures(
    agents_md: str,
    results: list[ProbeResult],
    model: str,
    *,
    timeout_s: int = 120,
) -> list[Edit]:
    """Aggregate probe diagnostics and propose AGENTS.md edits.

    Args:
        agents_md: Current AGENTS.md content.
        results: ProbeResults from current iteration.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        List of structured ``Edit`` proposals.
    """
    diagnostic_lines: list[str] = []
    for pr in results:
        diagnostic_lines.append(f"- Probe {pr.probe_id}: {pr.task}")
        for review in pr.behavior_reviews:
            diagnostic_lines.append(
                f"  * Behavior: {review.behavior} | "
                f"Assessment: {review.assessment} | "
                f"Evidence: {review.evidence} | "
                f"Improvement: {review.improvement}"
            )
        if pr.overall_notes:
            diagnostic_lines.append(f"  * Overall: {pr.overall_notes}")
        for edit in pr.proposed_edits:
            diagnostic_lines.append(
                f"  * ProposedEdit: {edit.action}@{edit.section}: {edit.content}"
            )

    if not diagnostic_lines:
        return []

    diagnostics_text = "\n".join(diagnostic_lines)

    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {
            "role": "user",
            "content": _DIAGNOSE_USER.format(
                agents_md=agents_md,
                diagnostics=diagnostics_text,
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
