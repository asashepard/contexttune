"""Apply structured edits to AGENTS.md via LLM merging.

Takes current AGENTS.md + proposed edits → merged AGENTS.md.
Re-enforces the 3,200-character budget on output.
"""
from __future__ import annotations

import re

from context_policy.kb.agents_md import AGENTS_MD_CHAR_BUDGET
from context_policy.llm.openai_compat import chat_completion
from context_policy.oracle.schema import Edit

_APPLY_SYSTEM = """\
You are an editor for AGENTS.md files — concise instruction documents
that guide a coding assistant's behavior on a specific repository.

You will be given the current AGENTS.md and a list of edits to apply.
Each edit specifies a section, an action (add/modify/strengthen/remove),
and content.

Rules:
- Apply ALL edits faithfully.
- "add" → insert the content into the specified section (create section if needed).
- "modify" → replace or rephrase the closest matching rule in that section.
- "strengthen" → make an existing rule more specific/forceful.
- "remove" → delete the matching rule from that section.
- Keep the final AGENTS.md under {char_budget} characters.
- Preserve the overall structure and formatting.
- Output ONLY the updated AGENTS.md. No commentary."""

_APPLY_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

EDITS TO APPLY:
{edits}

Output the updated AGENTS.MD."""


def apply_edits(
    agents_md: str,
    edits: list[Edit],
    model: str,
    *,
    timeout_s: int = 120,
) -> str:
    """Apply edits to AGENTS.md via LLM merging.

    Args:
        agents_md: Current AGENTS.md content.
        edits: List of structured edits to apply.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        Updated AGENTS.md text, capped at 3,200 characters.
    """
    if not edits:
        return agents_md

    edits_text = "\n".join(
        f"- [{e.action.upper()}] Section: {e.section} — {e.content}"
        for e in edits
    )

    messages = [
        {
            "role": "system",
            "content": _APPLY_SYSTEM.format(char_budget=AGENTS_MD_CHAR_BUDGET),
        },
        {
            "role": "user",
            "content": _APPLY_USER.format(
                agents_md=agents_md,
                edits=edits_text,
            ),
        },
    ]

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        timeout_s=timeout_s,
    )

    # Strip any markdown fences the LLM may wrap around the output
    result = raw.strip()
    result = re.sub(r"^```(?:markdown)?\s*", "", result)
    result = re.sub(r"\s*```$", "", result)

    # Enforce budget
    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = result[:AGENTS_MD_CHAR_BUDGET - 20] + "\n\n[... truncated]"

    return result
