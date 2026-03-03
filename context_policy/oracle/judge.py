"""LLM-as-judge evaluator for micro-test probes.

Flow:
1. ``simulate_response``: send AGENTS.md as system prompt + probe task
    as user prompt → get simulated AI response.
2. ``review_probe``: evaluate behavior quality and propose edits.
3. ``evaluate_probe``: combines (1) + (2).
"""
from __future__ import annotations

import json
import re

from context_policy.llm.openai_compat import chat_completion
from context_policy.oracle.schema import BehaviorReview, Edit, Probe, ProbeResult

# 60,000-char budget for system prompt (KB + AGENTS.md content)
SYSTEM_PROMPT_CHAR_BUDGET = 60_000

_SIMULATE_SYSTEM = """\
You are a coding assistant helping with the repository described below.
Follow the AGENTS.md guidelines when answering.

{agents_md}"""

_JUDGE_SYSTEM = """\
You are an evaluator/editor for AGENTS.md quality.
You will be given a TASK, the assistant RESPONSE, and EXPECTED BEHAVIORS.

Your outputs should focus on making future behavior better, not pass/fail scoring.
Assess each behavior with one of: "strong", "partial", "missing".

Return a JSON object with this exact shape:
{
    "behavior_reviews": [
        {
            "behavior": "...",
            "assessment": "strong|partial|missing",
            "evidence": "short evidence from response",
            "improvement": "what AGENTS.md should add/change"
        }
    ],
    "proposed_edits": [
        {"section": "...", "action": "add|modify|strengthen|remove", "content": "..."}
    ],
    "overall_notes": "short summary"
}

Rules:
- Prefer concrete, testable edits over vague advice.
- You may propose edits even when behaviors look strong.
- Output ONLY valid JSON."""

_JUDGE_USER = """\
TASK:
{task}

RESPONSE:
{response}

EXPECTED BEHAVIORS:
{behaviors}

Produce behavior_reviews and proposed_edits JSON."""


def simulate_response(
    agents_md: str,
    probe: Probe,
    model: str,
    *,
    timeout_s: int = 120,
) -> str:
    """Send AGENTS.md as system prompt + probe task as user prompt.

    Args:
        agents_md: The current AGENTS.md content.
        probe: The micro-test probe with task.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        The simulated AI response string.
    """
    system = _SIMULATE_SYSTEM.format(
        agents_md=agents_md[:SYSTEM_PROMPT_CHAR_BUDGET],
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": probe.task},
    ]
    print(
        f"[oracle.judge] simulate_response: task_chars={len(probe.task)} "
        f"agents_md_chars={len(agents_md)}",
        flush=True,
    )
    response = chat_completion(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        timeout_s=timeout_s,
    )
    print(f"[oracle.judge] simulate_response: response_chars={len(response)}", flush=True)
    return response


def review_probe(
    task: str,
    response: str,
    expected_behaviors: list[str],
    model: str,
    *,
    timeout_s: int = 120,
) -> tuple[list[BehaviorReview], list[Edit], str]:
    """Evaluate behavior quality and propose edits via LLM.

    Args:
        task: The original probe task.
        response: The simulated AI response.
        expected_behaviors: List of behavior strings to check.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        Tuple of (behavior reviews, proposed edits, overall notes).
    """
    behaviors_text = "\n".join(
        f"{i+1}. {b}" for i, b in enumerate(expected_behaviors)
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _JUDGE_USER.format(
                task=task,
                response=response,
                behaviors=behaviors_text,
            ),
        },
    ]

    print(
        f"[oracle.judge] review_probe: behaviors={len(expected_behaviors)} "
        f"response_chars={len(response)}",
        flush=True,
    )
    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        timeout_s=timeout_s,
    )
    print(f"[oracle.judge] review_probe: raw_chars={len(raw)}", flush=True)

    return _parse_review(raw, expected_behaviors)


def _parse_review(
    raw: str,
    expected_behaviors: list[str],
) -> tuple[list[BehaviorReview], list[Edit], str]:
    """Parse LLM JSON output into reviews/edits."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        print("[oracle.judge] review parse failed; attempting object extraction", flush=True)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                print("[oracle.judge] extracted object parse failed; using fallback review", flush=True)
                return _fallback_reviews(expected_behaviors)
        else:
            print("[oracle.judge] no JSON object found; using fallback review", flush=True)
            return _fallback_reviews(expected_behaviors)

    if not isinstance(obj, dict):
        print("[oracle.judge] parsed payload is not dict; using fallback review", flush=True)
        return _fallback_reviews(expected_behaviors)

    reviews_raw = obj.get("behavior_reviews", [])
    edits_raw = obj.get("proposed_edits", [])
    overall_notes = str(obj.get("overall_notes", "")).strip()

    reviews: list[BehaviorReview] = []
    for i, b in enumerate(expected_behaviors):
        if i < len(reviews_raw) and isinstance(reviews_raw[i], dict):
            item = reviews_raw[i]
            assessment = str(item.get("assessment", "partial")).strip().lower()
            if assessment not in {"strong", "partial", "missing"}:
                assessment = "partial"
            reviews.append(BehaviorReview(
                behavior=str(item.get("behavior", b)),
                assessment=assessment,
                evidence=str(item.get("evidence", "")).strip(),
                improvement=str(item.get("improvement", "")).strip(),
            ))
            continue

        reviews.append(BehaviorReview(
            behavior=b,
            assessment="missing",
            evidence="Review missing",
            improvement="Add explicit instruction for this behavior to AGENTS.md.",
        ))

    valid_actions = {"add", "modify", "strengthen", "remove"}
    edits: list[Edit] = []
    for item in edits_raw if isinstance(edits_raw, list) else []:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "")).strip() or "General"
        action = str(item.get("action", "add")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if action not in valid_actions:
            action = "add"
        edits.append(Edit(section=section, action=action, content=content))

    print(
        f"[oracle.judge] review parsed: reviews={len(reviews)} edits={len(edits)}",
        flush=True,
    )
    return reviews, edits, overall_notes


def _fallback_reviews(expected_behaviors: list[str]) -> tuple[list[BehaviorReview], list[Edit], str]:
    reviews = [
        BehaviorReview(
            behavior=b,
            assessment="missing",
            evidence="Failed to parse reviewer output.",
            improvement="Add explicit AGENTS.md guidance for this behavior.",
        )
        for b in expected_behaviors
    ]
    return reviews, [], "Parse error in reviewer output"


def evaluate_probe(
    agents_md: str,
    probe: Probe,
    model: str,
    *,
    timeout_s: int = 120,
) -> ProbeResult:
    """Full evaluation of one probe: simulate + judge.

    Args:
        agents_md: Current AGENTS.md content.
        probe: The micro-test probe.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        ``ProbeResult`` with behavior reviews and edit proposals.
    """
    response = simulate_response(agents_md, probe, model, timeout_s=timeout_s)
    behavior_reviews, proposed_edits, overall_notes = review_probe(
        task=probe.task,
        response=response,
        expected_behaviors=probe.expected_behaviors,
        model=model,
        timeout_s=timeout_s,
    )

    return ProbeResult(
        probe_id=probe.id,
        task=probe.task,
        response=response,
        behavior_reviews=behavior_reviews,
        proposed_edits=proposed_edits,
        overall_notes=overall_notes,
    )
