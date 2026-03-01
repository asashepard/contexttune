"""LLM-as-judge evaluator for micro-test probes.

Flow:
1. ``simulate_response``: send AGENTS.md as system prompt + probe task
   as user prompt → get simulated AI response.
2. ``judge_behaviors``: send a second LLM call evaluating the response
   against each expected behavior → PASS/FAIL verdicts.
3. ``evaluate_probe``: combines (1) + (2).
"""
from __future__ import annotations

import json
import re

from context_policy.llm.openai_compat import chat_completion
from context_policy.oracle.schema import BehaviorVerdict, Probe, ProbeResult

# 60,000-char budget for system prompt (KB + AGENTS.md content)
SYSTEM_PROMPT_CHAR_BUDGET = 60_000

_SIMULATE_SYSTEM = """\
You are a coding assistant helping with the repository described below.
Follow the AGENTS.md guidelines when answering.

{agents_md}"""

_JUDGE_SYSTEM = """\
You are an evaluator. You will be given:
1. A TASK that was posed to a coding assistant.
2. The assistant's RESPONSE.
3. A list of EXPECTED BEHAVIORS.

For each expected behavior, judge whether the response satisfies it.
Output a JSON array of objects, each with:
- "behavior": the expected behavior text
- "passed": true or false
- "reasoning": brief explanation (1-2 sentences)

Output ONLY the JSON array."""

_JUDGE_USER = """\
TASK:
{task}

RESPONSE:
{response}

EXPECTED BEHAVIORS:
{behaviors}

Judge each behavior as passed (true) or failed (false)."""


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
    return chat_completion(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        timeout_s=timeout_s,
    )


def judge_behaviors(
    task: str,
    response: str,
    expected_behaviors: list[str],
    model: str,
    *,
    timeout_s: int = 120,
) -> list[BehaviorVerdict]:
    """Evaluate a response against expected behaviors via LLM.

    Args:
        task: The original probe task.
        response: The simulated AI response.
        expected_behaviors: List of behavior strings to check.
        model: LLM model name.
        timeout_s: LLM call timeout.

    Returns:
        List of ``BehaviorVerdict`` (one per behavior).
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

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        timeout_s=timeout_s,
    )

    return _parse_verdicts(raw, expected_behaviors)


def _parse_verdicts(
    raw: str,
    expected_behaviors: list[str],
) -> list[BehaviorVerdict]:
    """Parse LLM JSON output into BehaviorVerdict objects."""
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
                # Fallback: treat all as failed
                return [
                    BehaviorVerdict(behavior=b, passed=False, reasoning="Parse error")
                    for b in expected_behaviors
                ]
        else:
            return [
                BehaviorVerdict(behavior=b, passed=False, reasoning="Parse error")
                for b in expected_behaviors
            ]

    if not isinstance(arr, list):
        return [
            BehaviorVerdict(behavior=b, passed=False, reasoning="Invalid format")
            for b in expected_behaviors
        ]

    verdicts: list[BehaviorVerdict] = []
    for i, b in enumerate(expected_behaviors):
        if i < len(arr) and isinstance(arr[i], dict):
            item = arr[i]
            verdicts.append(BehaviorVerdict(
                behavior=str(item.get("behavior", b)),
                passed=bool(item.get("passed", False)),
                reasoning=str(item.get("reasoning", "")),
            ))
        else:
            verdicts.append(BehaviorVerdict(
                behavior=b, passed=False, reasoning="Missing verdict",
            ))

    return verdicts


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
        ``ProbeResult`` with verdicts and pass rate.
    """
    response = simulate_response(agents_md, probe, model, timeout_s=timeout_s)
    verdicts = judge_behaviors(
        task=probe.task,
        response=response,
        expected_behaviors=probe.expected_behaviors,
        model=model,
        timeout_s=timeout_s,
    )

    passed = sum(1 for v in verdicts if v.passed)
    total = len(verdicts) if verdicts else 1
    pass_rate = passed / total

    return ProbeResult(
        probe_id=probe.id,
        category=probe.category,
        verdicts=verdicts,
        pass_rate=pass_rate,
    )
