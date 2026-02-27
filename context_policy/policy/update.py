"""LLM-driven policy update for adaptive rounds."""
from __future__ import annotations

import json
import sys

from context_policy.llm.openai_compat import chat_completion
from context_policy.policy.state import default_policy


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in policy update response")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def _sanitize_policy(candidate: dict, fallback: dict, next_version: str) -> dict:
    """Sanitize an LLM-proposed policy, enforcing bounds."""
    base = default_policy()
    policy = {
        "policy_version": next_version,
        "description": candidate.get("description", fallback.get("description", "Adaptive policy")),
        "total_char_budget": int(candidate.get("total_char_budget", fallback.get("total_char_budget", base["total_char_budget"]))),
        "context_prompt": candidate.get("context_prompt", fallback.get("context_prompt", base["context_prompt"])),
    }

    # Enforce budget bounds
    if policy["total_char_budget"] < 800:
        policy["total_char_budget"] = 800
    if policy["total_char_budget"] > 6000:
        policy["total_char_budget"] = 6000

    # Enforce context_prompt is a non-empty string, cap length
    if not isinstance(policy["context_prompt"], str) or not policy["context_prompt"].strip():
        policy["context_prompt"] = base["context_prompt"]
    policy["context_prompt"] = policy["context_prompt"][:2000]

    return policy


def update_policy_with_llm(
    *,
    model: str,
    current_policy: dict,
    round_summary: dict,
    next_version: str,
    timeout_s: int = 120,
) -> dict:
    system = (
        "You are a policy optimizer for context generation instructions. "
        "Return JSON only with keys: description, total_char_budget, context_prompt. "
        "The context_prompt tells an LLM how to generate context for a coding agent. "
        "Tune the prompt wording, emphasis, and budget to improve the resolved rate."
    )
    user = (
        "Current policy:\n"
        f"{json.dumps(current_policy, indent=2)}\n\n"
        "Round summary:\n"
        f"{json.dumps(round_summary, indent=2)}\n\n"
        "Rewrite the policy to improve resolved rate while avoiding extreme changes. "
        "Focus on making context_prompt more specific and actionable."
    )

    try:
        response = chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1200,
            timeout_s=timeout_s,
        )
        candidate = _extract_json_object(response)
        return _sanitize_policy(candidate, current_policy, next_version)
    except Exception as exc:
        print(f"WARNING: policy update failed, reusing current policy: {exc}", file=sys.stderr)
        fallback = dict(current_policy)
        fallback["policy_version"] = next_version
        fallback["description"] = f"Fallback copy due to update error: {exc}"
        return fallback
