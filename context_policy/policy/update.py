"""LLM-driven policy update for adaptive rounds."""
from __future__ import annotations

import json

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
    base = default_policy()
    policy = {
        "policy_version": next_version,
        "description": candidate.get("description", fallback.get("description", "Adaptive policy")),
        "total_char_budget": int(candidate.get("total_char_budget", fallback.get("total_char_budget", base["total_char_budget"]))),
        "card_order": candidate.get("card_order", fallback.get("card_order", base["card_order"])),
        "cards": candidate.get("cards", fallback.get("cards", base["cards"])),
        "llm_rewrite_instruction": fallback.get("llm_rewrite_instruction", base["llm_rewrite_instruction"]),
    }

    if policy["total_char_budget"] < 800:
        policy["total_char_budget"] = 800
    if policy["total_char_budget"] > 6000:
        policy["total_char_budget"] = 6000

    if not isinstance(policy["card_order"], list) or not policy["card_order"]:
        policy["card_order"] = list(base["card_order"])

    if not isinstance(policy["cards"], dict) or not policy["cards"]:
        policy["cards"] = dict(base["cards"])

    for card_id, cfg in policy["cards"].items():
        if not isinstance(cfg, dict):
            policy["cards"][card_id] = dict(base["cards"].get(card_id, {"enabled": True, "max_chars": 900, "priority": 0.5}))
            continue
        cfg.setdefault("enabled", True)
        cfg.setdefault("max_chars", 900)
        cfg.setdefault("priority", 0.5)
        cfg["max_chars"] = max(100, min(2000, int(cfg["max_chars"])))
        cfg["priority"] = max(0.0, min(1.0, float(cfg["priority"])))

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
        "You are a policy optimizer for context cards. Return JSON only with keys: "
        "description, total_char_budget, card_order, cards."
    )
    user = (
        "Current policy:\n"
        f"{json.dumps(current_policy, indent=2)}\n\n"
        "Round summary:\n"
        f"{json.dumps(round_summary, indent=2)}\n\n"
        "Rewrite the policy to improve resolved rate while avoiding extreme changes."
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
        fallback = dict(current_policy)
        fallback["policy_version"] = next_version
        fallback["description"] = f"Fallback copy due to update error: {exc}"
        return fallback
