"""Policy state for adaptive context generation."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CARD_ORDER = [
    "issue_focus",
    "fix_plan",
    "validation",
    "editing_norms",
]


def default_policy() -> dict:
    return {
        "policy_version": "v0",
        "description": "Deterministic baseline context policy.",
        "total_char_budget": 3200,
        "card_order": list(DEFAULT_CARD_ORDER),
        "cards": {
            "issue_focus": {"enabled": True, "max_chars": 1200, "priority": 0.98},
            "fix_plan": {"enabled": True, "max_chars": 900, "priority": 0.9},
            "validation": {"enabled": True, "max_chars": 700, "priority": 0.85},
            "editing_norms": {"enabled": True, "max_chars": 500, "priority": 0.75},
        },
        "llm_rewrite_instruction": (
            "Return JSON only. Tune card order, enabled flags, and max_chars to improve "
            "resolved rate while staying within total_char_budget."
        ),
    }


def load_policy(path: str | Path | None) -> dict:
    if path is None:
        return default_policy()
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    return json.loads(policy_path.read_text(encoding="utf-8"))


def write_policy(path: str | Path, policy: dict) -> None:
    policy_path = Path(path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
