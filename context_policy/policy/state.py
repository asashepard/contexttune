"""Policy state for adaptive context generation."""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_CONTEXT_PROMPT = """\
Given the issue description, produce a concise context document that will help a coding agent fix the bug. Include:
1. A focused summary of the issue (symptoms, expected vs actual behaviour).
2. A concrete fix plan: which files/functions to change and how.
3. Validation steps: which tests to run and what to check.
4. Editing norms: make the smallest correct change, follow existing style, avoid unrelated edits.
Stay within the character budget. Be specific, not generic."""


def default_policy() -> dict:
    return {
        "policy_version": "v0",
        "description": "Baseline context policy.",
        "total_char_budget": 3200,
        "context_prompt": _DEFAULT_CONTEXT_PROMPT,
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
