"""Context generation for SWE-bench instances.

Generates free-form context via LLM, steered by a tunable context_prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from context_policy.llm.openai_compat import chat_completion
from context_policy.policy.state import default_policy


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"
    return text[: max_chars - 1] + "…"


# ---------------------------------------------------------------------------
# Free-form LLM context generation
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a context generator for a coding agent that fixes software bugs. "
    "Produce a concise, actionable markdown document. "
    "Do NOT output code patches — only context that helps the agent."
)

_USER_TEMPLATE = """\
## Task
Repository: {repo}
Commit: {commit}

### Issue
{problem_statement}

### Instructions
{context_prompt}

Character budget: {budget} characters. Stay within the budget."""


def generate_context(
    instance: dict,
    *,
    policy: dict,
    model: str,
    round_id: str | None = None,
    source_task_batch: str | None = None,
    timeout_s: int = 120,
) -> dict:
    """Generate context for a task instance using the LLM.

    Args:
        instance: Task dict with repo, base_commit, problem_statement.
        policy: Policy dict with context_prompt and total_char_budget.
        model: Model name for inference.
        round_id: Optional round identifier.
        source_task_batch: Optional task batch source path.
        timeout_s: LLM request timeout.

    Returns:
        Context dict with repo, commit, body, metadata.
    """
    repo = instance.get("repo", "unknown")
    commit = instance.get("base_commit", "unknown")
    problem = instance.get("problem_statement", "")
    budget = int(policy.get("total_char_budget", 3200))
    context_prompt = policy.get("context_prompt", default_policy()["context_prompt"])

    user_msg = _USER_TEMPLATE.format(
        repo=repo,
        commit=commit,
        problem_statement=problem[:2000],  # cap problem statement to leave room
        context_prompt=context_prompt,
        budget=budget,
    )

    body = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max(256, budget // 3),  # rough tokens ~ chars/4
        timeout_s=timeout_s,
    )

    body = truncate(body.strip(), budget)

    context: dict = {
        "repo": repo,
        "commit": commit,
        "char_budget_total": budget,
        "body": body,
    }
    if policy.get("policy_version"):
        context["policy_version"] = policy["policy_version"]
    if round_id:
        context["round_id"] = round_id
    if source_task_batch:
        context["source_task_batch"] = source_task_batch
    return context


def generate_context_dry_run(
    instance: dict,
    *,
    policy: dict,
    round_id: str | None = None,
    source_task_batch: str | None = None,
) -> dict:
    """Generate a placeholder context without calling the LLM (for testing)."""
    repo = instance.get("repo", "unknown")
    commit = instance.get("base_commit", "unknown")
    budget = int(policy.get("total_char_budget", 3200))

    body = (
        f"[dry-run context for {repo}@{commit[:12]}]\n"
        f"Policy version: {policy.get('policy_version', 'unknown')}"
    )

    context: dict = {
        "repo": repo,
        "commit": commit,
        "char_budget_total": budget,
        "body": body,
    }
    if policy.get("policy_version"):
        context["policy_version"] = policy["policy_version"]
    if round_id:
        context["round_id"] = round_id
    if source_task_batch:
        context["source_task_batch"] = source_task_batch
    return context


# ---------------------------------------------------------------------------
# Rendering / persistence
# ---------------------------------------------------------------------------

def render_markdown(context: dict) -> str:
    """Render context dict to markdown string."""
    lines = [
        "# Repo Context",
        "",
        f"**Repo:** {context['repo']}",
        f"**Commit:** {context['commit'][:12]}",
        "",
        context.get("body", ""),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_context(context: dict, json_path: Path, md_path: Path) -> None:
    """Write context to JSON and markdown files."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    md_content = render_markdown(context)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)
