"""Baseline deterministic context generation from signals."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Character budgets (approx tokens = chars // 4)
TOTAL_CHAR_BUDGET = 3200  # ~800 tokens
CARD_CHAR_BUDGET = 900    # max per card


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters allowed.

    Returns:
        Truncated text with "…" suffix if truncated.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"
    return text[: max_chars - 1] + "…"


def _get_top_prefixes(modules: list[str], n: int = 5) -> list[str]:
    """Get top N module prefixes by frequency.

    Args:
        modules: List of module names (e.g., ["pkg.sub.mod", ...]).
        n: Number of prefixes to return.

    Returns:
        Sorted list of top prefixes.
    """
    prefixes: Counter[str] = Counter()
    for mod in modules:
        parts = mod.split(".")
        if parts and parts[0]:
            prefixes[parts[0]] += 1
    # Sort by count desc, then alphabetically
    top = sorted(prefixes.items(), key=lambda x: (-x[1], x[0]))[:n]
    return [p[0] for p in top]


def _get_depth1_dirs(tree_entries: list[dict]) -> list[str]:
    """Get top-level directories from tree entries.

    Args:
        tree_entries: List of {"path": str, "type": "file"|"dir"}.

    Returns:
        Sorted list of top-level directory names.
    """
    dirs = []
    for entry in tree_entries:
        if entry["type"] == "dir":
            path = entry["path"]
            # Depth 1 = no "/" in path
            if "/" not in path:
                dirs.append(path)
    return sorted(dirs)


def _build_repo_identity_card(signals: dict) -> dict:
    """Build repo identity card."""
    nodes = signals.get("import_graph", {}).get("nodes", [])
    tree_entries = signals.get("tree", {}).get("entries", [])

    prefixes = _get_top_prefixes(nodes, n=5)
    top_dirs = _get_depth1_dirs(tree_entries)[:8]
    module_count = len(nodes)

    lines = [
        f"- Repository: {signals.get('repo', 'unknown')}",
        f"- Commit: {signals.get('commit', 'unknown')[:12]}",
    ]
    if prefixes:
        lines.append(f"- Primary packages: {', '.join(prefixes)}")
    if top_dirs:
        lines.append(f"- Top-level dirs: {', '.join(top_dirs)}")
    lines.append(f"- Module count: {module_count}")

    body = "\n".join(lines)

    return {
        "id": "repo_identity",
        "title": "Repository Identity",
        "body": truncate(body, CARD_CHAR_BUDGET),
        "evidence": ["import_graph:nodes", "tree:depth1"],
        "confidence": 0.7,
    }


def _build_tests_howto_card(signals: dict) -> dict:
    """Build tests howto card."""
    test_manifest = signals.get("test_manifest", {})
    commands = test_manifest.get("suggested_commands", [])
    test_dirs = test_manifest.get("test_dirs", [])
    notes = test_manifest.get("notes", [])

    lines = []
    if commands:
        lines.append("Run tests with:")
        for cmd in commands[:3]:
            lines.append(f"  - `{cmd}`")
    if test_dirs:
        lines.append(f"Test directories: {', '.join(test_dirs)}")
    if notes:
        lines.append(f"Config: {', '.join(notes)}")

    body = "\n".join(lines) if lines else "No test configuration detected."

    return {
        "id": "tests_howto",
        "title": "Testing",
        "body": truncate(body, CARD_CHAR_BUDGET),
        "evidence": ["test_manifest:*"],
        "confidence": 0.8,
    }


def _build_editing_norms_card() -> dict:
    """Build editing norms card (global constant)."""
    body = """- Make the smallest change that fixes the issue.
- Avoid unrelated formatting changes.
- Follow existing code style and patterns.
- Prefer editing existing functions over adding new modules.
- Keep imports minimal and consistent with existing style."""

    return {
        "id": "editing_norms",
        "title": "Editing Guidelines",
        "body": body,
        "evidence": ["global_constant"],
        "confidence": 0.9,
    }


def _build_routing_guidance_card(signals: dict) -> dict:
    """Build routing guidance card."""
    hot_paths = signals.get("hot_paths", [])
    nodes = signals.get("import_graph", {}).get("nodes", [])
    test_dirs = signals.get("test_manifest", {}).get("test_dirs", [])

    prefixes = _get_top_prefixes(nodes, n=5)

    lines = []

    # Only include hot_paths line if non-empty
    if hot_paths:
        paths_str = ", ".join(hot_paths[:8])
        lines.append(f"- Start with these frequently-imported files: {paths_str}")

    if prefixes:
        prefixes_str = ", ".join(prefixes)
        lines.append(f"- For stack traces, look under: {prefixes_str}")

    if test_dirs:
        lines.append(f"- Tests are under: {', '.join(test_dirs)}")

    # Top modules as entrypoints
    if nodes:
        top_mods = sorted(nodes)[:8]
        lines.append(f"- Common modules: {', '.join(top_mods)}")

    body = "\n".join(lines) if lines else "No routing guidance available."

    return {
        "id": "routing_guidance",
        "title": "Where to Look",
        "body": truncate(body, CARD_CHAR_BUDGET),
        "evidence": ["hot_paths", "import_graph:nodes", "test_manifest:test_dirs"],
        "confidence": 0.6,
    }


def _build_pitfalls_card(signals: dict) -> dict:
    """Build pitfalls card with simple heuristics."""
    tree_entries = signals.get("tree", {}).get("entries", [])
    test_dirs = signals.get("test_manifest", {}).get("test_dirs", [])

    top_dirs = _get_depth1_dirs(tree_entries)

    lines = []

    # Check for src layout
    has_src = "src" in top_dirs
    # Check for package dir at top level (non-src, non-test, non-docs)
    pkg_dirs = [d for d in top_dirs if d not in ("src", "tests", "test", "docs", "doc", "examples")]
    if has_src and pkg_dirs:
        lines.append(f"- Note: Both `src/` and `{pkg_dirs[0]}/` exist. Check which is the main package.")

    # No test dirs warning
    if not test_dirs:
        lines.append("- No standard test directory detected. Tests may be elsewhere.")

    body = "\n".join(lines) if lines else "No specific pitfalls detected."

    return {
        "id": "pitfalls",
        "title": "Potential Pitfalls",
        "body": truncate(body, CARD_CHAR_BUDGET),
        "evidence": ["tree:*", "test_manifest:test_dirs"],
        "confidence": 0.5,
    }


def build_baseline_context(signals: dict, repo: str, commit: str) -> dict:
    """Build baseline context from signals.

    Args:
        signals: Signals dict from signals.json.
        repo: Repository in owner/name format.
        commit: Commit SHA.

    Returns:
        Context dict conforming to schema/context_schema.json.
    """
    # Fixed order for determinism
    cards = [
        _build_repo_identity_card(signals),
        _build_tests_howto_card(signals),
        _build_editing_norms_card(),
        _build_routing_guidance_card(signals),
        _build_pitfalls_card(signals),
    ]

    # Enforce total budget by truncating routing_guidance if needed
    total_chars = sum(len(c["body"]) for c in cards)
    if total_chars > TOTAL_CHAR_BUDGET:
        excess = total_chars - TOTAL_CHAR_BUDGET
        # Find routing_guidance and truncate it further
        for card in cards:
            if card["id"] == "routing_guidance":
                current_len = len(card["body"])
                new_len = max(50, current_len - excess)
                card["body"] = truncate(card["body"], new_len)
                break

    return {
        "repo": repo,
        "commit": commit,
        "char_budget_total": TOTAL_CHAR_BUDGET,
        "cards": cards,
    }


def render_markdown(context: dict) -> str:
    """Render context dict to stable markdown.

    Args:
        context: Context dict with cards.

    Returns:
        Markdown string (ends with newline).
    """
    lines = [
        "# Repo Context",
        "",
        f"**Repo:** {context['repo']}",
        f"**Commit:** {context['commit'][:12]}",
        "",
    ]

    for card in context["cards"]:
        lines.append(f"## {card['title']}")
        lines.append("")
        lines.append(card["body"])
        lines.append("")
        evidence_str = ", ".join(card["evidence"])
        lines.append(f"*Evidence: {evidence_str} | Confidence: {card['confidence']}*")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_context(context: dict, json_path: Path, md_path: Path) -> None:
    """Write context to JSON and markdown files.

    Args:
        context: Context dict.
        json_path: Path for context.json.
        md_path: Path for context.md.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON (pretty, sorted keys for determinism)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    # Write markdown
    md_content = render_markdown(context)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)
