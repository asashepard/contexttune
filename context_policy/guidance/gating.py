"""Validation and gating utilities for RepoGuidance objects."""
from __future__ import annotations

import re
from pathlib import Path

from context_policy.guidance.schema import RepoGuidance


MAX_LINE_COUNT = 120
MIN_LINE_COUNT = 3


def extract_path_references(text: str) -> list[str]:
    """Extract plausible file/directory path references from guidance text.

    Looks for patterns like ``src/foo/bar.py`` or ``tests/`` that appear in
    the guidance.  Used to validate that guidance doesn't hallucinate paths.
    """
    # Match things that look like relative paths (contain / and don't start with http)
    pattern = r'(?<!\w)([a-zA-Z0-9_.][a-zA-Z0-9_./\-]*(?:\.\w+|/))'
    candidates = re.findall(pattern, text)
    # Filter out URLs, version strings, etc.
    paths = []
    for c in candidates:
        if c.startswith("http") or c.startswith("//"):
            continue
        # Must contain at least one /
        if "/" not in c:
            continue
        # Strip trailing dots
        c = c.rstrip(".")
        if len(c) > 2:
            paths.append(c)
    return paths


def validate_guidance(
    guidance: RepoGuidance,
    repo_dir: Path | None = None,
    *,
    strict_paths: bool = False,
) -> list[str]:
    """Validate a RepoGuidance and return list of warning strings.

    An empty list means the guidance is valid.

    Args:
        guidance: The guidance to validate.
        repo_dir: Optional checkout directory.  When provided *and*
            ``strict_paths`` is True, path references in the guidance
            are checked against the actual tree.
        strict_paths: If True and repo_dir is given, warn on every
            path reference that doesn't exist on disk.

    Returns:
        List of human-readable warning strings.
    """
    warnings: list[str] = []

    # Budget check
    if not guidance.is_within_budget():
        warnings.append(
            f"Guidance exceeds char budget: {guidance.char_count()} > {guidance.char_budget}"
        )

    # Line count bounds
    n = len(guidance.lines)
    if n < MIN_LINE_COUNT:
        warnings.append(f"Too few lines ({n} < {MIN_LINE_COUNT})")
    if n > MAX_LINE_COUNT:
        warnings.append(f"Too many lines ({n} > {MAX_LINE_COUNT})")

    # Empty lines check (not fatal, just informational)
    empty = sum(1 for l in guidance.lines if not l.strip())
    if empty > n // 3 and n > 6:
        warnings.append(f"{empty}/{n} lines are blank")

    # Path reference validation
    if strict_paths and repo_dir is not None:
        rendered = guidance.render()
        refs = extract_path_references(rendered)
        for ref in refs:
            target = repo_dir / ref.rstrip("/")
            if not target.exists():
                warnings.append(f"Path reference not found in repo: {ref}")

    return warnings


def truncate_to_budget(guidance: RepoGuidance) -> RepoGuidance:
    """Return a copy truncated to fit within char_budget.

    Lines are dropped from the end until the guidance fits.
    """
    if guidance.is_within_budget():
        return guidance

    lines = list(guidance.lines)
    while lines and len("\n".join(lines)) > guidance.char_budget:
        lines.pop()

    return guidance.copy(lines=lines)
