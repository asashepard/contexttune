"""Micro-test probe generation from RepoKB sections.

Parses KB sections to extract hubs, entry points, and conventions,
then generates probes with expected behaviors.  Each probe has an ID,
category, task (user prompt), and list of expected behaviors.

Probe categories:
- hub-safety: from architecture hubs
- entry-point: from architecture entries
- naming: from map conventions
- architecture: from KB structure / changed files
- harvested: from real prompts if available

Capped at 10 total probes, deduplicated by ID.
"""
from __future__ import annotations

import hashlib
import re

from context_policy.kb.schema import RepoKB
from context_policy.oracle.schema import Probe

MAX_PROBES = 10


def _make_id(category: str, *parts: str) -> str:
    """Generate a short deterministic probe ID."""
    raw = f"{category}:{'|'.join(parts)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def _extract_hub_files(kb: RepoKB) -> list[tuple[str, str, str]]:
    """Extract (file, in_degree, importers) tuples from architecture."""
    results: list[tuple[str, str, str]] = []
    for line in kb.architecture.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3 and parts[0] != "File" and "---" not in parts[0]:
            results.append((parts[0], parts[1], parts[2]))
    return results


def _extract_entry_points_from_kb(kb: RepoKB) -> list[tuple[str, str, str]]:
    """Extract (file, kind, classification) from architecture."""
    results: list[tuple[str, str, str]] = []
    in_ep = False
    for line in kb.architecture.splitlines():
        if "Entry Points" in line:
            in_ep = True
            continue
        if in_ep and line.startswith("#"):
            break
        if in_ep and line.startswith("|"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3 and parts[0] != "File" and "---" not in parts[0]:
                results.append((parts[0], parts[1], parts[2]))
    return results


def _extract_convention_items(kb: RepoKB) -> list[str]:
    """Extract convention bullet points."""
    items: list[str] = []
    for line in kb.conventions.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:])
    return items


def generate_probes(kb: RepoKB) -> list[Probe]:
    """Generate micro-test probes from the KB.

    Args:
        kb: The RepoKB artifact to derive probes from.

    Returns:
        List of ``Probe`` objects, capped at 10 and deduplicated by ID.
    """
    probes: list[Probe] = []
    seen_ids: set[str] = set()

    def _add(probe: Probe) -> None:
        if probe.id not in seen_ids and len(probes) < MAX_PROBES:
            seen_ids.add(probe.id)
            probes.append(probe)

    # ── hub-safety probes ─────────────────────────────────────
    hubs = _extract_hub_files(kb)
    for file, in_degree, importers in hubs[:3]:
        pid = _make_id("hub-safety", file)
        _add(Probe(
            id=pid,
            category="hub-safety",
            task=(
                f"I need to modify `{file}` to change its behavior. "
                f"What other files should I update or check, and what "
                f"tests should I run?"
            ),
            expected_behaviors=[
                f"Mentions at least one of the importing files: {importers}",
                "Recommends running relevant tests",
                f"Acknowledges that {file} is a hub with many dependents",
            ],
        ))

    # ── entry-point probes ────────────────────────────────────
    eps = _extract_entry_points_from_kb(kb)
    for file, kind, classification in eps[:2]:
        pid = _make_id("entry-point", file)
        _add(Probe(
            id=pid,
            category="entry-point",
            task=(
                f"I want to add a new {kind} endpoint similar to "
                f"those in `{file}`. What patterns should I follow?"
            ),
            expected_behaviors=[
                f"References the existing {kind} pattern in {file}",
                "Describes the correct decorator or registration pattern",
                "Mentions where to add associated tests",
            ],
        ))

    # ── naming probes ─────────────────────────────────────────
    conventions = _extract_convention_items(kb)
    for conv in conventions[:2]:
        if "docstring" in conv.lower() or "naming" in conv.lower() or "style" in conv.lower():
            pid = _make_id("naming", conv)
            _add(Probe(
                id=pid,
                category="naming",
                task=(
                    f"I'm creating a new module in this repository. "
                    f"Given the convention: '{conv}', what naming "
                    f"pattern should I follow?"
                ),
                expected_behaviors=[
                    f"References the convention: {conv}",
                    "Provides a concrete naming example",
                ],
            ))

    # ── architecture probes ───────────────────────────────────
    # General architecture probe based on overall KB structure
    if kb.architecture:
        pid = _make_id("architecture", kb.repo)
        _add(Probe(
            id=pid,
            category="architecture",
            task=(
                f"I need to fix a bug that changes the behavior of a "
                f"core component in {kb.repo}. How should I scope my "
                f"changes to minimize blast radius?"
            ),
            expected_behaviors=[
                "Identifies hub files that should not be refactored carelessly",
                "Suggests checking integration points",
                "Recommends targeted test runs rather than the full suite",
            ],
        ))

    # Additional architecture probe for test command/infrastructure
    if kb.context:
        pid = _make_id("architecture", "testing")
        _add(Probe(
            id=pid,
            category="architecture",
            task=(
                f"I've just made a change to {kb.repo}. "
                f"How do I verify it works correctly?"
            ),
            expected_behaviors=[
                "Mentions the correct test command",
                "References test directories or conftest files",
            ],
        ))

    return probes
