"""Build a RepoKB deterministically from probe results.

No LLM calls — purely template-based rendering of structured probe
data into the KB sections.
"""
from __future__ import annotations

from pathlib import Path

from context_policy.kb.render import (
    render_architecture,
    render_context,
    render_conventions,
    render_symbol_map,
)
from context_policy.kb.schema import RepoKB
from context_policy.probes.schema import ProbeResults


def build_kb(
    repo: str,
    commit: str,
    probe_results: ProbeResults,
) -> RepoKB:
    """Build the RepoKB artifact from probe results.

    This is fully deterministic: same probe results → same KB.
    No LLM calls are made.

    Args:
        repo: Repository slug (e.g. "django/django").
        commit: Commit SHA the repo is checked out to.
        probe_results: Output of ``run_all_probes()``.

    Returns:
        A populated ``RepoKB`` at version 0.
    """
    architecture = render_architecture(
        probe_results.imports,
        probe_results.entry_points,
    )
    symbol_map = render_symbol_map(probe_results.symbols)
    context = render_context(probe_results.clusters, probe_results.tests)
    conventions = render_conventions(probe_results.conventions)

    return RepoKB(
        repo=repo,
        commit=commit,
        version=0,
        architecture=architecture,
        symbol_map=symbol_map,
        context=context,
        conventions=conventions,
    )
