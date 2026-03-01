"""Tree-sitter based static analysis probes for repository introspection.

Public API::

    from context_policy.probes import run_all_probes

    results = run_all_probes(repo_dir)
"""
from __future__ import annotations

from pathlib import Path

from context_policy.probes.clustering import build_clusters
from context_policy.probes.conventions import detect_conventions
from context_policy.probes.entrypoints import detect_entry_points
from context_policy.probes.imports import build_import_graph
from context_policy.probes.parser import parse_repo
from context_policy.probes.schema import ProbeResults
from context_policy.probes.symbols import build_symbol_index
from context_policy.probes.tests import detect_tests


def run_all_probes(repo_dir: Path) -> ProbeResults:
    """Run all deterministic probes against a checked-out repository.

    The probes are purely structural (tree-sitter + filesystem) and
    require no LLM calls.  Output is deterministic for a given repo
    snapshot.

    Args:
        repo_dir: Root of the checked-out repository.

    Returns:
        Aggregated ``ProbeResults`` containing import graph, symbol
        index, entry points, co-import clusters, test info, and
        detected conventions.
    """
    # Step 1: Parse all Python files
    trees = parse_repo(repo_dir)

    # Build source map (bytes) for symbol/import extraction
    source_map: dict[str, bytes] = {}
    for rel_path in trees:
        full = repo_dir / rel_path
        try:
            source_map[rel_path] = full.read_bytes()
        except OSError:
            source_map[rel_path] = b""

    # Step 2: Import graph
    import_graph = build_import_graph(trees, source_map)

    # Step 3: Symbol index (depends on import graph for callers)
    symbol_index = build_symbol_index(trees, source_map, import_graph)

    # Step 4: Entry points
    entry_points = detect_entry_points(trees, source_map)

    # Step 5: Co-import clustering (depends on import graph)
    clusters = build_clusters(import_graph)

    # Step 6: Test discovery
    test_info = detect_tests(repo_dir)

    # Step 7: Conventions
    conventions = detect_conventions(repo_dir, trees, source_map)

    return ProbeResults(
        repo_dir=str(repo_dir),
        imports=import_graph,
        symbols=symbol_index,
        entry_points=entry_points,
        clusters=clusters,
        tests=test_info,
        conventions=conventions,
    )


__all__ = ["run_all_probes", "ProbeResults"]
