"""Build and write repo signals."""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from context_policy.signals.import_graph import build_import_graph
from context_policy.signals.py_index import build_py_index
from context_policy.signals.test_manifest import build_test_manifest
from context_policy.utils.ignore import should_ignore_dir, should_ignore_file


def _get_commit(repo_dir: Path) -> str:
    """Get current HEAD commit SHA.

    Args:
        repo_dir: Repository directory.

    Returns:
        Commit SHA string, or empty string if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return ""


def _build_tree(repo_dir: Path, max_depth: int = 4) -> list[dict]:
    """Build directory tree as list of entries.

    Args:
        repo_dir: Repository root.
        max_depth: Maximum depth to traverse.

    Returns:
        Sorted list of {"path": str, "type": "file"|"dir"} entries.
    """
    entries: list[dict] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            items = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return

        for item in items:
            name = item.name

            if item.is_dir():
                if should_ignore_dir(name):
                    continue
                rel_path = item.relative_to(repo_dir).as_posix()
                entries.append({"path": rel_path, "type": "dir"})
                _walk(item, depth + 1)
            else:
                if should_ignore_file(name):
                    continue
                rel_path = item.relative_to(repo_dir).as_posix()
                entries.append({"path": rel_path, "type": "file"})

    _walk(repo_dir, 1)

    # Sort by path for determinism
    entries.sort(key=lambda e: e["path"])
    return entries


def _compute_hot_paths(import_graph: dict, py_index: dict) -> list[str]:
    """Compute hot paths: top 10 files by import in-degree.

    Args:
        import_graph: Import graph with nodes and edges.
        py_index: Python index with modules.

    Returns:
        Sorted list of up to 10 file paths (most imported first).
    """
    edges = import_graph.get("edges", [])
    if not edges:
        return []

    # Count in-degree (how many times each module is imported)
    in_degree: Counter[str] = Counter()
    for src, dst in edges:
        in_degree[dst] += 1

    # Get top 10 by count, then alphabetically for ties
    top_modules = sorted(in_degree.items(), key=lambda x: (-x[1], x[0]))[:10]

    # Convert module names to file paths
    hot_paths = []
    for module_name, _ in top_modules:
        # module.submodule -> module/submodule.py
        file_path = module_name.replace(".", "/") + ".py"
        hot_paths.append(file_path)

    return hot_paths


def build_signals(repo_dir: Path, repo: str, commit: str) -> dict:
    """Build all signals for a repository.

    Args:
        repo_dir: Path to checked-out repository.
        repo: Repository name in "owner/name" format.
        commit: Commit SHA.

    Returns:
        Dict with all signal data (deterministic, no timestamps).
    """
    # Build components
    py_index = build_py_index(repo_dir)
    import_graph = build_import_graph(repo_dir, py_index["modules"])
    test_manifest = build_test_manifest(repo_dir)
    tree_entries = _build_tree(repo_dir, max_depth=4)
    hot_paths = _compute_hot_paths(import_graph, py_index)

    return {
        "repo": repo,
        "commit": commit,
        "tree": {
            "max_depth": 4,
            "entry_count": len(tree_entries),
            "entries": tree_entries,
        },
        "py_index": py_index,
        "import_graph": import_graph,
        "test_manifest": test_manifest,
        "hot_paths": hot_paths,
    }


def write_signals(signals: dict, out_path: Path) -> None:
    """Write signals to JSON file.

    Args:
        signals: Signals dict to write.
        out_path: Output file path. Parent dirs created if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
