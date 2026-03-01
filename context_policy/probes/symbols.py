"""Symbol index probe — extracts signatures and callers.

For each file, extracts: symbol name, kind (function/class/constant),
full signature (parameters + return type), and which other files
call/import that symbol.
"""
from __future__ import annotations

from typing import Any

from context_policy.probes.schema import ImportGraph, SymbolEntry, SymbolIndex

# Caps
MAX_SYMBOLS_PER_FILE = 10
MAX_FILES_IN_INDEX = 30


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_function_signature(node: Any, source: bytes) -> str:
    """Build ``def name(params) -> ret`` signature string."""
    name = ""
    params = ""
    ret = ""
    name_node = node.child_by_field_name("name")
    if name_node:
        name = _node_text(name_node, source)
    params_node = node.child_by_field_name("parameters")
    if params_node:
        params = _node_text(params_node, source)
    ret_node = node.child_by_field_name("return_type")
    if ret_node:
        ret = " -> " + _node_text(ret_node, source)
    return f"def {name}{params}{ret}"


def _extract_class_signature(node: Any, source: bytes) -> str:
    """Build ``class Name(bases)`` signature string."""
    name = ""
    bases = ""
    name_node = node.child_by_field_name("name")
    if name_node:
        name = _node_text(name_node, source)
    # superclasses come from argument_list child
    for child in node.children:
        if child.type == "argument_list":
            bases = _node_text(child, source)
            break
    return f"class {name}{bases}" if bases else f"class {name}"


def _extract_constant(node: Any, source: bytes) -> tuple[str, str] | None:
    """Extract a module-level constant assignment ``NAME = value``.

    Returns ``(name, signature)`` or *None*.
    """
    # expression_statement -> assignment
    if node.type != "expression_statement":
        return None
    for child in node.children:
        if child.type == "assignment":
            left = child.child_by_field_name("left")
            right = child.child_by_field_name("right")
            if left and left.type == "identifier":
                name = _node_text(left, source)
                # Only consider UPPER_CASE names as constants
                if name.isupper() or (name[0].isupper() and "_" in name):
                    rhs = _node_text(right, source) if right else "..."
                    # Truncate long RHS
                    if len(rhs) > 80:
                        rhs = rhs[:77] + "..."
                    return name, f"{name} = {rhs}"
    return None


def _extract_symbols_from_tree(
    tree: Any,
    source: bytes,
    rel_path: str,
) -> list[SymbolEntry]:
    """Extract top-level symbols from one file's tree."""
    entries: list[SymbolEntry] = []
    for node in tree.root_node.children:
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node, source)
                sig = _extract_function_signature(node, source)
                entries.append(SymbolEntry(
                    name=name, kind="function", signature=sig,
                    file=rel_path, used_in=[],
                ))
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node, source)
                sig = _extract_class_signature(node, source)
                entries.append(SymbolEntry(
                    name=name, kind="class", signature=sig,
                    file=rel_path, used_in=[],
                ))
        elif node.type == "decorated_definition":
            # Unwrap decorator to get the definition
            for child in node.children:
                if child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        name = _node_text(name_node, source)
                        sig = _extract_function_signature(child, source)
                        entries.append(SymbolEntry(
                            name=name, kind="function", signature=sig,
                            file=rel_path, used_in=[],
                        ))
                elif child.type == "class_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        name = _node_text(name_node, source)
                        sig = _extract_class_signature(child, source)
                        entries.append(SymbolEntry(
                            name=name, kind="class", signature=sig,
                            file=rel_path, used_in=[],
                        ))
        else:
            const = _extract_constant(node, source)
            if const:
                name, sig = const
                entries.append(SymbolEntry(
                    name=name, kind="constant", signature=sig,
                    file=rel_path, used_in=[],
                ))

    return entries[:MAX_SYMBOLS_PER_FILE]


def build_symbol_index(
    trees: dict[str, Any],
    source_map: dict[str, bytes],
    import_graph: ImportGraph,
) -> SymbolIndex:
    """Build a symbol index with caller cross-references.

    Args:
        trees: Mapping of relative-path → tree-sitter Tree.
        source_map: Mapping of relative-path → raw source bytes.
        import_graph: Import graph for cross-referencing callers.

    Returns:
        Populated ``SymbolIndex``.
    """
    # Determine which files to index — take the top files by importance
    # (hub files first, then alphabetical), capped at MAX_FILES_IN_INDEX
    hub_files = {h.file for h in import_graph.hubs}
    all_files = sorted(trees.keys())

    # Prioritize hub files, then the rest
    ordered: list[str] = []
    for f in all_files:
        if f in hub_files:
            ordered.insert(0, f)
        else:
            ordered.append(f)
    files_to_index = ordered[:MAX_FILES_IN_INDEX]

    all_entries: list[SymbolEntry] = []
    by_file: dict[str, list[SymbolEntry]] = {}

    for rel_path in files_to_index:
        tree = trees[rel_path]
        source = source_map.get(rel_path, b"")
        entries = _extract_symbols_from_tree(tree, source, rel_path)

        # Cross-reference: which files import this file?
        importers = set(import_graph.imported_by.get(rel_path, []))

        for entry in entries:
            entry.used_in = sorted(importers)

        by_file[rel_path] = entries
        all_entries.extend(entries)

    return SymbolIndex(entries=all_entries, by_file=by_file)
