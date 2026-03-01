"""Dataclasses for tree-sitter probe results."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HubModule:
    """A high-in-degree module in the import graph."""

    file: str
    importers: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    in_degree: int = 0


@dataclass
class ImportGraph:
    """Directed import/dependency graph for a repository."""

    # file -> list of modules it imports
    edges: dict[str, list[str]] = field(default_factory=dict)
    # file -> list of files that import it
    imported_by: dict[str, list[str]] = field(default_factory=dict)
    # hub modules sorted by in-degree descending
    hubs: list[HubModule] = field(default_factory=list)


@dataclass
class SymbolEntry:
    """One extracted symbol with caller information."""

    name: str
    kind: str  # "function", "class", "constant"
    signature: str
    file: str
    used_in: list[str] = field(default_factory=list)


@dataclass
class SymbolIndex:
    """Symbol index for a repository."""

    entries: list[SymbolEntry] = field(default_factory=list)
    # file -> list of entries in that file
    by_file: dict[str, list[SymbolEntry]] = field(default_factory=dict)


@dataclass
class EntryPoint:
    """A detected entry point in the repository."""

    file: str
    kind: str  # "if_main", "route", "cli", "asgi_wsgi"
    classification: str  # "runtime", "tooling", "barrel"
    confidence: str  # "high", "medium", "low"
    detail: str = ""  # e.g. decorator text or pattern matched


@dataclass
class EntryPoints:
    """Collection of detected entry points."""

    entries: list[EntryPoint] = field(default_factory=list)


@dataclass
class FileCluster:
    """A group of files that share importers (co-import cluster)."""

    id: int
    files: list[str] = field(default_factory=list)
    shared_importers: list[str] = field(default_factory=list)
    score: float = 0.0  # average co-import score within cluster


@dataclass
class ImportChain:
    """A sequence of files where each imports the next."""

    files: list[str] = field(default_factory=list)
    length: int = 0


@dataclass
class ClusterResults:
    """Co-import clustering results."""

    clusters: list[FileCluster] = field(default_factory=list)
    chains: list[ImportChain] = field(default_factory=list)
    integrations: list[str] = field(default_factory=list)  # bridge files


@dataclass
class TestInfo:
    """Detected test infrastructure."""

    test_command: str = "pytest"
    test_dirs: list[str] = field(default_factory=list)
    conftest_paths: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)


@dataclass
class Conventions:
    """Detected coding conventions."""

    docstring_style: str = ""  # "google", "numpy", "sphinx", "unknown"
    type_hint_prevalence: float = 0.0  # 0.0-1.0
    import_style: str = ""  # description of import ordering
    linter_configs: list[str] = field(default_factory=list)  # detected config files
    detected_patterns: list[str] = field(default_factory=list)  # free-form


@dataclass
class ProbeResults:
    """Aggregate result of all probes for one repository."""

    repo_dir: str = ""
    imports: ImportGraph = field(default_factory=ImportGraph)
    symbols: SymbolIndex = field(default_factory=SymbolIndex)
    entry_points: EntryPoints = field(default_factory=EntryPoints)
    clusters: ClusterResults = field(default_factory=ClusterResults)
    tests: TestInfo = field(default_factory=TestInfo)
    conventions: Conventions = field(default_factory=Conventions)
