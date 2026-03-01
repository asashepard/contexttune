"""RepoKB — the structured knowledge base artifact for one repository.

A single unified context block containing architecture, symbol map,
context (clusters/tests), and conventions sections.  Rendered
deterministically from probe results with no LLM calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepoKB:
    """Structured knowledge base for one repository snapshot.

    Each section is a pre-rendered text block with line-based budgets:
    - architecture: 200 lines max
    - symbol_map: 300 lines max
    - context: 200 lines max
    - conventions: unlimited (typically small)
    """

    repo: str
    commit: str
    version: int = 0

    architecture: str = ""  # hubs, blast radius, entry points
    symbol_map: str = ""    # signature tables with callers
    context: str = ""       # co-import clusters, chains, integrations, tests
    conventions: str = ""   # detected patterns

    def render(self) -> str:
        """Concatenate all sections into a single KB block."""
        sections: list[str] = []
        if self.architecture:
            sections.append(f"## Architecture\n\n{self.architecture}")
        if self.symbol_map:
            sections.append(f"## Symbol Map\n\n{self.symbol_map}")
        if self.context:
            sections.append(f"## Context\n\n{self.context}")
        if self.conventions:
            sections.append(f"## Conventions\n\n{self.conventions}")
        return "\n\n".join(sections)

    def render_truncated(self, char_budget: int = 60_000) -> str:
        """Render with truncation for LLM system prompt injection.

        The 60,000-character budget applies only when sending KB content
        to an LLM, not to the stored artifact.
        """
        full = self.render()
        if len(full) <= char_budget:
            return full
        return full[:char_budget - 20] + "\n\n[KB truncated]"

    # ── serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "commit": self.commit,
            "version": self.version,
            "architecture": self.architecture,
            "symbol_map": self.symbol_map,
            "context": self.context,
            "conventions": self.conventions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RepoKB:
        return cls(
            repo=d["repo"],
            commit=d["commit"],
            version=int(d.get("version", 0)),
            architecture=d.get("architecture", ""),
            symbol_map=d.get("symbol_map", ""),
            context=d.get("context", ""),
            conventions=d.get("conventions", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RepoKB:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
