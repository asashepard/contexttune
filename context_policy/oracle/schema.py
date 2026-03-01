"""Dataclasses for the LLM-as-judge oracle evaluator loop."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Probe:
    """A micro-test probe generated from the KB."""

    id: str
    category: str  # "hub-safety", "entry-point", "naming", "architecture", "harvested"
    task: str  # the user prompt to send
    expected_behaviors: list[str] = field(default_factory=list)


@dataclass
class BehaviorVerdict:
    """Result of judging one expected behavior against a response."""

    behavior: str
    passed: bool
    reasoning: str = ""


@dataclass
class ProbeResult:
    """Result of evaluating one probe."""

    probe_id: str
    category: str
    verdicts: list[BehaviorVerdict] = field(default_factory=list)
    pass_rate: float = 0.0


@dataclass
class Edit:
    """A structured edit to apply to AGENTS.md."""

    section: str  # which AGENTS.md section
    action: str  # "add", "modify", "strengthen", "remove"
    content: str  # the text to add/modify/etc.


@dataclass
class OracleConfig:
    """Configuration for the oracle evaluator loop."""

    repo: str
    commit: str
    model: str
    iterations: int = 5
    timeout_s: int = 120
    output_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OracleState:
    """Persistent state for the oracle evaluator loop."""

    repo: str
    best_version: int = 0
    best_pass_rate: float = 0.0
    completed_iterations: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OracleState:
        return cls(
            repo=d["repo"],
            best_version=d.get("best_version", 0),
            best_pass_rate=d.get("best_pass_rate", 0.0),
            completed_iterations=d.get("completed_iterations", 0),
            history=list(d.get("history", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> OracleState:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
