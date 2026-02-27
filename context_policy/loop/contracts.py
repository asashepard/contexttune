"""Typed contracts for adaptive loop state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class TaskInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str | None = None
    environment_setup_commit: str | None = None
    source: str | None = None
    metadata: dict | None = None


@dataclass
class PolicyVersion:
    version: str
    policy_file: str


@dataclass
class RoundState:
    round_index: int
    round_id: str
    policy_version: str
    task_batch_file: str
    conditions: list[str]
    results: dict = field(default_factory=dict)


@dataclass
class LoopState:
    loop_id: str
    model: str
    runner: str
    created_at: str
    rounds: list[RoundState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def write_loop_state(path: Path, state: LoopState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def read_loop_state(path: Path) -> LoopState:
    data = json.loads(path.read_text(encoding="utf-8"))
    rounds = [RoundState(**row) for row in data.get("rounds", [])]
    return LoopState(
        loop_id=data["loop_id"],
        model=data["model"],
        runner=data["runner"],
        created_at=data.get("created_at", ""),
        rounds=rounds,
    )
