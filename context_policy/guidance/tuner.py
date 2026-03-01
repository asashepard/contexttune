"""Tuner module — bridges old hill-climbing API to the new oracle loop.

The old ``run_tuning_loop`` using SWE-Smith tasks is deprecated.
Use ``context_policy.oracle.loop.run_oracle_loop`` directly instead.

This module is kept for backward-compatible imports only.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from context_policy.guidance.schema import DEFAULT_CHAR_BUDGET, RepoGuidance


# ── configuration ──────────────────────────────────────────────


MAX_TUNING_ITERATIONS = 20


@dataclass
class TuningConfig:
    """Tuning configuration (legacy — mostly unused).

    Prefer ``OracleConfig`` from ``context_policy.oracle.schema``.
    """

    repo: str
    commit: str
    model: str

    # Oracle tuning budget
    iterations: int = 5

    # Runner settings
    timeout_s: int = 120

    # Guidance constraints
    char_budget: int = DEFAULT_CHAR_BUDGET

    # Output paths
    output_dir: str = ""  # set by caller

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if self.iterations > MAX_TUNING_ITERATIONS:
            raise ValueError(
                f"iterations={self.iterations} exceeds cap {MAX_TUNING_ITERATIONS}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


# ── tuning state (for resume — now in oracle.schema) ───────────


@dataclass
class TuningState:
    """Legacy tuning state.  Use ``OracleState`` instead."""

    repo: str
    best_version: int = 0
    best_score: float = 0.0
    history: list[dict] = field(default_factory=list)
    completed_iterations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TuningState:
        return cls(
            repo=d["repo"],
            best_version=d.get("best_version", 0),
            best_score=d.get("best_score", 0.0),
            history=list(d.get("history", [])),
            completed_iterations=d.get("completed_iterations", 0),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> TuningState:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── main tuning loop (delegates to oracle) ─────────────────────


def run_tuning_loop(config: TuningConfig) -> RepoGuidance:
    """Run tuning for one repository.

    .. deprecated::
        This is a compatibility wrapper.  Use
        ``context_policy.oracle.loop.run_oracle_loop`` directly.

    Args:
        config: Populated ``TuningConfig``.

    Returns:
        The best ``RepoGuidance`` found.
    """
    warnings.warn(
        "run_tuning_loop is deprecated. Use oracle.loop.run_oracle_loop.",
        DeprecationWarning,
        stacklevel=2,
    )
    from context_policy.oracle.loop import run_oracle_loop
    from context_policy.oracle.schema import OracleConfig

    oc = OracleConfig(
        repo=config.repo,
        commit=config.commit,
        model=config.model,
        iterations=config.iterations,
        timeout_s=config.timeout_s,
        output_dir=config.output_dir,
    )
    _kb, best = run_oracle_loop(oc)
    return best
